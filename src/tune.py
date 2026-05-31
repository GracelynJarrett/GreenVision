"""Tuning entrypoint for GreenVision Phase 2 experiments.

Runs one tuning profile at a time using profile definitions in `config.yaml`.
Each profile controls:
- Backbone group unfreezing strategy
- Augmentation policy
- Learning rate and regularization values
- MLflow run naming and checkpoint output

This module preserves project constants and artifact conventions while keeping
training, tuning, and validation separated by responsibility.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
from copy import deepcopy
from typing import Any, Dict, Optional

import mlflow
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader

from src.dataset import make_datasets_with_transforms, save_class_mapping, get_transforms_from_config
from src.model import DiseaseClassifier
from src.utils import (
    EarlyStopping,
    get_device,
    save_best_checkpoint,
    setup_logging,
    train_one_epoch,
    validate_one_epoch,
)


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration and return dictionary payload."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def configure_mlflow(cfg: Dict[str, Any]) -> None:
    """Configure MLflow tracking settings from configuration."""
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri") or cfg["paths"].get("mlflow_artifacts_dir", "mlruns")
    experiment_name = mlflow_cfg.get("experiment_name", cfg["project"].get("name", "GreenVision"))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def _build_profile(cfg: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
    """Merge tuning defaults with a selected tuning profile."""
    tuning_cfg = cfg.get("tuning", {})
    profiles = tuning_cfg.get("profiles", {})
    defaults = tuning_cfg.get("defaults", {})

    if profile_name not in profiles:
        raise KeyError(f"Unknown tuning profile '{profile_name}'. Available: {sorted(profiles.keys())}")

    # Build an isolated profile object so per-run edits never mutate config state.
    profile = deepcopy(defaults)
    profile.update(deepcopy(profiles[profile_name]))

    # Use base augmentation as fallback if profile augmentation is missing.
    if "augmentation" not in profile:
        profile["augmentation"] = deepcopy(cfg.get("augmentation", {}))

    return profile


def build_tuned_dataloaders(cfg: Dict[str, Any], profile: Dict[str, Any]):
    """Create tuned train/val/test dataloaders based on profile augmentation and batch size."""
    data_dir = cfg["paths"]["data_dir"]
    image_size = cfg["project"].get("image_size", 224)
    splits = cfg.get("splits", {})

    # Profile-level augmentation overrides base augmentation for that specific run.
    augmentation_cfg = profile.get("augmentation", cfg.get("augmentation", {}))
    train_transform = get_transforms_from_config(augmentation_cfg, image_size=image_size, train=True)
    eval_transform = get_transforms_from_config(augmentation_cfg, image_size=image_size, train=False)

    train_ds, val_ds, test_ds, class_to_idx = make_datasets_with_transforms(
        data_dir,
        train_transform=train_transform,
        val_transform=eval_transform,
        test_transform=eval_transform,
        train_frac=splits.get("train", 0.75),
        val_frac=splits.get("validation", 0.15),
        test_frac=splits.get("test", 0.10),
        seed=cfg["project"].get("seed", 42),
        validate_images=splits.get("validation_images", True),
    )

    save_class_mapping(class_to_idx, path=cfg["paths"].get("class_to_idx", "models/class_to_idx.json"))

    # Batch size is profile-specific to let each freezing strategy use its own memory budget.
    batch_size = int(profile.get("batch_size", cfg["training"].get("batch_size", 16)))
    num_workers = int(cfg["training"].get("num_workers", 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_to_idx


def build_model_and_optimizer(
    cfg: Dict[str, Any],
    profile: Dict[str, Any],
    device: torch.device,
    profile_name: str,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, nn.Module, Optional[torch.optim.lr_scheduler.LRScheduler]]:
    """Create tuned model, optimizer, criterion, and optional scheduler."""
    model = DiseaseClassifier(
        num_classes=cfg["project"].get("num_classes", 38),
        dropout_rate=float(profile.get("dropout", cfg["model"].get("dropout", 0.5))),
    ).to(device)

    total_groups = int(cfg.get("tuning", {}).get("default_total_backbone_groups", 8))
    trainable_groups = profile.get("trainable_groups", list(range(1, total_groups + 1)))

    # M5 intentionally tunes the entire backbone, while M2-M4 only unfreeze selected groups.
    if profile_name == "M5":
        model.unfreeze_backbone()
    else:
        model.set_trainable_backbone_groups(trainable_groups, total_groups=total_groups)

    learning_rate = float(profile.get("learning_rate", 5e-5))
    weight_decay = float(profile.get("weight_decay", cfg["training"].get("regularization", {}).get("weight_decay", 1e-4)))
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = None
    warmup_epochs = int(profile.get("warmup_epochs", 1))
    epochs = int(profile.get("epochs", 8))
    min_lr_scale = float(profile.get("min_lr_scale", 0.1))
    # Warmup + cosine decay keeps initial updates gentle for transfer learning stability.
    if epochs > 1 and warmup_epochs >= 0:
        warmup_steps = max(1, warmup_epochs)
        cosine_steps = max(1, epochs - warmup_steps)

        # Warmup moves smoothly from 10% of base LR to 100% of base LR.
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(1.0, 0.1 + 0.9 * ((epoch + 1) / max(1, warmup_steps))),
        )
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=learning_rate * min_lr_scale)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion, scheduler


def run_tuning_profile(
    cfg: Dict[str, Any],
    profile_name: str,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Run one tuning profile end-to-end and return best/test metrics."""
    profile = _build_profile(cfg, profile_name)
    device = get_device(use_gpu=True)
    if device.type != "cuda":
        raise RuntimeError(
            "GPU is required for tuning, but CUDA is not available. "
            "Activate the correct environment and verify your CUDA-enabled PyTorch install."
        )
    logger.info("GPU-only tuning confirmed on device: %s", device)

    train_loader, val_loader, test_loader, class_to_idx = build_tuned_dataloaders(cfg, profile)
    model, optimizer, criterion, scheduler = build_model_and_optimizer(cfg, profile, device, profile_name)

    epochs = int(profile.get("epochs", 8))
    if epochs <= 0:
        raise ValueError(f"Profile {profile_name} has invalid epochs={epochs}")

    early_stopping = EarlyStopping(
        patience=int(profile.get("early_stopping_patience", cfg["training"].get("regularization", {}).get("early_stopping_patience", 8))),
        min_delta=float(profile.get("early_stopping_min_delta", cfg["training"].get("regularization", {}).get("early_stopping_min_delta", 0.0))),
    )

    best_val_loss = math.inf
    best_metrics: Dict[str, float] = {}

    # Keep checkpoint placement aligned with existing project artifact conventions.
    model_dir = cfg["paths"].get("model_dir", "models")
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, f"{profile_name.lower()}_best.pth")

    logger.info("Starting tuning profile %s for %d epochs", profile_name, epochs)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        mlflow.log_metrics(
            {
                f"{profile_name}_train_loss": float(train_metrics["loss"]),
                f"{profile_name}_train_accuracy": float(train_metrics["accuracy"]),
                f"{profile_name}_val_loss": float(val_metrics["loss"]),
                f"{profile_name}_val_accuracy": float(val_metrics["accuracy"]),
            },
            step=epoch,
        )

        logger.info(
            "%s epoch %d/%d - train_loss=%.4f train_acc=%.2f val_loss=%.4f val_acc=%.2f",
            profile_name,
            epoch,
            epochs,
            train_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["accuracy"],
        )

        # Save only when validation improves, mirroring the base training best-checkpoint flow.
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_metrics = {
                "best_epoch": float(epoch),
                "best_val_loss": float(val_metrics["loss"]),
                "best_val_accuracy": float(val_metrics["accuracy"]),
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
            }
            save_best_checkpoint(
                model,
                checkpoint_path,
                epoch=epoch,
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
                extra_metadata={
                    "profile": profile_name,
                    "train_loss": float(train_metrics["loss"]),
                    "train_accuracy": float(train_metrics["accuracy"]),
                },
            )
            if os.path.isfile(checkpoint_path):
                mlflow.log_artifact(checkpoint_path, artifact_path=f"checkpoints/{profile_name}")

        # Step the scheduler once per epoch after optimizer updates.
        if scheduler is not None:
            scheduler.step()

        if early_stopping.step(float(val_metrics["loss"])):
            logger.info("Early stopping triggered for %s at epoch %d", profile_name, epoch)
            break

    if os.path.isfile(checkpoint_path):
        model.load_checkpoint(checkpoint_path)

    test_metrics = validate_one_epoch(model, test_loader, criterion, device)
    mlflow.log_metrics(
        {
            f"{profile_name}_test_loss": float(test_metrics["loss"]),
            f"{profile_name}_test_accuracy": float(test_metrics["accuracy"]),
        }
    )

    logger.info(
        "%s test metrics - loss=%.4f accuracy=%.2f samples=%d",
        profile_name,
        test_metrics["loss"],
        test_metrics["accuracy"],
        int(test_metrics["samples"]),
    )

    result = {
        "train_samples": float(len(train_loader.dataset)),
        "val_samples": float(len(val_loader.dataset)),
        "test_samples": float(len(test_loader.dataset)),
        "class_count": float(len(class_to_idx)),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
    }
    result.update(best_metrics)
    return result


def append_metrics_log(log_path: str, run_name: str, profile_name: str, metrics: Dict[str, float], run_id: str) -> None:
    """Append a tuning profile metrics section to the metrics log markdown file."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    # Append mode keeps prior base/tuning entries intact for chronological experiment history.
    lines = [
        "",
        f"## Tuning Model {profile_name} Metrics",
        "",
        f"**Run date:** 2026-05-30  ",
        f"**Run name:** `{run_name}`  ",
        f"**MLflow run ID:** `{run_id}`",
        "",
        "### Final Metrics",
        "",
        "| Split | Metric | Value |",
        "| --- | --- | ---: |",
        f"| {profile_name} | Best validation loss | {metrics.get('best_val_loss', 0.0):.4f} |",
        f"| {profile_name} | Best validation accuracy | {metrics.get('best_val_accuracy', 0.0):.2f}% |",
        f"| {profile_name} | Final training loss | {metrics.get('train_loss', 0.0):.4f} |",
        f"| {profile_name} | Final training accuracy | {metrics.get('train_accuracy', 0.0):.2f}% |",
        f"| Test | Loss | {metrics.get('test_loss', 0.0):.4f} |",
        f"| Test | Accuracy | {metrics.get('test_accuracy', 0.0):.2f}% |",
        "",
        "### Notes",
        "",
        f"- Dataset size used for tuning: {int(metrics.get('train_samples', 0))} train images, {int(metrics.get('val_samples', 0))} validation images, {int(metrics.get('test_samples', 0))} test images.",
        f"- Class mapping preserved with {int(metrics.get('class_count', 0))} classes.",
        f"- Best checkpoint saved to models/{profile_name.lower()}_best.pth.",
    ]

    with open(log_path, "a", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main(config_path: str, profile_name: str) -> None:
    """Run a single tuning profile from configuration."""
    cfg = load_config(config_path)

    logs_dir = cfg["paths"].get("logs_dir", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"tune_{profile_name.lower()}.log")
    setup_logging(log_file=log_file, level=getattr(logging, cfg.get("logging", {}).get("level", "INFO")))

    logger = logging.getLogger(__name__)
    configure_mlflow(cfg)

    # Prefix-based naming keeps tuning runs easy to filter in MLflow UI.
    run_prefix = cfg.get("mlflow", {}).get("run_name_tuning_prefix", "phase2_tuning")
    run_name = f"{run_prefix}_{profile_name}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("tuning_profile", profile_name)
        mlflow.log_param("config_path", config_path)

        metrics = run_tuning_profile(cfg, profile_name, logger)
        append_metrics_log(
            log_path=os.path.join("doc", "Model_metrix_Log.md"),
            run_name=run_name,
            profile_name=profile_name,
            metrics=metrics,
            run_id=run.info.run_id,
        )

    logger.info("Completed tuning profile %s", profile_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GreenVision tuning profile")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--profile", required=True, help="Tuning profile name (M2, M3, M4, M5)")
    args = parser.parse_args()
    main(args.config, args.profile)
