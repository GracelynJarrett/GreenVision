"""Training orchestration for GreenVision (Phase 1 entrypoint).

Section 1: configuration, logging, data loading, and model setup.

This module is intentionally small and delegates functionality to support
modules (`src/dataset.py`, `src/model.py`, `src/utils.py`). Subsequent
sections (training loop, MLflow logging, checkpointing) will be added
incrementally.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
from typing import Any, Dict, Optional

import yaml
import torch
import mlflow
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import make_datasets, save_class_mapping
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
    """Load YAML configuration from `path` and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def configure_mlflow(cfg: Dict[str, Any]) -> None:
    """Configure MLflow tracking and experiment selection from config."""
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri") or cfg["paths"].get("mlflow_artifacts_dir", "mlruns")
    experiment_name = mlflow_cfg.get("experiment_name", cfg["project"].get("name", "GreenVision"))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_config_to_mlflow(cfg: Dict[str, Any]) -> None:
    """Log a compact subset of the training configuration to MLflow."""
    mlflow.log_params(
        {
            "project_name": cfg["project"].get("name", "GreenVision"),
            "seed": cfg["project"].get("seed", 42),
            "image_size": cfg["project"].get("image_size", 224),
            "num_classes": cfg["project"].get("num_classes", 38),
            "batch_size": cfg["training"].get("batch_size", 16),
            "phase1_epochs": cfg["training"].get("phase1", {}).get("epochs", 0),
            "phase1_learning_rate": cfg["training"].get("phase1", {}).get("learning_rate", 0.0),
            "phase2_epochs": cfg["training"].get("phase2", {}).get("epochs", 0),
            "phase2_learning_rate": cfg["training"].get("phase2", {}).get("learning_rate", 0.0),
            "mlflow_ui_port": cfg.get("mlflow", {}).get("ui_port", 5000),
        }
    )


def build_dataloaders(cfg: Dict[str, Any]):
    """Create train/val/test datasets and DataLoaders using configuration.

    Returns (train_loader, val_loader, test_loader, class_to_idx)
    """
    data_dir = cfg["paths"]["data_dir"]
    img_size = cfg["project"].get("image_size", 224)
    splits = cfg.get("splits", {})

    train_ds, val_ds, test_ds, class_to_idx = make_datasets(
        data_dir,
        image_size=img_size,
        train_frac=splits.get("train", 0.75),
        val_frac=splits.get("validation", 0.15),
        test_frac=splits.get("test", 0.10),
        seed=cfg["project"].get("seed", 42),
        validate_images=splits.get("validation_images", True),
    )

    # Persist class mapping for serving
    save_class_mapping(class_to_idx, path=cfg["paths"].get("class_to_idx", "models/class_to_idx.json"))

    batch_size = cfg["training"].get("batch_size", 16)
    num_workers = cfg["training"].get("num_workers", 0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_to_idx


def build_model_and_optimizer(cfg: Dict[str, Any], device: torch.device):
    """Instantiate the model, loss, and Phase 1 optimizer."""
    num_classes = cfg["project"].get("num_classes", 38)
    model = DiseaseClassifier(num_classes=num_classes, dropout_rate=cfg["model"].get("dropout", 0.5))
    model.to(device)

    if cfg["training"].get("phase1", {}).get("freeze_backbone", True):
        model.freeze_backbone()

    # Phase 1 optimizer: only train parameters with requires_grad=True
    lr = cfg["training"]["phase1"].get("learning_rate", 1e-3)
    weight_decay = cfg["training"]["phase1"].get("weight_decay", 1e-4)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion


def build_phase_optimizer(cfg: Dict[str, Any], model: torch.nn.Module, phase_name: str) -> torch.optim.Optimizer:
    """Build an optimizer for the requested training phase."""
    phase_cfg = cfg["training"][phase_name]
    learning_rate = phase_cfg.get("learning_rate", 1e-3)
    weight_decay = phase_cfg.get("weight_decay", 1e-4)
    return torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def run_training_phase(
    *,
    cfg: Dict[str, Any],
    phase_name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    checkpoint_path: str,
    logger: logging.Logger,
) -> Optional[Dict[str, float]]:
    """Run one training phase and persist the best checkpoint.

    The phase runner keeps phase-specific orchestration in one place so the
    entrypoint stays small. It trains for the configured number of epochs,
    evaluates on the validation set after each epoch, saves the best model,
    and applies early stopping when validation loss stops improving.
    """
    phase_cfg = cfg["training"].get(phase_name, {})
    epochs = int(phase_cfg.get("epochs", 0))
    if epochs <= 0:
        logger.info("Skipping %s because epochs <= 0", phase_name)
        return None

    early_stopping = EarlyStopping(
        patience=int(cfg["training"].get("regularization", {}).get("early_stopping_patience", 10)),
        min_delta=float(cfg["training"].get("regularization", {}).get("early_stopping_min_delta", 0.0)),
    )

    best_val_loss = math.inf
    best_metrics: Dict[str, float] = {}

    logger.info("Starting %s for %d epochs", phase_name, epochs)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        mlflow.log_metrics(
            {
                f"{phase_name}_train_loss": float(train_metrics["loss"]),
                f"{phase_name}_train_accuracy": float(train_metrics["accuracy"]),
                f"{phase_name}_val_loss": float(val_metrics["loss"]),
                f"{phase_name}_val_accuracy": float(val_metrics["accuracy"]),
            },
            step=epoch,
        )

        logger.info(
            "%s epoch %d/%d - train_loss=%.4f train_acc=%.2f val_loss=%.4f val_acc=%.2f",
            phase_name,
            epoch,
            epochs,
            train_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["accuracy"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = {
                "epoch": float(epoch),
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
            }

            save_best_checkpoint(
                model,
                checkpoint_path,
                epoch=epoch,
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
                extra_metadata={
                    "train_loss": float(train_metrics["loss"]),
                    "train_accuracy": float(train_metrics["accuracy"]),
                },
            )

            if os.path.isfile(checkpoint_path):
                mlflow.log_artifact(checkpoint_path, artifact_path=f"checkpoints/{phase_name}")

        if early_stopping.step(val_metrics["loss"]):
            logger.info("Early stopping triggered for %s at epoch %d", phase_name, epoch)
            break

    if os.path.isfile(checkpoint_path):
        model.load_checkpoint(checkpoint_path)

    if best_metrics:
        logger.info(
            "%s best checkpoint: epoch=%d val_loss=%.4f val_acc=%.2f",
            phase_name,
            int(best_metrics["epoch"]),
            best_metrics["val_loss"],
            best_metrics["val_accuracy"],
        )

    return best_metrics or None


def evaluate_test_set(
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Evaluate the final model on the held-out test set."""
    metrics = validate_one_epoch(model, test_loader, criterion, device)
    logger.info(
        "Test evaluation - loss=%.4f accuracy=%.2f samples=%d",
        metrics["loss"],
        metrics["accuracy"],
        int(metrics["samples"]),
    )
    return metrics


def log_final_artifacts(cfg: Dict[str, Any], logger: logging.Logger) -> None:
    """Log key training artifacts to MLflow after training finishes."""
    artifact_paths = [
        cfg["paths"].get("class_to_idx", "models/class_to_idx.json"),
        cfg["paths"].get("checkpoint_phase1", "models/phase1_best.pth"),
        cfg["paths"].get("checkpoint_phase2", "models/phase2_best.pth"),
    ]

    for artifact_path in artifact_paths:
        if artifact_path and os.path.isfile(artifact_path):
            mlflow.log_artifact(artifact_path)
            logger.info("Logged MLflow artifact: %s", artifact_path)


def main(config_path: str):
    cfg = load_config(config_path)

    # Setup logging according to config
    logs_dir = cfg["paths"].get("logs_dir", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "train.log") if cfg.get("logging", {}).get("log_to_file", True) else None
    setup_logging(log_file=log_file, level=getattr(logging, cfg.get("logging", {}).get("level", "INFO")))

    logger = logging.getLogger(__name__)
    logger.info("Loaded config: %s", config_path)

    device = get_device(use_gpu=True)

    configure_mlflow(cfg)

    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(cfg)

    model, optimizer, criterion = build_model_and_optimizer(cfg, device)

    logger.info("Data loaders and model ready. Train batches: %d", len(train_loader))
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    with mlflow.start_run(run_name=cfg.get("mlflow", {}).get("run_name_phase1", "greenvision-training")):
        log_config_to_mlflow(cfg)

        phase1_checkpoint = cfg["paths"].get("checkpoint_phase1", "models/phase1_best.pth")
        phase1_metrics = run_training_phase(
            cfg=cfg,
            phase_name="phase1",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            checkpoint_path=phase1_checkpoint,
            logger=logger,
        )

        if phase1_metrics is not None:
            logger.info("Phase 1 complete. Best validation accuracy: %.2f", phase1_metrics["val_accuracy"])

        phase2_cfg = cfg["training"].get("phase2", {})
        phase2_epochs = int(phase2_cfg.get("epochs", 0))
        if phase2_epochs > 0:
            model.unfreeze_backbone()
            phase2_optimizer = build_phase_optimizer(cfg, model, "phase2")
            phase2_checkpoint = cfg["paths"].get("checkpoint_phase2", "models/phase2_best.pth")
            phase2_metrics = run_training_phase(
                cfg=cfg,
                phase_name="phase2",
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=phase2_optimizer,
                device=device,
                checkpoint_path=phase2_checkpoint,
                logger=logger,
            )

            if phase2_metrics is not None:
                logger.info("Phase 2 complete. Best validation accuracy: %.2f", phase2_metrics["val_accuracy"])

        test_batches = len(test_loader)
        logger.info("Test loader ready with %d batches. Class mapping saved for %d classes.", test_batches, len(class_to_idx))

        test_metrics = evaluate_test_set(model, test_loader, criterion, device, logger)
        mlflow.log_metrics(
            {
                "test_loss": float(test_metrics["loss"]),
                "test_accuracy": float(test_metrics["accuracy"]),
            }
        )
        logger.info(
            "Final test metrics recorded: loss=%.4f accuracy=%.2f",
            test_metrics["loss"],
            test_metrics["accuracy"],
        )

        log_final_artifacts(cfg, logger)

    # Further training orchestration (phase training loops, MLflow, early stopping)
    # will be added in subsequent sections.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GreenVision - Phase 1")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
