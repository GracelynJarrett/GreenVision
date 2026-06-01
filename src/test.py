"""Phase 4 testing entrypoint for GreenVision.

This module loads the selected checkpoint, runs deterministic evaluation on
the held-out test split, and writes the final certification report.
The test flow is intentionally separate from validation so Phase 4 stays
focused on final model certification only.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from typing import Any, Dict, List, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import load_class_mapping, make_datasets
from src.model import DiseaseClassifier
from src.utils import get_device, setup_logging


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration from disk and return it as a dictionary."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def configure_test_logging(cfg: Dict[str, Any]) -> logging.Logger:
    """Set up console and file logging for test runs."""
    logs_dir = cfg["paths"].get("logs_dir", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Keep test logs in the same project log directory so the branch has
    # one predictable place to inspect final certification output.
    log_file = os.path.join(logs_dir, "test.log") if cfg.get("logging", {}).get("log_to_file", True) else None
    setup_logging(log_file=log_file, level=getattr(logging, cfg.get("logging", {}).get("level", "INFO")))
    return logging.getLogger(__name__)


def build_test_loader(cfg: Dict[str, Any]) -> tuple[DataLoader, Dict[str, int]]:
    """Build the deterministic test DataLoader and return class mapping."""
    data_dir = cfg["paths"]["data_dir"]
    image_size = cfg["project"].get("image_size", 224)
    splits = cfg.get("splits", {})

    # Build the full split set so indices stay consistent; we only use the test portion.
    _, _, test_ds, class_to_idx = make_datasets(
        data_dir,
        image_size=image_size,
        train_frac=splits.get("train", 0.75),
        val_frac=splits.get("validation", 0.15),
        test_frac=splits.get("test", 0.10),
        seed=cfg["project"].get("seed", 42),
        validate_images=splits.get("validation_images", True),
    )

    # Use test-specific batch settings when available for independent tuning,
    # but fall back to validation settings if testing config is not present.
    batch_size = int(cfg.get("testing", {}).get("batch_size", cfg.get("validation", {}).get("batch_size", cfg["training"].get("batch_size", 16))))
    num_workers = int(cfg.get("testing", {}).get("num_workers", cfg.get("validation", {}).get("num_workers", cfg["training"].get("num_workers", 0))))

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader, class_to_idx


def resolve_checkpoint_path(cfg: Dict[str, Any], checkpoint_path: str | None) -> str:
    """Return the checkpoint path to use for testing."""
    if checkpoint_path:
        return checkpoint_path

    # Default to the selected fine-tuned checkpoint so Phase 4 can run even
    # when the caller does not pass a path explicitly.
    fallback_path = cfg["paths"].get("checkpoint_phase2")
    if fallback_path:
        return fallback_path

    raise ValueError("No checkpoint path provided and no fallback checkpoint_phase2 configured")


def load_model(cfg: Dict[str, Any], checkpoint_path: str, device: torch.device) -> DiseaseClassifier:
    """Create the model, load weights, and move it to the selected device."""
    num_classes = int(cfg["project"].get("num_classes", 38))
    dropout_rate = float(cfg.get("model", {}).get("dropout", 0.5))

    # Recreate the same classifier shape used during training so the saved
    # weights can load without any architecture mismatch.
    model = DiseaseClassifier(num_classes=num_classes, dropout_rate=dropout_rate)
    model.to(device)
    model.load_checkpoint(checkpoint_path)
    model.eval()
    return model


def _class_names_by_index(class_to_idx: Dict[str, int]) -> List[str]:
    """Return class names ordered by their numeric label index."""
    if not class_to_idx:
        raise ValueError("class_to_idx mapping is empty")

    # Sort by the stored indices so report rows line up with the model's
    # output order instead of relying on dictionary insertion order.
    ordered = sorted(class_to_idx.items(), key=lambda item: item[1])
    class_names = [name for name, _ in ordered]

    expected_indices = list(range(len(class_names)))
    actual_indices = [index for _, index in ordered]
    if actual_indices != expected_indices:
        raise ValueError(f"class_to_idx indices must be contiguous from 0 to {len(class_names) - 1}")

    return class_names


def evaluate_test_set(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Run inference on the test split and build the final certification report."""
    total_samples = 0
    total_correct = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    misclassified_examples: List[Dict[str, Any]] = []
    misclassification_counter: Counter[tuple[int, int]] = Counter()

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            # Move each test batch to the active device, but keep the
            # forward pass in eval/no-grad mode because Phase 4 is certification only.
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1).values

            batch_correct = (predictions == targets).sum().item()
            batch_size = targets.size(0)

            total_samples += int(batch_size)
            total_correct += int(batch_correct)

            batch_targets = targets.tolist()
            batch_predictions = predictions.tolist()
            batch_confidences = confidences.tolist()

            y_true.extend(batch_targets)
            y_pred.extend(batch_predictions)

            for sample_index, (target, prediction, confidence) in enumerate(
                zip(batch_targets, batch_predictions, batch_confidences)
            ):
                if target != prediction:
                    misclassification_counter[(target, prediction)] += 1
                    if len(misclassified_examples) < 25:
                        misclassified_examples.append(
                            {
                                "batch_index": int(batch_index),
                                "sample_index": int(sample_index),
                                "true_index": int(target),
                                "pred_index": int(prediction),
                                "true_class": class_names[target],
                                "pred_class": class_names[prediction],
                                "confidence": float(confidence),
                            }
                        )

    # Build the confusion matrix after the full pass so we have the complete
    # set of predictions before computing per-class summaries.
    num_classes = len(class_names)
    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, prediction in zip(y_true, y_pred):
        confusion_matrix[target][prediction] += 1

    per_class_accuracy: List[Dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        row_total = sum(confusion_matrix[class_index])
        row_correct = confusion_matrix[class_index][class_index]
        accuracy = (row_correct / row_total * 100.0) if row_total else 0.0
        per_class_accuracy.append(
            {
                "class_index": int(class_index),
                "class_name": class_name,
                "correct": int(row_correct),
                "total": int(row_total),
                "accuracy": float(accuracy),
            }
        )

    top_confusions = [
        {
            "true_index": int(true_index),
            "true_class": class_names[true_index],
            "pred_index": int(pred_index),
            "pred_class": class_names[pred_index],
            "count": int(count),
        }
        for (true_index, pred_index), count in misclassification_counter.most_common(20)
    ]

    overall_accuracy = (total_correct / total_samples * 100.0) if total_samples else 0.0
    return {
        "overall_accuracy": float(overall_accuracy),
        "total_samples": int(total_samples),
        "confusion_matrix": confusion_matrix,
        "class_names": list(class_names),
        "per_class_accuracy": per_class_accuracy,
        "top_confusions": top_confusions,
        "misclassified_examples": misclassified_examples,
    }


def save_test_report(report: Dict[str, Any], output_path: str) -> None:
    """Persist the test report as JSON for the final certification record."""
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # JSON keeps the test artifact easy to diff, inspect, and archive as
    # the final certification record for the selected model.
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


def main(config_path: str, checkpoint_path: str | None = None) -> Dict[str, Any]:
    """Run the Phase 4 test workflow and return the final test report."""
    cfg = load_config(config_path)
    logger = configure_test_logging(cfg)

    device = get_device(use_gpu=True)
    resolved_checkpoint = resolve_checkpoint_path(cfg, checkpoint_path)

    if not os.path.isfile(resolved_checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {resolved_checkpoint}")

    # Reuse the saved mapping artifact so test reports use the exact
    # same label order as training, validation, and inference.
    class_to_idx = load_class_mapping(cfg["paths"].get("class_to_idx", "models/class_to_idx.json"))
    class_names = _class_names_by_index(class_to_idx)

    expected_classes = int(cfg["project"].get("num_classes", 38))
    if len(class_names) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, found {len(class_names)}")

    test_loader, _ = build_test_loader(cfg)
    model = load_model(cfg, resolved_checkpoint, device)

    logger.info("Loaded test checkpoint: %s", resolved_checkpoint)
    logger.info("Test loader ready with %d batches", len(test_loader))

    # Run the final certification pass and persist the results as an artifact.
    report = evaluate_test_set(model, test_loader, device, class_names)

    logs_dir = cfg["paths"].get("logs_dir", "logs")
    report_path = os.path.join(logs_dir, "test_report.json")
    save_test_report(report, report_path)

    logger.info(
        "Test complete - accuracy=%.2f samples=%d report=%s",
        report["overall_accuracy"],
        report["total_samples"],
        report_path,
    )
    logger.info("Top test misclassifications: %s", report["top_confusions"][:5])

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GreenVision Phase 4 checkpoint")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. Defaults to paths.checkpoint_phase2 in config.",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint)
