"""Phase 4 smoke test for the test entrypoint.

This test uses lightweight stand-ins for the model and data to verify the
test orchestration flow works end-to-end without pulling from the real dataset.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

import src.test as test


def test_test_main_smoke(tmp_path: Path, monkeypatch) -> None:
    """Smoke test the test entrypoint with lightweight stand-ins."""
    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "models" / "phase2_best.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    cfg = {
        "project": {
            "name": "GreenVision",
            "seed": 42,
            "image_size": 224,
            "num_classes": 38,
        },
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "class_to_idx": str(tmp_path / "models" / "class_to_idx.json"),
            "checkpoint_phase2": str(checkpoint_path),
            "logs_dir": str(tmp_path / "logs"),
        },
        "training": {
            "batch_size": 8,
            "num_workers": 0,
        },
        "validation": {
            "batch_size": 4,
            "num_workers": 0,
        },
        "testing": {
            "batch_size": 4,
            "num_workers": 0,
        },
        "splits": {
            "train": 0.75,
            "validation": 0.15,
            "test": 0.10,
            "validation_images": False,
        },
        "logging": {
            "level": "INFO",
            "log_to_file": False,
        },
        "model": {
            "dropout": 0.5,
        },
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # Generate a 38-class mapping so the test satisfies the project contract.
    class_to_idx = {f"class_{index:02d}": index for index in range(38)}
    expected_report = {
        "overall_accuracy": 95.0,
        "total_samples": 10,
        "confusion_matrix": [[1, 0], [0, 1]],
        "class_names": ["class_00", "class_01"],
        "per_class_accuracy": [],
        "top_confusions": [],
        "misclassified_examples": [],
    }
    captured = {}

    monkeypatch.setattr(test, "configure_test_logging", lambda cfg: logging.getLogger("test-test"))
    monkeypatch.setattr(test, "get_device", lambda use_gpu=True: "cpu")
    monkeypatch.setattr(test, "load_class_mapping", lambda path: class_to_idx)
    monkeypatch.setattr(test, "build_test_loader", lambda cfg: ([1, 2], class_to_idx))
    monkeypatch.setattr(test, "load_model", lambda cfg, checkpoint, device: object())
    monkeypatch.setattr(test, "evaluate_test_set", lambda model, test_loader, device, class_names: expected_report)
    monkeypatch.setattr(test, "save_test_report", lambda report, output_path: captured.update({"report": report, "output_path": output_path}))

    report = test.main(str(config_path))

    assert report == expected_report
    assert captured["report"] == expected_report
    assert captured["output_path"] == str(tmp_path / "logs" / "test_report.json")
