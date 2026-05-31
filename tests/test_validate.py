from __future__ import annotations

import logging
from pathlib import Path

import yaml

import src.validate as validate


def test_validate_main_smoke(tmp_path: Path, monkeypatch) -> None:
    """Smoke test the validation entrypoint with lightweight stand-ins."""
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

    class_to_idx = {f"class_{index:02d}": index for index in range(38)}
    expected_report = {
        "overall_accuracy": 91.5,
        "total_samples": 2,
        "confusion_matrix": [[1, 0], [0, 1]],
        "class_names": ["class_00", "class_01"],
        "per_class_accuracy": [],
        "top_confusions": [],
        "misclassified_examples": [],
    }
    captured = {}

    monkeypatch.setattr(validate, "configure_validation_logging", lambda cfg: logging.getLogger("validate-test"))
    monkeypatch.setattr(validate, "get_device", lambda use_gpu=True: "cpu")
    monkeypatch.setattr(validate, "load_class_mapping", lambda path: class_to_idx)
    monkeypatch.setattr(validate, "build_validation_loader", lambda cfg: ([1, 2], class_to_idx))
    monkeypatch.setattr(validate, "load_model", lambda cfg, checkpoint, device: object())
    monkeypatch.setattr(validate, "evaluate_validation_set", lambda model, val_loader, device, class_names: expected_report)
    monkeypatch.setattr(validate, "save_validation_report", lambda report, output_path: captured.update({"report": report, "output_path": output_path}))

    report = validate.main(str(config_path))

    assert report == expected_report
    assert captured["report"] == expected_report
    assert captured["output_path"] == str(tmp_path / "logs" / "validation_report.json")