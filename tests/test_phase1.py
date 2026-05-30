"""Phase 1 smoke test for the GreenVision training entrypoint.

This test uses the real dataset loader against the repository's `data/`
folder so we verify the 38-class mapping is produced correctly, while the
actual training loop is stubbed to keep the test fast and deterministic.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

import src.train as train


class DummyParameter:
    """Minimal stand-in for a tensor parameter used in logging."""

    def __init__(self, size: int) -> None:
        self._size = size

    def numel(self) -> int:
        return self._size


class FakeModel:
    """Lightweight model double used by the phase 1 smoke test."""

    def __init__(self) -> None:
        self.loaded_checkpoint: str | None = None
        self.device = None
        self._parameters = [DummyParameter(12), DummyParameter(8)]

    def parameters(self):
        return list(self._parameters)

    def freeze_backbone(self) -> None:
        return None

    def unfreeze_backbone(self) -> None:
        return None

    def to(self, device):
        self.device = device
        return self

    def load_checkpoint(self, path: str) -> None:
        self.loaded_checkpoint = path

    def state_dict(self):
        return {"fake_weight": 1}


@pytest.fixture()
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_phase1_smoke(repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the Phase 1 orchestration against the real PlantVillage data layout."""
    config_path = tmp_path / "config.yaml"
    class_mapping_path = tmp_path / "models" / "class_to_idx.json"

    cfg = {
        "project": {
            "name": "GreenVision",
            "seed": 42,
            "image_size": 224,
            "num_classes": 38,
        },
        "paths": {
            "data_dir": str(repo_root / "data"),
            "model_dir": str(tmp_path / "models"),
            "checkpoint_best": str(tmp_path / "models" / "best_model.pth"),
            "checkpoint_phase1": str(tmp_path / "models" / "phase1_best.pth"),
            "checkpoint_phase2": str(tmp_path / "models" / "phase2_best.pth"),
            "class_to_idx": str(class_mapping_path),
            "idx_to_class": str(tmp_path / "models" / "idx_to_class.json"),
            "logs_dir": str(tmp_path / "logs"),
            "mlflow_artifacts_dir": str(tmp_path / "mlruns"),
        },
        "mlflow": {
            "experiment_name": "GreenVision",
            "tracking_uri": str(tmp_path / "mlruns"),
            "ui_port": 5000,
            "run_name_phase1": "phase1_feature_extraction",
            "run_name_phase2": "phase2_fine_tuning",
            "log_system_metrics": True,
            "nested_runs": False,
        },
        "model": {
            "backbone": "efficientnet_b0",
            "pretrained": True,
            "feature_dim": 1280,
            "hidden_dim": 128,
            "dropout": 0.5,
        },
        "splits": {
            "train": 0.75,
            "validation": 0.15,
            "test": 0.10,
            "validation_images": False,
        },
        "training": {
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": False,
            "phase1": {
                "epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "freeze_backbone": True,
            },
            "phase2": {
                "epochs": 0,
                "learning_rate": 0.0001,
                "weight_decay": 0.0001,
                "unfreeze_backbone": True,
            },
            "regularization": {
                "dropout": 0.5,
                "weight_decay": 0.0001,
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 0.0,
            },
        },
        "logging": {
            "level": "INFO",
            "log_to_file": False,
        },
    }

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    captured = {
        "train_calls": 0,
        "val_calls": 0,
        "test_calls": 0,
        "logged_params": None,
        "logged_metrics": [],
        "logged_artifacts": [],
        "saved_checkpoint": None,
    }

    fake_model = FakeModel()

    def fake_setup_logging(*args, **kwargs):
        return None

    def fake_get_device(*args, **kwargs):
        return "cpu"

    def fake_configure_mlflow(*args, **kwargs):
        return None

    def fake_build_model_and_optimizer(*args, **kwargs):
        return fake_model, object(), object()

    def fake_train_one_epoch(*args, **kwargs):
        captured["train_calls"] += 1
        return {"loss": 0.40, "accuracy": 82.5, "samples": 8.0}

    def fake_validate_one_epoch(*args, **kwargs):
        captured["val_calls"] += 1
        return {"loss": 0.25, "accuracy": 88.0, "samples": 4.0}

    def fake_evaluate_test_set(*args, **kwargs):
        captured["test_calls"] += 1
        return {"loss": 0.20, "accuracy": 90.0, "samples": 4.0}

    def fake_log_params(params):
        captured["logged_params"] = dict(params)

    def fake_log_metrics(metrics, step=None):
        captured["logged_metrics"].append((dict(metrics), step))

    def fake_log_artifact(path, artifact_path=None):
        captured["logged_artifacts"].append((Path(path).name, artifact_path))

    @contextmanager
    def fake_start_run(*args, **kwargs):
        yield object()

    def fake_save_best_checkpoint(model, path, *, epoch, val_loss, val_accuracy, extra_metadata=None):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text("checkpoint", encoding="utf-8")
        captured["saved_checkpoint"] = path

    monkeypatch.setattr(train, "setup_logging", fake_setup_logging)
    monkeypatch.setattr(train, "get_device", fake_get_device)
    monkeypatch.setattr(train, "configure_mlflow", fake_configure_mlflow)
    monkeypatch.setattr(train, "build_model_and_optimizer", fake_build_model_and_optimizer)
    monkeypatch.setattr(train, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train, "validate_one_epoch", fake_validate_one_epoch)
    monkeypatch.setattr(train, "evaluate_test_set", fake_evaluate_test_set)
    monkeypatch.setattr(train, "save_best_checkpoint", fake_save_best_checkpoint)
    monkeypatch.setattr(train.mlflow, "set_tracking_uri", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.mlflow, "set_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(train.mlflow, "log_params", fake_log_params)
    monkeypatch.setattr(train.mlflow, "log_metrics", fake_log_metrics)
    monkeypatch.setattr(train.mlflow, "log_artifact", fake_log_artifact)

    train.main(str(config_path))

    saved_mapping = json.loads(class_mapping_path.read_text(encoding="utf-8"))
    assert len(saved_mapping) == 38
    assert "Background_without_leaves" not in saved_mapping
    assert captured["train_calls"] == 1
    assert captured["val_calls"] == 1
    assert captured["test_calls"] == 1
    assert captured["logged_params"]["mlflow_ui_port"] == 5000
    assert any("phase1_train_loss" in metric for metric, _ in captured["logged_metrics"])
    assert any("test_accuracy" in metric for metric, _ in captured["logged_metrics"])
    assert captured["saved_checkpoint"] == cfg["paths"]["checkpoint_phase1"]
    assert fake_model.loaded_checkpoint == cfg["paths"]["checkpoint_phase1"]
    assert any(name == "class_to_idx.json" for name, _ in captured["logged_artifacts"])