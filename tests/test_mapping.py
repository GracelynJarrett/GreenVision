import json
import tempfile
from pathlib import Path

from src.dataset import load_class_mapping, save_class_mapping


def test_save_load_mapping(tmp_path: Path):
    mapping = {"apple_scab": 0, "healthy": 1}
    out = tmp_path / "models" / "class_to_idx.json"
    save_class_mapping(mapping, path=str(out))
    loaded = load_class_mapping(path=str(out))
    assert mapping == loaded
