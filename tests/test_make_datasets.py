import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from src.dataset import make_datasets


def _create_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    img.save(path)


def test_make_datasets_basic(tmp_path: Path):
    # Create a small synthetic dataset with 3 classes and 6 images each
    root = tmp_path / "data"
    classes = ["a", "b", "c"]
    for cls in classes:
        for i in range(6):
            p = root / cls / f"img_{i}.jpg"
            _create_image(p)

    train, val, test, mapping = make_datasets(str(root), image_size=224, seed=0)
    # check mapping and sizes
    assert set(mapping.keys()) == set(classes)
    total = len(train) + len(val) + len(test)
    assert total == 18


def test_make_datasets_stratify_fallback(tmp_path: Path):
    # Create classes where one class has only a single image to force fallback
    root = tmp_path / "data2"
    classes = ["a", "b", "c"]
    for cls in classes:
        count = 1 if cls == "c" else 5
        for i in range(count):
            p = root / cls / f"img_{i}.jpg"
            _create_image(p)

    # Should not raise even if stratified split is impossible; validate_images=False
    train, val, test, mapping = make_datasets(str(root), image_size=224, seed=0)
    total = len(train) + len(val) + len(test)
    # Expect all images accounted for
    assert total == 11
