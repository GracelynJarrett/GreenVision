
"""Dataset utilities for GreenVision.

Provides transform builders, stratified train/val/test splits using
`torchvision.datasets.ImageFolder`, and helpers to save/load the
`class_to_idx` mapping required by the serving code.

Public functions:
- `get_transforms`
- `make_datasets`
- `save_class_mapping`
- `load_class_mapping`

Follow project rules: ImageNet normalization and 224x224 image size.
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Dict, List, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# Project-mandated ImageNet normalization
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
EXCLUDED_CLASSES = {"Background_without_leaves"}


def _build_imagefolder(data_dir: str, transform=None) -> datasets.ImageFolder:
    """Build an ImageFolder and remove any excluded classes from it.

    The raw PlantVillage export in this workspace includes a background folder,
    but the project is configured for the 38 disease/healthy classes only.
    This helper drops the background class so the dataset, mapping, and model
    output size stay aligned.
    """
    folder = datasets.ImageFolder(data_dir, transform=transform)

    # Replace the default PIL loader with a safe loader that handles
    # permission errors, corrupted files, or other I/O problems by
    # returning a black RGB image of the expected size. This prevents
    # DataLoader worker crashes when a single file is unreadable.
    def _safe_pil_loader(path: str):
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except Exception as e:  # PermissionError, OSError, PIL.UnidentifiedImageError, etc.
            logger.warning("Unreadable image (will use fallback): %s — %s", path, e)
            # Return a small black RGB image; transforms will resize/crop as needed.
            return Image.new("RGB", (224, 224), (0, 0, 0))

    folder.loader = _safe_pil_loader

    kept_classes = [class_name for class_name in folder.classes if class_name not in EXCLUDED_CLASSES]
    if len(kept_classes) == len(folder.classes):
        return folder

    class_to_idx = {class_name: index for index, class_name in enumerate(kept_classes)}
    filtered_samples = []
    for path, target in folder.samples:
        class_name = folder.classes[target]
        if class_name in EXCLUDED_CLASSES:
            continue
        filtered_samples.append((path, class_to_idx[class_name]))

    folder.classes = kept_classes
    folder.class_to_idx = class_to_idx
    folder.samples = filtered_samples
    folder.targets = [target for _, target in filtered_samples]
    return folder


def get_transforms(image_size: int = 224, *, train: bool = True):
    """Return torchvision transforms for training or validation/inference.

    Args:
        image_size: Output image size (default 224).
        train: If True, return training (augmented) transforms; otherwise
            return validation/inference transforms.

    Returns:
        A `torchvision.transforms.Compose` transform.
    """
    if train:
        # Training augmentations:
        # - RandomResizedCrop: preserves object scale while providing
        #   varied crops to reduce overfitting.
        # - RandomHorizontalFlip / Rotation: small geometric transforms
        #   that simulate natural variations without destroying
        #   diagnostic disease patterns.
        # - ColorJitter: mild color changes to improve robustness to
        #   lighting differences.
        # - ToTensor + Normalize: convert to tensor and apply ImageNet
        #   normalization required by the pretrained EfficientNet backbone.
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ])

    # Validation / inference transforms:
    # - Resize -> CenterCrop: deterministic resizing used for evaluation.
    # - ToTensor + Normalize: match training normalization.
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
    ])


def make_datasets(
    data_dir: str,
    image_size: int = 224,
    train_frac: float = 0.75,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
    validate_images: bool = True,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, int]]:
    """Create stratified train/val/test datasets from an ImageFolder.

    The function uses `sklearn.model_selection.train_test_split` with
    stratification by class to ensure class proportions are preserved across
    splits. Each returned dataset is a `torch.utils.data.Subset` wrapping an
    `ImageFolder` instance with appropriate transforms applied.

    Args:
        data_dir: Root directory of the dataset arranged for ImageFolder.
        image_size: Image size passed to transforms.
        train_frac: Fraction for training set.
        val_frac: Fraction for validation set.
        test_frac: Fraction for test set.
        seed: Random seed for reproducible splits.

    Returns:
        (train_dataset, val_dataset, test_dataset, class_to_idx)
    """
    # Validate input directory exists
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    # Ensure fractions add up to 1.0
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    # Create a base ImageFolder without transforms to access file list and
    # `class_to_idx`. We create separate ImageFolder instances later so that
    # each split can have its own transform pipeline.
    base = _build_imagefolder(data_dir)
    if len(base.samples) == 0:
        raise ValueError(f"No images found under {data_dir}")

    class_to_idx = base.class_to_idx

    # Optionally validate images to skip corrupted files before splitting.
    # This avoids training crashes at the cost of a short filesystem pass.
    if validate_images:
        valid_indices: List[int] = []
        valid_targets: List[int] = []
        for i, (path, tgt) in enumerate(base.samples):
            try:
                with Image.open(path) as img:
                    img.verify()
                valid_indices.append(i)
                valid_targets.append(tgt)
            except Exception:
                logger.warning("Skipping invalid image: %s", path)

        if len(valid_indices) == 0:
            raise ValueError("No valid images found after validation pass")

        indices = valid_indices
        targets = valid_targets
    else:
        # indices and corresponding class targets used for stratified splitting
        indices = list(range(len(base)))
        targets = [s[1] for s in base.samples]

    # Split off the test set first to preserve class proportions.
    test_size = test_frac
    trainval_size = 1.0 - test_size

    # Prefer stratified splitting to preserve class proportions; if stratified
    # splitting fails (too-few samples per class), fallback to a non-stratified
    # shuffled split and log a warning so the user can inspect class counts.
    try:
        trainval_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=targets, random_state=seed
        )
    except ValueError:
        logger.warning(
            "Stratified split failed (small class counts). Falling back to non-stratified split."
        )
        trainval_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=seed
        )

    # Now split the remaining train/val portion. Compute relative validation
    # fraction with respect to the train+val pool.
    rel_val = val_frac / trainval_size if trainval_size > 0 else 0.0

    try:
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=rel_val,
            stratify=[targets[i] for i in trainval_idx],
            random_state=seed,
        )
    except ValueError:
        logger.warning(
            "Stratified train/val split failed (small class counts). Using non-stratified split."
        )
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=rel_val, random_state=seed
        )

    # Build separate ImageFolder instances with appropriate transforms so that
    # data augmentation is only applied to the training split.
    train_folder = _build_imagefolder(data_dir, transform=get_transforms(image_size, train=True))
    val_folder = _build_imagefolder(data_dir, transform=get_transforms(image_size, train=False))
    test_folder = _build_imagefolder(data_dir, transform=get_transforms(image_size, train=False))

    # Wrap the folders with Subset to restrict to the split indices.
    train_dataset = Subset(train_folder, train_idx)
    val_dataset = Subset(val_folder, val_idx)
    test_dataset = Subset(test_folder, test_idx)

    logger.info(
        "Created datasets: train=%d, val=%d, test=%d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    return train_dataset, val_dataset, test_dataset, class_to_idx


def get_transforms_from_config(augmentation_cfg: Dict[str, object], image_size: int = 224, *, train: bool = True):
    """Return transforms built from config values while preserving defaults.

    Args:
        augmentation_cfg: Augmentation section from configuration.
        image_size: Output image size.
        train: If True, build training transforms; otherwise validation transforms.

    Returns:
        A `torchvision.transforms.Compose` transform.
    """
    train_cfg = augmentation_cfg.get("train", {}) if isinstance(augmentation_cfg, dict) else {}
    val_cfg = augmentation_cfg.get("validation", {}) if isinstance(augmentation_cfg, dict) else {}
    norm_cfg = augmentation_cfg.get("normalization", {}) if isinstance(augmentation_cfg, dict) else {}

    mean = norm_cfg.get("mean", IMAGE_NET_MEAN)
    std = norm_cfg.get("std", IMAGE_NET_STD)

    if train:
        scale = train_cfg.get("random_resized_crop", {}).get("scale", [0.8, 1.0])
        horizontal_flip_prob = train_cfg.get("horizontal_flip_prob", 0.5)
        vertical_flip_prob = train_cfg.get("vertical_flip_prob", 0.5)
        rotation_degrees = train_cfg.get("rotation_degrees", 15)
        color_jitter = train_cfg.get("color_jitter", {})
        brightness = color_jitter.get("brightness", 0.1)
        contrast = color_jitter.get("contrast", 0.1)
        saturation = color_jitter.get("saturation", 0.1)
        hue = color_jitter.get("hue", 0.05)

        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(float(scale[0]), float(scale[1]))),
            transforms.RandomHorizontalFlip(p=float(horizontal_flip_prob)),
            transforms.RandomVerticalFlip(p=float(vertical_flip_prob)),
            transforms.RandomRotation(degrees=float(rotation_degrees)),
            transforms.ColorJitter(
                brightness=float(brightness),
                contrast=float(contrast),
                saturation=float(saturation),
                hue=float(hue),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    resize = val_cfg.get("resize", 256)
    center_crop = val_cfg.get("center_crop", image_size)
    return transforms.Compose([
        transforms.Resize(int(resize)),
        transforms.CenterCrop(int(center_crop)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def make_datasets_with_transforms(
    data_dir: str,
    train_transform,
    val_transform,
    test_transform,
    train_frac: float = 0.75,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
    validate_images: bool = True,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, int]]:
    """Create stratified train/val/test datasets with caller-provided transforms."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    base = _build_imagefolder(data_dir)
    if len(base.samples) == 0:
        raise ValueError(f"No images found under {data_dir}")

    class_to_idx = base.class_to_idx

    if validate_images:
        valid_indices: List[int] = []
        valid_targets: List[int] = []
        for index, (path, target) in enumerate(base.samples):
            try:
                with Image.open(path) as img:
                    img.verify()
                valid_indices.append(index)
                valid_targets.append(target)
            except Exception:
                logger.warning("Skipping invalid image: %s", path)

        if len(valid_indices) == 0:
            raise ValueError("No valid images found after validation pass")

        indices = valid_indices
        targets = valid_targets
    else:
        indices = list(range(len(base)))
        targets = [sample[1] for sample in base.samples]

    test_size = test_frac
    trainval_size = 1.0 - test_size

    try:
        trainval_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=targets, random_state=seed
        )
    except ValueError:
        logger.warning(
            "Stratified split failed (small class counts). Falling back to non-stratified split."
        )
        trainval_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)

    rel_val = val_frac / trainval_size if trainval_size > 0 else 0.0

    try:
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=rel_val,
            stratify=[targets[i] for i in trainval_idx],
            random_state=seed,
        )
    except ValueError:
        logger.warning(
            "Stratified train/val split failed (small class counts). Using non-stratified split."
        )
        train_idx, val_idx = train_test_split(trainval_idx, test_size=rel_val, random_state=seed)

    train_folder = _build_imagefolder(data_dir, transform=train_transform)
    val_folder = _build_imagefolder(data_dir, transform=val_transform)
    test_folder = _build_imagefolder(data_dir, transform=test_transform)

    train_dataset = Subset(train_folder, train_idx)
    val_dataset = Subset(val_folder, val_idx)
    test_dataset = Subset(test_folder, test_idx)

    logger.info(
        "Created tuned datasets: train=%d, val=%d, test=%d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    return train_dataset, val_dataset, test_dataset, class_to_idx


def save_class_mapping(class_to_idx: Dict[str, int], path: str = "models/class_to_idx.json") -> None:
    """Persist `class_to_idx` mapping as JSON for use during inference.

    Args:
        class_to_idx: Mapping returned by ImageFolder (`{class_name: index}`).
        path: Output path for JSON file.
    """
    # Ensure the target directory exists, then write the mapping as JSON.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)
    logger.info("Wrote class_to_idx to %s", path)


def load_class_mapping(path: str = "models/class_to_idx.json") -> Dict[str, int]:
    """Load persisted `class_to_idx` mapping.

    Raises FileNotFoundError when the mapping cannot be found.
    """
    # Read the saved mapping created during training. This mapping is used by
    # the serving layer to convert predicted indices back to human-readable
    # class names.
    if not os.path.isfile(path):
        raise FileNotFoundError(f"class_to_idx file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # quick smoke test when run as script
    logging.basicConfig(level=logging.INFO)
    try:
        d = os.path.join("data")
        train, val, test, mapping = make_datasets(d)
        # Persist the discovered class mapping so the serving app can load it.
        save_class_mapping(mapping)
        logger.info("Smoke test succeeded. Classes: %d", len(mapping))
    except Exception as e:
        logger.error("Smoke test failed: %s", e)
