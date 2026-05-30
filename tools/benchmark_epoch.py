"""Quick benchmark: time a fixed number of training batches.

Run this from the repo root with the project venv. It performs a small
training loop (forward, loss, backward, step) for N batches and reports:
- device used (CPU/GPU)
- batches processed and total elapsed time
- time per batch and estimated time per epoch and for Phase 1 training

This is lightweight and intended to give a practical per-epoch estimate
on your machine before a full run.
"""
from __future__ import annotations

import os
import sys
import time
from math import ceil

import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import make_datasets
from src.model import DiseaseClassifier


def load_config(path: str = "config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml", batches_to_time: int = 50):
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = cfg["paths"]["data_dir"]
    img_size = cfg["project"].get("image_size", 224)

    train_ds, val_ds, test_ds, class_to_idx = make_datasets(
        data_dir,
        image_size=img_size,
        train_frac=cfg.get("splits", {}).get("train", 0.75),
        val_frac=cfg.get("splits", {}).get("validation", 0.15),
        test_frac=cfg.get("splits", {}).get("test", 0.10),
        seed=cfg["project"].get("seed", 42),
        validate_images=False,
    )

    batch_size = cfg["training"].get("batch_size", 16)
    num_workers = cfg["training"].get("num_workers", 0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    num_train_images = len(train_ds)
    batches_per_epoch = ceil(num_train_images / batch_size)
    print(f"Train images: {num_train_images}, batch_size: {batch_size}, batches/epoch: {batches_per_epoch}")

    model = DiseaseClassifier(num_classes=cfg["project"].get("num_classes", 38), dropout_rate=cfg["model"].get("dropout", 0.5))
    model.to(device)
    if cfg["training"].get("phase1", {}).get("freeze_backbone", True):
        model.freeze_backbone()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["training"]["phase1"].get("learning_rate", 1e-3))
    criterion = nn.CrossEntropyLoss()

    # Warm-up a few batches to avoid first-iteration overhead
    warmup = 5
    batches = 0
    print(f"Warming up {warmup} batches...")
    it = iter(train_loader)
    for _ in range(warmup):
        try:
            inputs, targets = next(it)
        except StopIteration:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Timed section
    print(f"Timing {batches_to_time} batches...")
    start = time.perf_counter()
    for _ in range(batches_to_time):
        try:
            inputs, targets = next(it)
        except StopIteration:
            it = iter(train_loader)
            inputs, targets = next(it)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        batches += 1

    elapsed = time.perf_counter() - start
    per_batch = elapsed / max(batches, 1)
    epoch_est = per_batch * batches_per_epoch
    total_epochs = cfg["training"].get("phase1", {}).get("epochs", 10)
    total_est = epoch_est * total_epochs

    print(f"Timed batches: {batches}, total time: {elapsed:.2f}s, per batch: {per_batch:.4f}s")
    print(f"Estimated time per epoch: {epoch_est/60:.2f} minutes")
    print(f"Estimated Phase-1 ({total_epochs} epochs) time: {total_est/60:.2f} minutes")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batches", type=int, default=50)
    args = parser.parse_args()
    main(config_path=args.config, batches_to_time=args.batches)
