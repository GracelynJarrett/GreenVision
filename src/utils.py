"""Utility functions for GreenVision training pipeline.

Provides:
- Logging configuration
- Device detection and management
- Metrics tracking and computation
- Training and validation loops
- Early stopping and checkpoint management

Follow project rules: explicit error handling, small readable functions,
and inline comments for non-obvious logic.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure logging to console and optional file.

    Sets up a logger that outputs to both the console and an optional file.
    This centralizes logging configuration and ensures consistent formatting
    across all modules.

    Args:
        log_file: Optional path to write logs to a file. If None, only console
            output is used.
        level: Logging level (e.g., logging.INFO, logging.DEBUG). Default is INFO.

    Returns:
        Configured logger instance.
    """
    # Get the root logger to configure all child loggers
    logger = logging.getLogger()
    logger.setLevel(level)

    # Define a consistent format for all log messages:
    # timestamp | level | module name | message
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler: output to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler: optional file output
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Logging to file: %s", log_file)

    return logger


def get_device(use_gpu: bool = True) -> torch.device:
    """Detect and return the appropriate device (CPU or GPU).

    Checks if CUDA (GPU) is available and, if requested, returns the GPU device.
    Otherwise, returns CPU. This helper ensures consistent device selection
    across the training pipeline.

    Args:
        use_gpu: If True, use GPU if available; otherwise use CPU. Default is True.

    Returns:
        torch.device object representing the selected device.
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger = logging.getLogger(__name__)
        logger.info("GPU available. Using device: %s", device)
        # Log GPU details for debugging
        gpu_count = torch.cuda.device_count()
        logger.info("GPU count: %d, GPU name: %s", gpu_count, torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger = logging.getLogger(__name__)
        logger.info("Using device: CPU")

    return device


def move_to_device(obj, device: torch.device):
    """Recursively move tensors, models, or nested structures to a device.

    This helper safely moves PyTorch objects (tensors, models, or nested dicts/lists
    of tensors) to the specified device (CPU or GPU).

    Args:
        obj: Object to move (tensor, model, dict, list, or nested structure).
        device: Target device (torch.device object).

    Returns:
        The object after moving all tensors to the target device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        # Recursively move all values in a dictionary
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively move all elements in a list or tuple
        moved = [move_to_device(item, device) for item in obj]
        return type(obj)(moved)
    else:
        # For non-tensor objects, return as-is
        return obj


class MetricTracker:
    """Track running loss and accuracy during training or validation.

    This helper keeps the training loop clean by encapsulating the logic for
    accumulating batch-level statistics and computing epoch-level summaries.
    It also optionally tracks per-class correct counts so the caller can inspect
    class-wise performance without duplicating bookkeeping logic.
    """

    def __init__(self) -> None:
        """Initialize the tracker with empty running totals."""
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated statistics to zero."""
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_samples = 0
        self.class_correct: Dict[int, int] = {}
        self.class_total: Dict[int, int] = {}

    def update(self, loss: float, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Update running metrics from a batch.

        Args:
            loss: Batch loss value as a Python float.
            outputs: Model logits with shape (batch_size, num_classes).
            targets: Ground-truth labels with shape (batch_size,).
        """
        if outputs.ndim != 2:
            raise ValueError(f"outputs must be 2D logits, got shape {tuple(outputs.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"targets must be 1D labels, got shape {tuple(targets.shape)}")
        if outputs.size(0) != targets.size(0):
            raise ValueError(
                f"Batch size mismatch: outputs batch={outputs.size(0)}, targets batch={targets.size(0)}"
            )

        # Compute the predicted class for each sample.
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == targets).sum().item()
        batch_size = targets.size(0)

        # Update global running totals.
        self.total_loss += float(loss) * batch_size
        self.total_correct += int(correct)
        self.total_samples += int(batch_size)

        # Track per-class counts for optional class-wise inspection.
        for target, prediction in zip(targets.tolist(), predictions.tolist()):
            self.class_total[target] = self.class_total.get(target, 0) + 1
            if target == prediction:
                self.class_correct[target] = self.class_correct.get(target, 0) + 1

    def average_loss(self) -> float:
        """Return the mean loss across all seen samples."""
        if self.total_samples == 0:
            return 0.0
        return self.total_loss / self.total_samples

    def accuracy(self) -> float:
        """Return overall classification accuracy as a percentage."""
        if self.total_samples == 0:
            return 0.0
        return (self.total_correct / self.total_samples) * 100.0

    def summary(self) -> Dict[str, float]:
        """Return a compact dictionary of the current metric values."""
        return {
            "loss": self.average_loss(),
            "accuracy": self.accuracy(),
            "samples": float(self.total_samples),
        }


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train the model for one epoch and return aggregated metrics.

    This helper keeps the training loop in `train.py` focused on orchestration.
    It performs the standard sequence:
    1. Set model to train mode.
    2. Loop over batches.
    3. Move data to the requested device.
    4. Run forward pass, compute loss, backpropagate, and update weights.
    5. Track epoch-level loss and accuracy.

    Args:
        model: PyTorch model to train.
        dataloader: Training dataloader.
        criterion: Loss function (e.g., CrossEntropyLoss).
        optimizer: Optimizer used to update parameters.
        device: Target compute device.

    Returns:
        Dictionary with epoch `loss`, `accuracy`, and `samples`.
    """
    model.train()
    tracker = MetricTracker()

    for batch_index, (inputs, targets) in enumerate(dataloader):
        # Move batch tensors to the selected device.
        inputs = move_to_device(inputs, device)
        targets = move_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        tracker.update(loss.item(), outputs.detach(), targets.detach())

    return tracker.summary()


def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model for one epoch and return aggregated metrics.

    Validation runs without gradient tracking so it is faster and does not
    modify model weights. The same metric aggregation logic is used as in
    training, but the optimizer step is omitted.

    Args:
        model: PyTorch model to evaluate.
        dataloader: Validation dataloader.
        criterion: Loss function used for evaluation.
        device: Target compute device.

    Returns:
        Dictionary with epoch `loss`, `accuracy`, and `samples`.
    """
    model.eval()
    tracker = MetricTracker()

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(dataloader):
            # Move batch tensors to the selected device.
            inputs = move_to_device(inputs, device)
            targets = move_to_device(targets, device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            tracker.update(loss.item(), outputs, targets)

    return tracker.summary()


class EarlyStopping:
    """Stop training when validation loss stops improving.

    This helper is used by `train.py` to prevent overfitting and avoid
    unnecessary epochs once the model has converged. It tracks the best
    validation loss seen so far and counts how many consecutive epochs have
    failed to improve it by at least `min_delta`.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        """Initialize early stopping state.

        Args:
            patience: Number of consecutive non-improving epochs allowed before
                stopping.
            min_delta: Minimum loss improvement required to reset patience.

        Raises:
            ValueError: If patience is negative or min_delta is negative.
        """
        if patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")

        self.patience = patience
        self.min_delta = float(min_delta)
        self.best_loss = float("inf")
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, current_loss: float) -> bool:
        """Update state with the latest validation loss.

        Args:
            current_loss: Most recent validation loss.

        Returns:
            True if training should stop, otherwise False.
        """
        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = float(current_loss)
            self.bad_epochs = 0
            self.should_stop = False
            return False

        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        """Reset early stopping state for a new training phase."""
        self.best_loss = float("inf")
        self.bad_epochs = 0
        self.should_stop = False


def save_best_checkpoint(
    model: torch.nn.Module,
    path: str,
    *,
    epoch: int,
    val_loss: float,
    val_accuracy: float,
    extra_metadata: Optional[Dict[str, float]] = None,
) -> None:
    """Save a best-model checkpoint with training metadata.

    This helper centralizes the checkpoint format used by `train.py`. The
    resulting file stores model weights plus a minimal set of metadata useful
    for reproducing and comparing runs.

    Args:
        model: Trained model to serialize.
        path: Destination path for the checkpoint file.
        epoch: Epoch number associated with the checkpoint.
        val_loss: Validation loss at the time of saving.
        val_accuracy: Validation accuracy at the time of saving.
        extra_metadata: Optional additional scalar metadata to include.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
    }

    if extra_metadata:
        checkpoint["extra_metadata"] = dict(extra_metadata)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(checkpoint, path)

    logging.getLogger(__name__).info(
        "Saved best checkpoint to %s (epoch=%d, val_loss=%.4f, val_accuracy=%.2f)",
        path,
        epoch,
        val_loss,
        val_accuracy,
    )
