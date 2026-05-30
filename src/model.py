"""Model utilities for GreenVision.

Provides the EfficientNet-B0 based plant disease classifier with a custom
classification head. The model follows a two-phase training strategy:
  - Phase 1: Freeze the backbone, train the classifier head.
  - Phase 2: Unfreeze deeper layers and fine-tune at a lower learning rate.

Public class:
- `DiseaseClassifier`

Architecture:
- Backbone: EfficientNet-B0 (pretrained on ImageNet)
- Feature dimension: 1280
- Custom head: Linear(1280→128) + ReLU + Dropout(0.5) + Linear(128→38)
- Number of classes: 38 (PlantVillage dataset)
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from timm import create_model

logger = logging.getLogger(__name__)

# Project-mandated constants
NUM_CLASSES = 38
EFFICIENTNET_FEATURE_DIM = 1280
HIDDEN_DIM = 128
DROPOUT_RATE = 0.5


class DiseaseClassifier(nn.Module):
    """Plant disease classifier using EfficientNet-B0 + custom head.

    The classifier consists of a pretrained EfficientNet-B0 backbone followed
    by a custom classifier head. The backbone can be frozen (phase-1 training)
    or unfrozen (phase-2 fine-tuning) to control which layers are updated
    during training.

    Attributes:
        backbone: EfficientNet-B0 feature extractor.
        classifier: Custom classification head.
        device: Device where model parameters reside (CPU or GPU).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = DROPOUT_RATE):
        """Initialize the disease classifier.

        Loads a pretrained EfficientNet-B0 backbone and replaces its
        default classifier with a custom head designed for plant disease
        classification.

        Args:
            num_classes: Number of output classes (default 38 for PlantVillage).
            dropout_rate: Dropout probability in the classifier head (default 0.5).

        Raises:
            ValueError: If num_classes <= 0 or dropout_rate not in [0, 1).
        """
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")
        if not (0 <= dropout_rate < 1):
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        # Load pretrained EfficientNet-B0 backbone.
        # The 'timm' library provides efficient implementations; using
        # 'efficientnet_b0' with pretrained=True loads ImageNet weights.
        logger.info("Loading pretrained EfficientNet-B0 backbone...")
        self.backbone = create_model("efficientnet_b0", pretrained=True, num_classes=0)
        # num_classes=0 removes the default classifier; we replace it below.

        # Build custom classifier head.
        # The head takes the backbone feature vector (1280-dim) and outputs
        # logits for num_classes. The intermediate layer (128 units) and
        # dropout help reduce overfitting and provide a representation bottleneck.
        self.classifier = nn.Sequential(
            nn.Linear(EFFICIENTNET_FEATURE_DIM, HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

        self.num_classes = num_classes
        self.device = None  # Will be set when model is moved to a device

        logger.info(
            "Initialized DiseaseClassifier: %d classes, feature_dim=%d, hidden_dim=%d, dropout=%.2f",
            num_classes,
            EFFICIENTNET_FEATURE_DIM,
            HIDDEN_DIM,
            dropout_rate,
        )

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters to prevent weight updates.

        Used during Phase 1 training: the backbone's pretrained ImageNet
        weights are preserved, and only the classifier head is trained.
        This approach:
        - Reduces training time (fewer parameters to update)
        - Stabilizes training (pretrained features are not overwritten)
        - Prevents overfitting early in training
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        logger.info("Backbone frozen: requires_grad=False for all backbone parameters")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters to allow fine-tuning.

        Used during Phase 2: deeper layers of the backbone are unfrozen
        and trained at a lower learning rate to adapt high-level features
        to plant-specific disease patterns while preserving low-level
        ImageNet features.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

        logger.info("Backbone unfrozen: requires_grad=True for all backbone parameters")

    def to(self, device: torch.device | str) -> DiseaseClassifier:
        """Move model to specified device (CPU or GPU).

        Args:
            device: Target device (e.g., 'cpu', 'cuda', torch.device('cuda:0')).

        Returns:
            Self for method chaining.
        """
        super().to(device)
        self.device = device
        logger.info("Model moved to device: %s", device)
        return self

    def train(self, mode: bool = True) -> DiseaseClassifier:
        """Set model to training mode.

        In training mode:
        - Dropout is active (stochastically zeros activations)
        - Batch normalization tracks running statistics
        - Gradients are computed for all parameters with requires_grad=True

        Args:
            mode: If True, enable training mode; if False, enable eval mode.

        Returns:
            Self for method chaining.
        """
        super().train(mode)
        logger.debug("Model mode set to: train=%s", mode)
        return self

    def eval(self) -> DiseaseClassifier:
        """Set model to evaluation mode.

        In evaluation mode:
        - Dropout is disabled
        - Batch normalization uses running statistics
        - Gradients are typically not computed (use with torch.no_grad())

        Returns:
            Self for method chaining.
        """
        super().eval()
        logger.debug("Model mode set to: eval")
        return self

    def save_checkpoint(self, path: str) -> None:
        """Save model state_dict and metadata to a checkpoint file.

        The checkpoint contains only the model weights and bias terms,
        not the optimizer state. For full training recovery, also save
        the optimizer state separately if needed.

        Args:
            path: Output path for the checkpoint file (typically .pth).
        """
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
        }

        torch.save(checkpoint, path)
        logger.info("Model checkpoint saved to: %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load model state_dict from a checkpoint file.

        Args:
            path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            KeyError: If the checkpoint format is invalid.
        """
        if not __import__('os').path.isfile(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device or 'cpu')

        if 'model_state_dict' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict' key")

        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model checkpoint loaded from: %s", path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and classifier.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224).
               Assumes images are already normalized to ImageNet distribution.

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # Extract features using the pretrained backbone.
        features = self.backbone(x)

        # Classify using the custom head.
        logits = self.classifier(features)

        return logits
