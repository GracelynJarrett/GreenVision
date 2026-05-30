"""Unit tests for src/model.py DiseaseClassifier.

Tests cover:
- Model initialization and architecture
- Forward pass with correct input/output shapes
- Freeze/unfreeze backbone parameter control
- Device movement (CPU/GPU)
- Train/eval mode switching
- Checkpoint save/load round-trip
"""
import tempfile
from pathlib import Path

import pytest
import torch

from src.model import DiseaseClassifier, NUM_CLASSES


class TestDiseaseClassifierInit:
    """Test model initialization and architecture."""

    def test_init_default(self):
        """Test default initialization with standard hyperparameters."""
        model = DiseaseClassifier()
        assert model.num_classes == NUM_CLASSES
        assert isinstance(model.backbone, torch.nn.Module)
        assert isinstance(model.classifier, torch.nn.Sequential)

    def test_init_custom_classes(self):
        """Test initialization with custom number of classes."""
        model = DiseaseClassifier(num_classes=10)
        assert model.num_classes == 10

    def test_init_custom_dropout(self):
        """Test initialization with custom dropout rate."""
        model = DiseaseClassifier(dropout_rate=0.3)
        # The Dropout layer is the 3rd element (0-indexed) in the sequential classifier
        dropout_layer = model.classifier[2]
        assert isinstance(dropout_layer, torch.nn.Dropout)
        assert dropout_layer.p == 0.3

    def test_init_invalid_num_classes(self):
        """Test that invalid num_classes raises ValueError."""
        with pytest.raises(ValueError):
            DiseaseClassifier(num_classes=0)

    def test_init_invalid_dropout(self):
        """Test that invalid dropout_rate raises ValueError."""
        with pytest.raises(ValueError):
            DiseaseClassifier(dropout_rate=1.5)


class TestDiseaseClassifierForward:
    """Test forward pass and output shapes."""

    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        model = DiseaseClassifier().eval()
        batch_size = 4
        # Input: (batch, 3 channels, 224x224)
        x = torch.randn(batch_size, 3, 224, 224)
        logits = model(x)
        # Output: (batch, num_classes)
        assert logits.shape == (batch_size, NUM_CLASSES)

    def test_forward_single_image(self):
        """Test forward pass with a single image."""
        model = DiseaseClassifier().eval()
        x = torch.randn(1, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (1, NUM_CLASSES)

    def test_forward_no_grad(self):
        """Test forward pass during inference (no gradient tracking)."""
        model = DiseaseClassifier().eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, NUM_CLASSES)
        assert not logits.requires_grad


class TestDiseaseClassifierFreeze:
    """Test backbone freezing and unfreezing."""

    def test_freeze_backbone(self):
        """Test that freeze_backbone sets all backbone params to requires_grad=False."""
        model = DiseaseClassifier()
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad is False

    def test_unfreeze_backbone(self):
        """Test that unfreeze_backbone sets all backbone params to requires_grad=True."""
        model = DiseaseClassifier()
        model.freeze_backbone()  # First freeze
        model.unfreeze_backbone()  # Then unfreeze
        for param in model.backbone.parameters():
            assert param.requires_grad is True

    def test_classifier_always_trainable(self):
        """Test that classifier head remains trainable after freezing."""
        model = DiseaseClassifier()
        model.freeze_backbone()
        # Classifier should still be trainable
        for param in model.classifier.parameters():
            assert param.requires_grad is True


class TestDiseaseClassifierDeviceAndMode:
    """Test device movement and train/eval modes."""

    def test_to_cpu(self):
        """Test moving model to CPU."""
        model = DiseaseClassifier()
        model.to('cpu')
        assert model.device == 'cpu'
        # Check that parameters are on CPU
        for param in model.parameters():
            assert param.device.type == 'cpu'

    def test_to_device_object(self):
        """Test moving model using torch.device object."""
        model = DiseaseClassifier()
        device = torch.device('cpu')
        model.to(device)
        assert model.device == device

    def test_train_mode(self):
        """Test switching to train mode."""
        model = DiseaseClassifier()
        model.train()
        assert model.training is True
        # Dropout should be active in training mode
        dropout_layer = model.classifier[2]
        assert dropout_layer.training is True

    def test_eval_mode(self):
        """Test switching to eval mode."""
        model = DiseaseClassifier()
        model.eval()
        assert model.training is False
        # Dropout should be inactive in eval mode
        dropout_layer = model.classifier[2]
        assert dropout_layer.training is False

    def test_method_chaining(self):
        """Test that to/train/eval return self for chaining."""
        model = DiseaseClassifier()
        result = model.to('cpu').train().eval()
        assert result is model


class TestDiseaseClassifierCheckpoint:
    """Test checkpoint save/load functionality."""

    def test_save_load_checkpoint(self):
        """Test round-trip save and load checkpoint."""
        model = DiseaseClassifier()
        # Modify a weight to verify it is loaded correctly
        original_weight = model.classifier[0].weight.data.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"
            model.save_checkpoint(str(checkpoint_path))
            assert checkpoint_path.exists()

            # Create a new model and load the checkpoint
            new_model = DiseaseClassifier()
            new_model.load_checkpoint(str(checkpoint_path))

            # Verify weights are identical
            loaded_weight = new_model.classifier[0].weight.data
            assert torch.allclose(original_weight, loaded_weight)

    def test_load_checkpoint_missing_file(self):
        """Test that loading from missing file raises FileNotFoundError."""
        model = DiseaseClassifier()
        with pytest.raises(FileNotFoundError):
            model.load_checkpoint("/nonexistent/path/model.pth")

    def test_checkpoint_preserves_num_classes(self):
        """Test that checkpoint stores and can retrieve num_classes."""
        model = DiseaseClassifier(num_classes=20)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"
            model.save_checkpoint(str(checkpoint_path))

            # Load and verify num_classes is preserved
            checkpoint = torch.load(checkpoint_path)
            assert checkpoint['num_classes'] == 20


class TestDiseaseClassifierIntegration:
    """Integration tests combining multiple features."""

    def test_phase1_training_setup(self):
        """Test Phase 1 setup: frozen backbone, trainable head."""
        model = DiseaseClassifier()
        model.freeze_backbone()
        model.train()

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        # Should have fewer trainable params than total (backbone is frozen)
        assert trainable_params < total_params
        # Trainable params should mainly be from the classifier
        assert trainable_params > 0

    def test_phase2_training_setup(self):
        """Test Phase 2 setup: unfrozen backbone, fine-tuning."""
        model = DiseaseClassifier()
        model.unfreeze_backbone()
        model.train()

        # All parameters should be trainable
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        assert trainable_params == total_params

    def test_inference_workflow(self):
        """Test typical inference workflow: eval mode, no grad, move to device."""
        model = DiseaseClassifier()
        model.to('cpu').eval()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (2, NUM_CLASSES)
        assert not logits.requires_grad
