"""Verify that the Production model can be loaded from MLflow Model Registry.

This script tests that:
1. The Production model exists in Model Registry
2. The model can be loaded successfully
3. The model can perform inference (if test image provided)

Usage:
    python -m src.verify --config config.yaml
    python -m src.verify --config config.yaml --test-image data/Apple___healthy/image.JPG
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Optional

import yaml
import torch
from PIL import Image
from torchvision import transforms

import mlflow

from src.utils import get_device, setup_logging


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration from `path` and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def configure_mlflow(cfg: Dict[str, Any]) -> None:
    """Configure MLflow tracking from config."""
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri") or cfg["paths"].get("mlflow_artifacts_dir", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)


def verify_production_model_exists(cfg: Dict[str, Any], logger: logging.Logger) -> bool:
    """Verify that the Production model exists in Model Registry.
    
    Args:
        cfg: Configuration dictionary
        logger: Logger instance
        
    Returns:
        True if exists, False otherwise
    """
    try:
        model_name = cfg.get("mlflow", {}).get("model_name", "GreenVision")
        client = mlflow.tracking.MlflowClient()
        
        # Get latest Production version
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            logger.error("No Production model found in Model Registry for '%s'", model_name)
            return False
        
        prod_version = versions[0]
        logger.info(
            "✅ Found Production model: %s (version %s, status: %s)",
            model_name,
            prod_version.version,
            prod_version.status,
        )
        return True
    except Exception as e:
        logger.error("Failed to verify Production model exists: %s", str(e))
        return False


def load_production_model(cfg: Dict[str, Any], logger: logging.Logger):
    """Load the Production model from Model Registry.
    
    Args:
        cfg: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Loaded model or None if failed
    """
    try:
        model_name = cfg.get("mlflow", {}).get("model_name", "GreenVision")
        model = mlflow.pytorch.load_model(f"models:/{model_name}/Production")
        logger.info("✅ Successfully loaded Production model '%s' from Model Registry", model_name)
        return model
    except Exception as e:
        logger.error("Failed to load Production model: %s", str(e))
        return None


def test_inference(
    model,
    image_path: str,
    cfg: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> bool:
    """Test inference on a test image.
    
    Args:
        model: Loaded PyTorch model
        image_path: Path to test image
        cfg: Configuration dictionary
        device: Torch device
        logger: Logger instance
        
    Returns:
        True if inference succeeded, False otherwise
    """
    if not os.path.isfile(image_path):
        logger.warning("Test image not found: %s (skipping inference test)", image_path)
        return True  # Not a failure, just skip

    try:
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_size = cfg["project"].get("image_size", 224)
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        tensor = transform(img).unsqueeze(0).to(device)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            output = model(tensor)
            pred_idx = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, pred_idx].item()
        
        logger.info(
            "✅ Inference test passed: predicted class index %d with confidence %.4f",
            pred_idx,
            confidence,
        )
        return True
    except Exception as e:
        logger.error("Inference test failed: %s", str(e))
        return False


def main(config_path: str, test_image: Optional[str] = None):
    """Main verification workflow."""
    cfg = load_config(config_path)

    # Setup logging
    logs_dir = cfg["paths"].get("logs_dir", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "verify.log") if cfg.get("logging", {}).get("log_to_file", True) else None
    setup_logging(log_file=log_file, level=getattr(logging, cfg.get("logging", {}).get("level", "INFO")))

    logger = logging.getLogger(__name__)
    logger.info("=== Production Model Verification ===")

    device = get_device(use_gpu=True)
    configure_mlflow(cfg)

    # Step 1: Verify model exists in Registry
    if not verify_production_model_exists(cfg, logger):
        logger.error("❌ Verification failed: Production model does not exist")
        return 1

    # Step 2: Load Production model
    model = load_production_model(cfg, logger)
    if model is None:
        logger.error("❌ Verification failed: Could not load Production model")
        return 1

    model = model.to(device)
    model.eval()

    # Step 3: Optional inference test
    if test_image:
        if not test_inference(model, test_image, cfg, device, logger):
            logger.error("❌ Verification failed: Inference test failed")
            return 1

    logger.info("✅ All verification checks passed!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Production model from MLflow Model Registry")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--test-image", default=None, help="Optional: path to test image for inference verification")
    args = parser.parse_args()

    exit_code = main(args.config, args.test_image)
    exit(exit_code)
