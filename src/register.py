"""Register and promote a trained model to MLflow Model Registry.

This standalone script registers a checkpoint to MLflow's Model Registry
and promotes it to Production stage. Can be run after validation/testing
to migrate the best model for serving.

Usage:
    python -m src.register --config config.yaml --checkpoint models/m3_best.pth
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Optional

import yaml
import torch
import mlflow
from mlflow.tracking import MlflowClient

from src.model import DiseaseClassifier
from src.utils import get_device, setup_logging


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration from `path` and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def configure_mlflow(cfg: Dict[str, Any]) -> None:
    """Configure MLflow tracking and experiment selection from config."""
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri") or cfg["paths"].get("mlflow_artifacts_dir", "mlruns")
    experiment_name = mlflow_cfg.get("experiment_name", cfg["project"].get("name", "GreenVision"))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def register_model_from_checkpoint(
    checkpoint_path: str,
    cfg: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> Optional[str]:
    """Register a checkpoint to MLflow Model Registry and promote to Production.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        cfg: Configuration dictionary
        device: Torch device (cuda or cpu)
        logger: Logger instance
        
    Returns:
        Model URI if successful, None otherwise
    """
    if not os.path.isfile(checkpoint_path):
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return None

    # Load model from checkpoint
    try:
        num_classes = cfg["project"].get("num_classes", 38)
        model = DiseaseClassifier(num_classes=num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        logger.info("Loaded model from checkpoint: %s", checkpoint_path)
    except Exception as e:
        logger.error("Failed to load checkpoint: %s", str(e))
        return None

    # Start MLflow run for registration
    with mlflow.start_run(run_name="model-registration") as run:
        # Log model to MLflow as PyTorch artifact
        try:
            model_info = mlflow.pytorch.log_model(model, artifact_path="model", input_example=None)
            model_uri = model_info.model_uri
            logger.info("Logged model to MLflow: %s", model_uri)
        except Exception as e:
            logger.error("Failed to log model to MLflow: %s", str(e))
            return None

        # Register model in Model Registry
        try:
            model_name = cfg.get("mlflow", {}).get("model_name", "GreenVision")
            registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
            logger.info(
                "Registered model '%s' version %s",
                model_name,
                registered_model.version,
            )
        except Exception as e:
            logger.error("Failed to register model: %s", str(e))
            return None

        # Promote to Production stage
        try:
            client = MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=registered_model.version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info(
                "Promoted model '%s' version %s to Production stage",
                model_name,
                registered_model.version,
            )
        except Exception as e:
            logger.error("Failed to promote model to Production: %s", str(e))
            return None

    return model_uri


def verify_production_model(cfg: Dict[str, Any], logger: logging.Logger) -> bool:
    """Verify that the Production model can be loaded from Model Registry.
    
    Args:
        cfg: Configuration dictionary
        logger: Logger instance
        
    Returns:
        True if load successful, False otherwise
    """
    try:
        model_name = cfg.get("mlflow", {}).get("model_name", "GreenVision")
        loaded_model = mlflow.pytorch.load_model(f"models:/{model_name}/Production")
        logger.info("✅ Successfully loaded '%s/Production' from Model Registry", model_name)
        return True
    except Exception as e:
        logger.error("Failed to load Production model: %s", str(e))
        return False


def main(config_path: str, checkpoint_path: str):
    """Main orchestration: register checkpoint and verify Production load."""
    cfg = load_config(config_path)

    # Setup logging
    logs_dir = cfg["paths"].get("logs_dir", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "register.log") if cfg.get("logging", {}).get("log_to_file", True) else None
    setup_logging(log_file=log_file, level=getattr(logging, cfg.get("logging", {}).get("level", "INFO")))

    logger = logging.getLogger(__name__)
    logger.info("Checkpoint registration started: %s", checkpoint_path)

    device = get_device(use_gpu=True)
    configure_mlflow(cfg)

    # Register model
    model_uri = register_model_from_checkpoint(checkpoint_path, cfg, device, logger)
    if model_uri is None:
        logger.error("Model registration failed.")
        return 1

    # Verify Production model can be loaded
    if not verify_production_model(cfg, logger):
        logger.warning("Production model verification failed, but registration completed.")
        return 1

    logger.info("✅ Registration and verification complete!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register model checkpoint to MLflow Model Registry")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth file)")
    args = parser.parse_args()

    exit_code = main(args.config, args.checkpoint)
    exit(exit_code)
