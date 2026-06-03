"""
FastAPI inference server for GreenVision plant disease classification.

Loads the trained EfficientNet-B0 model from MLflow and serves predictions
via /health and /predict endpoints. Handles image preprocessing, validation,
and response formatting.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import mlflow
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Environment/config file paths
PROJECT_ROOT = Path(__file__).parent.parent
TREATMENTS_PATH = PROJECT_ROOT / "config" / "treatments.yaml"
MODEL_VERSION = "Production"
CONFIDENCE_THRESHOLD = 0.40  # Low confidence flag threshold
MAX_FILE_SIZE_MB = 10
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Image preprocessing constants (must match validation transforms)
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# GLOBAL STATE (loaded on startup)
# ============================================================================

model = None
idx_to_class = None
treatments = None
device = None


def extract_plant_type(class_name: str) -> str:
    """
    Extract plant type from class name by splitting on first underscore.
    
    Examples:
        "Tomato___Late_blight" -> "Tomato"
        "Pepper,_bell___Bacterial_spot" -> "Pepper, bell"
        "Background_without_leaves" -> "Background"
    
    Args:
        class_name: Raw class name from training.
    
    Returns:
        Plant type string (formatted, no underscores).
    """
    # Split on first underscore sequence (usuall ___)
    parts = class_name.split("___")
    if len(parts) > 0:
        plant = parts[0].replace("_", " ")
        return plant
    return class_name


def extract_disease_name(class_name: str) -> str:
    """
    Extract and format disease name from class name (remove plant prefix).
    
    Splits on ___, takes the disease part, replaces underscores with spaces,
    and applies title case formatting.
    
    Examples:
        "Tomato___Late_blight" -> "Late blight"
        "Apple___Apple_scab" -> "Apple scab"
        "Tomato___healthy" -> "Healthy"
        "Background_without_leaves" -> "Without leaves"
    
    Args:
        class_name: Raw class name from training (e.g., "Tomato___Late_blight").
    
    Returns:
        Formatted disease string ready for display.
    """
    parts = class_name.split("___")
    if len(parts) == 2:
        disease = parts[1].replace("_", " ").title()
        return disease
    # Fallback for Background or other single-part names
    return class_name.replace("_", " ").title()


def load_model_and_artifacts() -> tuple:
    """
    Load the trained model, class mappings, and treatment recommendations from MLflow.
    
    This function:
    1. Connects to MLflow and loads the model from the Production registry
    2. Downloads and inverts the class_to_idx.json artifact
    3. Loads treatment recommendations from config/treatments.yaml
    4. Sets the model to eval mode and moves to appropriate device
    
    Returns:
        Tuple of (model, idx_to_class, treatments, device)
    
    Raises:
        RuntimeError: If model or artifacts fail to load.
    """
    try:
        logger.info("Loading model from MLflow Registry...")
        
        # Determine device (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load model from MLflow Registry
        model = mlflow.pytorch.load_model("models:/GreenVision/Production")
        model.to(device)
        model.eval()  # Set to evaluation mode
        logger.info(f"Model loaded successfully from MLflow ({MODEL_VERSION})")
        
   #     # Load and invert class_to_idx from MLflow artifacts
  #      logger.info("Loading class mappings from MLflow artifacts...")
   #     client = mlflow.MlflowClient()
        
        # Get the latest Production version info
   #     versions = client.get_registered_model("GreenVision").latest_versions
   #     production_version = next(v for v in versions if v.version == MODEL_VERSION or v.current_stage == "Production")
        
        # Construct artifact URI and download class_to_idx.json
    #    artifact_uri = production_version.source
    #    artifact_path = mlflow.artifacts.download_artifacts(
    #        artifact_uri=artifact_uri,
   #         dst_path="/tmp/mlflow_artifacts"
   #     )
        
   #     class_to_idx_path = Path(artifact_path) / "class_to_idx.json"
   #     with open(class_to_idx_path, "r") as f:
        #    class_to_idx = json.load(f)
        
      # ===========================  
        # new added secion
      # ==============================
      
      # Load class_to_idx locally
        logger.info("Loading class mappings from local file...")

        class_to_idx_path = PROJECT_ROOT / "models" / "class_to_idx.json"

        with open(class_to_idx_path, "r") as f:
            class_to_idx = json.load(f)

        # Invert mapping
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}

        logger.info(f"Loaded {len(idx_to_class)} class mappings")  
        
        
        
        
        
        # Invert to create idx_to_class mapping
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        logger.info(f"Loaded {len(idx_to_class)} class mappings")
        
        # Load treatment recommendations
        logger.info(f"Loading treatments from {TREATMENTS_PATH}...")
        with open(TREATMENTS_PATH, "r") as f:
            treatments = yaml.safe_load(f)
        logger.info(f"Loaded treatments for {len(treatments)} disease classes")
        
        # Load treatments YAML
        with open(TREATMENTS_PATH, "r") as f:
            treatments = yaml.safe_load(f)
        
        return model, idx_to_class, treatments, device
    
    except Exception as e:
        logger.error(f"Failed to load model and artifacts: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")


# ============================================================================
# PREPROCESSING
# ============================================================================

def get_validation_transforms() -> transforms.Compose:
    """
    Return the validation preprocessing pipeline (deterministic, no augmentation).
    
    Must match the transforms used in src/validate.py exactly.
    - Resize to 224x224
    - Convert to tensor
    - Normalize with ImageNet statistics
    
    Returns:
        torchvision.transforms.Compose object.
    """
    return transforms.Compose([
        
    transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )

    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL Image for inference.
    
    Applies validation transforms: resize, convert to tensor, normalize.
    
    Args:
        image: PIL Image object (RGB or grayscale).
    
    Returns:
        Preprocessed tensor of shape (1, 3, 224, 224) ready for model.
    
    Raises:
        ValueError: If image cannot be converted to RGB.
    """
   
  # Ensure 3 channels
    image = image.convert("RGB")

    # Apply transforms
    transform = get_validation_transforms()
    tensor = transform(image)

    # Add batch dimension
    return tensor.unsqueeze(0)



# ============================================================================
# FASTAPI APP & LIFESPAN
# ============================================================================

async def lifespan(app: FastAPI):
    """
    Async context manager for app startup/shutdown.
    
    Loads model and artifacts on startup; cleans up resources on shutdown.
    """
    # Startup
    global model, idx_to_class, treatments, device
    try:
        model, idx_to_class, treatments, device = load_model_and_artifacts()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if model is not None:
        del model
    torch.cuda.empty_cache()


# Create FastAPI app with lifespan
app = FastAPI(
    title="GreenVision Plant Disease Classifier",
    description="Plant leaf disease classification API using EfficientNet-B0",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# ENDPOINTS (to be implemented next)
# ============================================================================

@app.get("/health")
async def health_check() -> Dict:
    """
    Health check endpoint. Confirms API is running and model is loaded.
    
    Returns:
        JSON with status, model_loaded, model_version, and num_classes.
    """
    
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    try:
        # ✅ Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        # ✅ Preprocess
        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.to(device)

        # ✅ Run model
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)

        # ✅ Top-5 predictions
        top_probs, top_indices = torch.topk(probs, 5)

        top_predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = idx_to_class[idx.item()]
            top_predictions.append({
                "class": class_name,
                "confidence": float(prob.item())
            })

        # ✅ Top-1 prediction (FIXED ORDER)
        confidence_tensor, predicted_idx_tensor = torch.max(probs, dim=1)

        confidence = float(confidence_tensor.item())
        predicted_idx = int(predicted_idx_tensor.item())

        predicted_class = idx_to_class[predicted_idx]   # ✅ FIX (no .item())

        # ✅ Treatment lookup
        treatment_info = treatments.get(predicted_class, {})
        plant = predicted_class.split("___")[0]

        # ✅ FIX: define disease_name
        disease_name = treatment_info.get("disease_name", predicted_class)

        
       # ✅ Case 1: Background / Not a plant
        if predicted_class == "Background_without_leaves":
            return {
                "plant": "Unknown",
                "disease": "Not a plant",
                "confidence": confidence,
                "treatment": "This image does not appear to contain a plant leaf.",
                "top_predictions": top_predictions
                }

        # ✅ Case 2: Low confidence (uncertain prediction)
        elif confidence < CONFIDENCE_THRESHOLD:
            return {
                "plant": "Unknown",
                "disease": "Uncertain",
                "confidence": confidence,
                "treatment": "The model is not confident. Please upload a clearer image of a leaf.",
                "top_predictions": top_predictions
                }

        # ✅ Case 3: Normal valid prediction
        return {
            "plant": plant,
            "disease": disease_name,
            "confidence": confidence,
            "treatment": treatment_info.get("treatment", "No treatment available."),
            "top_predictions": top_predictions
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
