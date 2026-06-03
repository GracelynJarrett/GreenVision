# FastAPI Inference Endpoint (serve.py)

## Overview

`serve.py` implements a FastAPI server that exposes the GreenVision model for real-time plant disease prediction. The server loads the trained EfficientNet-B0 model and class mappings from MLflow Registry, preprocesses uploaded images using the same transforms as validation, and returns disease predictions with confidence scores and treatment recommendations.

**Key responsibilities:**
- Load model from MLflow: `models:/GreenVision/Production`
- Load class-name mappings from MLflow artifacts (invert `class_to_idx.json`)
- Validate and preprocess image uploads
- Run inference in deterministic mode (eval mode, no augmentation)
- Return structured predictions with confidence flags and treatment data
- Provide health checks for monitoring

---

## Architecture Flow

```
[Image Upload] 
    ↓
[FastAPI /predict endpoint]
    ↓
[Validate file type & decode]
    ↓
[Preprocess: Resize to 224×224, normalize with ImageNet stats]
    ↓
[Run model inference (eval mode)]
    ↓
[Extract top class & confidence]
    ↓
[Invert class_to_idx to get disease name]
    ↓
[Load treatment from config/treatments.yaml]
    ↓
[Construct JSON response]
    ↓
[Return to client]
```

---

## Endpoints

### GET /health

**Purpose:** Confirms the API is reachable and the model is loaded in memory.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "Production",
  "num_classes": 38
}
```

**Response (503 Service Unavailable):**
If model fails to load on startup:
```json
{
  "status": "unhealthy",
  "error": "Model failed to load from MLflow"
}
```

---

### POST /predict

**Purpose:** Accept an image upload and return a disease prediction.

**Request:**
- Content-Type: `multipart/form-data`
- Field name: `file` (required)
- Accepted MIME types: `image/jpeg`, `image/png`, `image/gif`, `image/webp`
- Max file size: 10 MB (configurable in `serve.py`)

**Response (200 OK):**
```json
{
  "plant_type": "Tomato",
  "disease": "Late blight",
  "confidence": 0.94,
  "low_confidence": false,
  "is_healthy": false,
  "treatment": "Remove affected foliage; apply copper fungicide; improve air circulation.",
  "model_version": "Production",
  "raw_class": "Tomato___Late_blight"
}
```

**Response (400 Bad Request):**

*Non-image file:*
```json
{
  "error": "Invalid file type. Expected image (JPEG, PNG, GIF, WebP)."
}
```

*Corrupted image:*
```json
{
  "error": "Unable to decode image. File may be corrupted."
}
```

**Response (413 Payload Too Large):**
```json
{
  "error": "File size exceeds 10 MB limit."
}
```

**Response (500 Internal Server Error):**
```json
{
  "error": "Inference failed. Please try again."
}
```

---

## Error Handling Strategy

| Scenario | Status | Action | Rationale |
|----------|--------|--------|-----------|
| Non-image file (PDF, text, etc.) | 400 | Reject with clear message | Prevent wasted inference |
| Corrupted/unreadable image data | 400 | Reject with clear message | Graceful failure for invalid input |
| Confidence < 40% | 200 | Return prediction + `"low_confidence": true` | Dashboard warns user but still shows result + plant_type |
| Unknown disease (plant recognized, disease not in training) | 200 | Return best guess + `"low_confidence": true` | Farmer knows plant type; should seek expert advice |
| Model not loaded on startup | 503 | Return error, exit process | Fail-fast; dashboard knows API is down |
| Inference exception | 500 | Log error, return generic message | Prevents leaking internal details |

**Plant Type Extraction:**
The `plant_type` field is extracted from the class name by splitting on the first `___`. The `disease` field contains just the disease part (second half), formatted with title case and underscores replaced by spaces.

Examples:
- `Tomato___Late_blight` → `plant_type: "Tomato"`, `disease: "Late blight"`
- `Apple___Apple_scab` → `plant_type: "Apple"`, `disease: "Apple scab"`
- `Pepper,_bell___Bacterial_spot` → `plant_type: "Pepper, bell"`, `disease: "Bacterial spot"`
- `Background_without_leaves` → `plant_type: "Background"`, `disease: "Without leaves"`

This allows the dashboard to show context even when disease confidence is low: *"This appears to be a [plant_type]. Disease detected: [disease], but confidence is low. Please consult a local agricultural extension agent."*

---

## Preprocessing Pipeline

**Constraints:**
- Must match validation transforms **exactly** — same order, same parameters
- No data augmentation (deterministic inference)
- Input: PIL Image from file upload
- Output: Normalized tensor ready for model

**Steps:**
1. Decode uploaded file → PIL Image
2. Resize to 224×224 (preserve aspect ratio with padding, or center crop — match validation)
3. Convert to RGB if grayscale
4. Convert to tensor
5. Normalize with ImageNet stats:
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`

**Reference:** Mirror the transforms applied in `src/validate.py` or the validation DataLoader in `src/dataset.py`.

---

## MLflow Model & Artifact Loading

### Startup Procedure

1. **Connect to MLflow Tracking Server:**
   - Read MLflow URI from environment or default to local `mlruns/`
   
2. **Load Model:**
   - Use `mlflow.pytorch.load_model("models:/GreenVision/Production")`
   - Verify model is in `eval()` mode
   - Move to appropriate device (GPU if available, else CPU)

3. **Load Class Mappings:**
   - Download `class_to_idx.json` artifact from the Production version
   - Invert mapping: `idx_to_class = {v: k for k, v in class_to_idx.items()}`
   - Validate: ensure keys are 0 to 37 (38 classes total)

4. **Load Treatments:**
   - Load `config/treatments.yaml`
   - Validate: all classes in `idx_to_class` have treatment entries

**Pseudocode:**
```python
def load_model_and_artifacts():
    client = mlflow.MlflowClient()
    model = mlflow.pytorch.load_model("models:/GreenVision/Production")
    model.eval()
    
    # Fetch MLflow artifacts
    artifact_uri = client.get_registered_model("GreenVision").latest_versions[0].source
    class_to_idx = load_artifact(artifact_uri, "class_to_idx.json")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    treatments = load_yaml("config/treatments.yaml")
    return model, idx_to_class, treatments
```

---

## Treatment Recommendations

**Storage:** `config/treatments.yaml`

**Format:**
```yaml
Apple___Apple_scab:
  disease_name: "Apple Scab"
  treatment: "Apply fungicide in spring; prune infected branches."

Tomato___Late_blight:
  disease_name: "Tomato Late Blight"
  treatment: "Remove affected foliage; apply copper fungicide; improve air circulation."

Tomato___healthy:
  disease_name: "Healthy Tomato"
  treatment: "Continue routine monitoring and preventive care."
```

**Key principles:**
- Keyed by raw class name (as stored in `class_to_idx.json`)
- `disease_name`: Human-readable, title-cased version for display
- `treatment`: Actionable recommendation for a farmer (not too technical, not too vague)
- All 38 classes must have entries

**Sourcing:** These should be researched from agricultural extension offices or domain experts, not generated. Placeholder recommendations are acceptable for MVP.

---

## Request/Response Schema (Final)

### /predict Response (200 OK)

```json
{
  "plant_type": "Tomato",
  "disease": "Late blight",
  "confidence": 0.94,
  "low_confidence": false,
  "is_healthy": false,
  "treatment": "Remove affected foliage; apply copper fungicide; improve air circulation.",
  "model_version": "Production",
  "raw_class": "Tomato___Late_blight"
}
```

**Field definitions:**
- `plant_type` (str): Plant type extracted from class name (e.g., "Tomato", "Blueberry", "Corn")
- `disease` (str): Disease name formatted for display (disease part only, title case, no underscores). Examples: "Late blight", "Apple scab", "Healthy"
- `confidence` (float): Softmax probability of predicted class [0.0, 1.0]
- `low_confidence` (bool): `true` if confidence < 40%, else `false`
- `is_healthy` (bool): `true` if class name contains "healthy" (case-insensitive)
- `treatment` (str): Treatment recommendation from config
- `model_version` (str): MLflow model version deployed (e.g., "Production")
- `raw_class` (str): The actual class name from training (for debugging)

---

## Implementation Checklist

Before moving to integration testing:

- [ ] `/health` endpoint returns correct schema and model status
- [ ] `/predict` validates file type (MIME check)
- [ ] Corrupted images are rejected with 400
- [ ] Preprocessing matches `src/validate.py` transforms exactly
- [ ] Model runs in `eval()` mode during inference
- [ ] `idx_to_class` inversion is correct (verify all 38 classes)
- [ ] Confidence threshold (40%) triggers `low_confidence` flag
- [ ] Treatment keys match all classes in the model
- [ ] Response includes `raw_class` for debugging
- [ ] Error responses are informative (not exception stack traces)
- [ ] Server starts and handles concurrent requests
- [ ] MLflow model URIs are environment-agnostic (use env vars if needed)

---

## Configuration & Environment

**Environment variables (optional):**
```bash
MLFLOW_TRACKING_URI=<local or remote MLflow server>
MODEL_DEVICE=cpu  # or 'cuda'
TREATMENTS_PATH=config/treatments.yaml
MAX_FILE_SIZE_MB=10
```

If not set, defaults:
- MLFLOW_TRACKING_URI: `./mlruns` (local)
- MODEL_DEVICE: auto-detect (GPU if available)
- TREATMENTS_PATH: `config/treatments.yaml`
- MAX_FILE_SIZE_MB: 10

---

## Testing Strategy

*See `tests/test_serve.py` for integration tests:*
- Healthy leaf upload → high confidence, `is_healthy=true`
- Diseased leaf upload → correct disease, treatment match
- Invalid file type → 400 error
- Corrupted image → 400 error
- Low confidence scenario (if crafted) → `low_confidence=true`
- Deterministic inference (same image twice → same result)
- Concurrent `/predict` requests → no race conditions

