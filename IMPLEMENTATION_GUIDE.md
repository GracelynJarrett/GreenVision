# Implementation Guide

---

## 📌 Overview

GreenVision is a deep learning system designed to classify plant leaf diseases from images.

The model uses transfer learning with EfficientNet-B0 and is trained on the PlantVillage dataset, which contains over 54,000 labeled images across 38 classes of plant diseases and healthy leaves.

The system includes:
- A convolutional neural network for classification
- MLflow for experiment tracking
- A FastAPI service for model deployment and prediction

---

## 🚀 Quick Start

Train the model (example):

```bash
python train.py --config configs/train.yaml
```

Run the API:

```bash
uvicorn app:app --reload --port 8000
```

---

## 📁 Project Structure

- `data/` — dataset (PlantVillage)
- `models/` — saved model weights
- `src/` — training and API code
- `train.py` — training script
- `app.py` — FastAPI server
- `IMPLEMENTATION_GUIDE.md` — this guide

---

## 📊 Dataset

GreenVision uses the PlantVillage dataset:
- 54,306 images
- 38 classes (disease + healthy)
- 14 crop types

Images are organized into folders, where each folder represents a class.

Class mappings are saved during training to ensure correct predictions during inference.

---

## ⚙️ Architecture

### ✅ Base Model

GreenVision uses **EfficientNet-B0**, a convolutional neural network pretrained on the ImageNet dataset.

EfficientNet is designed to scale efficiently and provides strong performance while remaining computationally efficient.

---

### ✅ Transfer Learning Strategy

The model reuses features learned from ImageNet and adapts them to plant disease classification.

Architecture flow:

Input Image → EfficientNet Feature Extractor → Custom Classifier → Output (38 classes)

---

### ✅ Feature Extraction (Freezing Layers)

Initially, the EfficientNet feature layers are **frozen**, meaning their weights are not updated.

This:
- Preserves learned features (edges, textures, patterns)
- Reduces training time
- Prevents overfitting

---

### ✅ Custom Classification Head

The original classifier is replaced with:

- Fully connected layer (input → 128)
- ReLU activation
- Dropout (0.5)
- Output layer (128 → 38 classes)

---

### ✅ Fine-Tuning

After initial training:
- Some deeper layers are unfrozen
- Model is trained with a lower learning rate

This improves performance on plant-specific features.

---

## 🎨 Design Decisions

### ✅ EfficientNet-B0

Chosen for:
- High accuracy with low computational cost
- Strong pretrained features from ImageNet

---

### ✅ Transfer Learning

Used because:
- Dataset is smaller than ImageNet
- Faster training
- Better generalization

---

### ✅ Data Augmentation

Training data includes:
- RandomResizedCrop
- Horizontal/Vertical flips
- Small rotations
- Color jitter

These simulate real-world variations while preserving labels.

---

## Inference & Serving (FastAPI)

Guidelines:
- Ensure `model.eval()` and correct device mapping.
- Validate input content-type and enforce file-size limits.
- Use a saved `idx_to_class` mapping to return human-readable labels.

Minimal predict endpoint (concept):

```python
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image'):
        raise HTTPException(status_code=400, detail='Invalid file type')
    data = await file.read()
    image = Image.open(BytesIO(data)).convert('RGB')
    tensor = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = int(output.argmax(dim=1).item())
    return {'prediction': idx_to_class[pred]}
```

---

### ✅ Normalization

Images are normalized using ImageNet values:

- Mean: [0.485, 0.456, 0.406]  
- Std: [0.229, 0.224, 0.225]

---

### ✅ Train vs Validation

- Training → randomized transforms  
- Validation → fixed transforms (Resize + CenterCrop)

Ensures fair evaluation.

---

### ✅ Regularization

To prevent overfitting:
- Dropout
- Weight decay
- Early stopping

---

### ✅ Batch Size

Batch size (16–32) balances:
- Memory usage
- Training speed
- Generalization

---

## 🖼️ Data Pipeline

### ✅ Dataset Loading

Images are loaded using a folder-based structure.

---

### ✅ Transformations

**Training:**
- RandomResizedCrop
- Flips
- Color jitter
- Normalize

**Validation / Inference:**
- Resize
- CenterCrop
- Normalize

---

### ✅ Class Mapping

Class indices are saved: idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
Used during inference.

---

## 🏁 Training Strategy

### ✅ Step 1: Feature Extraction
- Freeze EfficientNet
- Train classifier only

---

### ✅ Step 2: Fine-Tuning
- Unfreeze deeper layers
- Use lower learning rate

---

### ✅ Step 3: Monitoring
- Track loss and accuracy
- Check for overfitting

---

### ✅ Step 4: Early Stopping
- Stop training when validation loss stops improving

---

## 🧪 Experiment Tracking (MLflow)

### ✅ Purpose
- Track experiments
- Compare models
- Ensure reproducibility

---

### ✅ Logged Data

**Parameters:**
- Learning rate
- Batch size
- Epochs

**Metrics:**
- Train loss
- Validation loss
- Accuracy

---

### ✅ Artifacts
- Model weights
- Class mapping

---

## ⚠️ Error Handling

### Common issues and fixes

Corrupted images

```python
try:
    image = Image.open(path)
except Exception:
    # log the filename and skip
    continue
```

Shape mismatch

Ensure input tensor shape: `[batch, 3, 224, 224]`.

Class mapping errors

Save and reuse `idx_to_class` (persist as an artifact and load during inference).

Normalization issues

Use the same preprocessing pipeline for validation and inference.

Overfitting

Mitigation: stronger augmentation, dropout, weight decay, and early stopping.

Device mismatch

Ensure `model.to(device)` and `inputs = inputs.to(device)` before forward pass.

