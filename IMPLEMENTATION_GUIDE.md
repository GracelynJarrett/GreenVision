# Implementation Guide

---

## 📌 Overview

GreenVision is a deep learning system designed to classify plant leaf diseases from images.

The model uses transfer learning with EfficientNet-B0 and is trained on the PlantVillage dataset, which contains over 54,000 labeled images across 38 classes of plant diseases and healthy leaves. Typical model performance after training reaches ~90–96% validation accuracy using transfer learning.


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

### ✅ Data Splitting Strategy

The dataset is split into three subsets:

- **Training set (75%)** → used to train the model  
- **Validation set (15%)** → used for tuning and early stopping  
- **Test set (10%)** → used for final evaluation  
The test set is held out and used only once for final evaluation after all training and tuning decisions are complete.

**Why this matters:**
- Prevents data leakage
- Ensures unbiased evaluation of model performance
- Separates tuning decisions from final performance reporting

---

**What good looks like:**
- Model performs similarly on validation and test sets
- Test accuracy is close to validation accuracy

---

**What bad looks like:**
- Test accuracy much lower than validation → overfitting
- No test set → unreliable final performance estimate


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


**Starting Metrics:**
- Freeze 100% of EfficientNet feature layers
- Train classifier head for 5–10 epochs
- Learning rate: ~1e-3

**What good looks like:**
- Training accuracy quickly rises above **80–90% within a few epochs**
- Validation accuracy reaches **~85–92%**
- Training and validation curves stay relatively close (no large gap)

If validation accuracy is much lower than training accuracy, this indicates early overfitting or insufficient regularization.

**What bad looks like:**
- Training accuracy increases, but validation accuracy stays **below ~70–75%**
- Large gap between training and validation accuracy (>10%)
- Validation loss stops improving early or increases

This usually indicates:
- Insufficient data augmentation
- Overfitting in the classifier head
- Learning rate too high

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


**Starting Metrics:**
- Unfreeze top 25–50% of layers
- Learning rate: ~1e-4
- Train for 5–15 additional epochs

**What good looks like:**
- Validation accuracy improves by **~2–5% over feature extraction baseline**
- Final validation accuracy reaches **~90–96%**
- Validation loss continues decreasing without large spikes

If validation accuracy drops during fine-tuning, the learning rate is likely too high or too many layers were unfrozen.


**What bad looks like:**
- Validation accuracy drops immediately after unfreezing layers
- Training loss decreases but validation loss increases sharply
- Model becomes unstable (accuracy fluctuates heavily between epochs)

This usually indicates:
- Learning rate is too high for fine-tuning
- Too many layers were unfrozen at once
- Pretrained weights are being overwritten too aggressively

---

## 🎨 Design Decisions

Note: EfficientNet-B0 is a course requirement and is used as the pretrained backbone throughout this project.

### Design Decisions (detailed)

Below are the eight architecture & design decisions required by the deliverable. Each decision includes a short answer, 2–4 sentences of reasoning, and a minimal code snippet showing how the decision maps to implementation.

1) Model — EfficientNet‑B0 & fine‑tuning strategy
- Answer: Use EfficientNet‑B0 (course requirement). Train a custom classifier head for Phase‑1 with the backbone frozen, then unfreeze the top layers and fine‑tune at a lower LR for Phase‑2.
- Reasoning: EfficientNet‑B0 provides strong ImageNet features while remaining computationally efficient; freezing the backbone first stabilizes training and avoids overwriting pretrained features, while selective fine‑tuning adapts high‑level features to plant disease patterns.
- Snippet:

```python
# Freeze backbone (phase-1)
for p in model.features.parameters():
    p.requires_grad = False

# Replace classifier head
model.classifier = nn.Sequential(
    nn.Linear(feat_dim, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, num_classes)
)

# Optimizer for head-only training
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
```

2) Data pipeline — splits & augmentation
- Answer: Split dataset into Train/Val/Test = 75% / 15% / 10%. Use conservative baseline augmentation in `train.py`: RandomResizedCrop, RandomHorizontalFlip, small rotations (±10–15°), mild ColorJitter; validation/inference use Resize → CenterCrop → Normalize.
- Reasoning: The split prevents leakage and separates tuning from final evaluation. Conservative augmentation increases generalization while preserving disease cues; stronger augmentation will be explored in `tune.py` to avoid accidentally destroying diagnostic features.
- Snippet:

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(0.1,0.1,0.1,0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

3) Serving — FastAPI load & class names
- Answer: Load the best checkpoint at startup, set `model.eval()` and `model.to(device)`, and load a persisted `idx_to_class` mapping (JSON) to translate indices to labels.
- Reasoning: Loading at startup minimizes per-request overhead; persisting class mapping guarantees consistent human-readable labels across runs and deployments.
- Snippet:

```python
model = load_model('models/best.pth')
model.eval()
model.to(device)
with open('models/idx_to_class.json') as f:
    idx_to_class = json.load(f)
```

4) Normalization — values & application
- Answer: Use ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) applied to training, validation, and inference pipelines.
- Reasoning: The pretrained EfficientNet expects inputs normalized to the same distribution as ImageNet; applying identical normalization everywhere prevents distribution mismatch between train and inference.
- Snippet:

```python
normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
```

5) Class name persistence — mapping consistency
- Answer: Save `class_to_idx` (from the training dataset) as JSON or artifact; at inference rebuild `idx_to_class = {v:k for k,v in class_to_idx.items()}` and load it in the serving app.
- Reasoning: The model outputs indices; without a persisted mapping the same index could point to different labels across runs, causing incorrect predictions. Persisting the mapping ensures reproducible, human‑readable outputs.
- Snippet:

```python
with open('models/class_to_idx.json','w') as f:
    json.dump(dataset.class_to_idx, f)

# Later in inference
with open('models/class_to_idx.json') as f:
    class_to_idx = json.load(f)
idx_to_class = {v:k for k,v in class_to_idx.items()}
```

6) Low‑confidence predictions — API behavior & dashboard
- Answer: When the top softmax probability < 0.5, the API returns `{"prediction": "uncertain", "confidence": <score>}` and the dashboard highlights the example as low‑confidence for manual review.
- Reasoning: Returning a low‑confidence flag avoids overconfident incorrect labels in production and surfaces ambiguous inputs to the user or annotator for corrective action; the dashboard can aggregate low‑confidence rates per class to guide data collection.
- Snippet:

```python
probs = torch.softmax(output, dim=1)[0]
conf, idx = torch.max(probs, dim=0)
if conf < 0.5:
    return {"prediction": "uncertain", "confidence": float(conf)}
else:
    return {"prediction": idx_to_class[int(idx)], "confidence": float(conf)}
```

7) Error handling — bad uploads & corrupted files
- Answer: API validates `content_type` and returns 400 for non-image uploads; dataset loading wraps `Image.open()` in try/except and logs/skips corrupted files.
- Reasoning: Early validation prevents wasted work and gives clear feedback to users; skipping corrupted files during dataset creation prevents training crashes while logging the filenames for inspection.
- Snippet:

```python
if not file.content_type.startswith('image'):
    raise HTTPException(status_code=400, detail='Invalid file type')
data = await file.read()
try:
    img = Image.open(BytesIO(data)).convert('RGB')
except Exception:
    raise HTTPException(status_code=400, detail='Could not read image')
```

8) Experiment tracking & artifacts (MLflow)
- Answer: Use MLflow to log parameters, metrics, and artifacts (model weights and `class_to_idx`) for every run; store final best models under `models/` and keep run IDs in MLflow for traceability.
- Reasoning: MLflow provides experiment comparability and reproducibility, making it straightforward to compare Phase‑1 vs Phase‑2 results, and to retrieve the exact artifacts used for a reported metric or demo.
- Snippet:

```python
with mlflow.start_run() as run:
    mlflow.log_params(cfg)
    mlflow.log_metric('val_acc', val_acc)
    mlflow.log_artifact('models/best.pth')
    mlflow.log_artifact('models/class_to_idx.json')
```

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

## ⚙️ Hyperparameters (defaults)

The following default values are provided as a starting point. These are tunable via `config.yaml` and `tune.py` — treat them as sensible defaults, not final settings.

- Batch size: 16
- Image size: 224
- Phase-1 (head) epochs: 10
- Phase-2 (fine-tune) epochs: 10
- Base learning rate (phase-1): 1e-3
- Fine-tune learning rate (phase-2): 1e-4
- Dropout (classifier head): 0.5
- Weight decay: 1e-4
- Early stopping patience: 10 epochs

Note: defaults favor quick, stable transfer learning iterations; rely on early stopping and `tune.py` experiments to find the final configuration for production runs.


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


**What good looks like:**
- Small gap between training and validation accuracy (<5–7%)
- Validation loss does not increase while training loss decreases
- Model generalizes well to unseen images

A large gap between training and validation performance indicates overfitting, which can be improved with stronger dropout, augmentation, or weight decay.

**What bad looks like:**
- Training accuracy is very high (95–100%) but validation is much lower
- Validation loss increases while training loss decreases
- Model performs poorly on new/unseen images

This indicates overfitting and suggests:
- Increase dropout
- Add stronger augmentation
- Increase weight decay

---

### ✅ Batch Size

Batch size (16–32) balances:
- Memory usage
- Training speed
- Generalization

**What good looks like:**
- Stable training (loss decreases smoothly)
- No large fluctuations between batches
- Comparable or slightly better validation accuracy with batch size 16 vs 32

If training is noisy or unstable, the batch size may be too small.
If validation accuracy drops with larger batches, generalization may be reduced.

**What bad looks like:**
- Training loss fluctuates heavily (very noisy curves)
- Validation accuracy is inconsistent across epochs
- Out-of-memory errors (batch too large)

Interpretation:
- Too small → unstable gradients
- Too large → weaker generalization and possible performance drop


---
### ✅ Prediction Confidence & Stability

**Starting Metric:**
- Use softmax probabilities to inspect prediction confidence

**What good looks like:**
- Correct predictions have high confidence scores
- Model produces consistent results for similar images

---

**What bad looks like:**
- Low confidence scores (<50%) for many predictions
- Predictions change drastically with small image variations
- Model is overconfident on incorrect predictions

This may indicate poor generalization or insufficient training.



## 🖼️ Data Pipeline

### ✅ Dataset Loading

Images are loaded using PIL and converted to RGB format to ensure consistency:

```python
image = Image.open(path).convert('RGB')
```

Why this matters:

Ensures all images have 3 channels (required for EfficientNet)
Prevents errors from grayscale or corrupted formats

What good looks like:

All images load without errors
Input shape is consistent across the dataset

What bad looks like:

Crashes due to corrupted images
Inconsistent tensor shapes (e.g., missing channels)


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


**Why preserving order matters:**
- The model outputs class indices (0–37), not labels
- If the mapping changes, predictions will be incorrect

To ensure consistency:
- `class_to_idx` is saved during training
- `idx_to_class` is reconstructed and reused during inference

---

**What good looks like:**
- Predictions correctly map to human-readable labels
- Consistent results across training and deployment

---

**What bad looks like:**
- Wrong label returned for correct prediction index
- Inconsistent predictions between runs

This ensures the model's outputs remain stable and interpretable.



---

### ✅ Conservative Augmentation (Baseline)

**Starting Metrics:**
- RandomResizedCrop (scale ~0.8–1.0)
- HorizontalFlip (p=0.5)
- Small rotations (±10–15 degrees)
- Mild color jitter (brightness/contrast/saturation ~0.1–0.2)

**Why this works:**
- PlantVillage images are relatively clean and centered
- Conservative augmentation improves generalization without distorting key disease features
- Prevents the model from learning overly specific patterns while preserving biological realism

---

**What good looks like:**
- Validation accuracy improves compared to no augmentation (**+2–5% gain**)
- Training and validation accuracy remain close (<5–7% gap)
- Model performs well on slightly varied or rotated images
- Training remains stable (loss decreases smoothly)

---

**What bad looks like:**
- No improvement in validation accuracy compared to no augmentation
- Training accuracy drops significantly (<70–80%) early on
- Model struggles to learn basic patterns

This usually indicates:
- Augmentation is too strong (distorting important features)
- Images are being altered beyond realistic plant conditions
- Key visual disease signals (spots, color patterns) are being lost



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


**What good looks like:**
- Training loss steadily decreases
- Validation loss decreases and then plateaus
- Accuracy improves and stabilizes

Typical final performance for PlantVillage with transfer learning:
- Validation accuracy: **~90–96%**
- Balanced performance across classes (no severe class bias)



**What bad looks like:**
- Training loss does not decrease → learning issue (LR too low or bug)
- Validation loss increases while training loss decreases → overfitting
- Accuracy plateaus very early (<80%) → model underfitting or weak features

If all metrics are poor:
- Check data pipeline (normalization, labels, transforms)
- Verify model is in correct mode (`train()` vs `eval()`)

---

### ✅ Step 4: Early Stopping
- Stop training when validation loss stops improving
Early stopping patience: 10 (reduced to 3–5 during rapid experimentation)

---

### ✅ Learning Rate Behavior

**Starting Metrics:**
- Initial LR: 1e-3 (feature extraction), 1e-4 (fine-tuning)
- Optional: ReduceLROnPlateau or step decay

**What good looks like:**
- Loss decreases smoothly
- Model continues improving after early epochs
- No sudden spikes in validation loss

---

**What bad looks like:**
- Loss oscillates heavily → learning rate too high
- Loss decreases very slowly → learning rate too low
- Validation accuracy stagnates early

Learning rate is one of the most sensitive hyperparameters and often the first thing to adjust during debugging.

---

### ✅ Class-Level Performance

**Starting Metric:**
- Monitor accuracy across all 38 classes (if possible)

**What good looks like:**
- Model performs consistently across most classes
- No single class is significantly worse than others
- Confusion between similar diseases is limited

---

**What bad looks like:**
- Certain classes have very low accuracy
- Model predicts a few dominant classes repeatedly
- Confusion between visually similar diseases is high

This may indicate class imbalance or insufficient feature learning for specific diseases.


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

---

## ❓ Uncertainties & Tests

I am intentionally honest about open questions that will be resolved during implementation. Each item below is a testable experiment recorded in MLflow.

- Epochs for Phase‑1 vs Phase‑2: default is 10/10 but this is a starting point. Test: train Phase‑1 baseline, then fine‑tune from the Phase‑1 checkpoint with different epoch counts and compare validation accuracy and val loss curves.
- Augmentation strength (rotation degrees, color jitter): conservative in baseline; test with `tune.py` grid search over rotation magnitude and jitter magnitude, select setting with best val accuracy without loss of diagnostic signal.
- Number of layers to unfreeze: start with top 25–50% layers for fine‑tuning; test progressive unfreeze (top‑k experiments) and compare gains vs. overfitting risk.
- Batch size and LR pairing: defaults given; test a small sweep (batch size 16/24/32 with LR scaling) to find stable training for available GPU memory.

These experiments are recorded as MLflow runs; decisions are finalized only after inspecting run comparisons and learning curves.

## 📚 Citations & Tools

Course materials used:
- Architecture and Design Decisions (course handout provided by professor)

AI assistance disclosure:
- This document was drafted and edited with help from GitHub Copilot and this assistant. I reviewed and adapted all content and verified the code snippets.

---


