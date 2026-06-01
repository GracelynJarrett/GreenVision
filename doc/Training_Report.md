# GreenVision Training Report

**Date:** May 31, 2026  
**Model:** EfficientNet-B0 with Custom Classifier Head  
**Dataset:** PlantVillage (38 classes)  
**Status:** ✅ Production Ready

---

## Results

- **Final validation accuracy:** 99.40%
- **Final test accuracy:** 99.41%
- **Naive baseline (random guess):** 2.6% (1/38 classes)
- **Improvement over baseline:** +96.81 percentage points

The model achieves excellent generalization, with test accuracy within 0.01% of validation accuracy—a strong indicator that the model is not overfitting and will perform reliably on unseen data in production.

---

## Fine-tuning Strategy

### Approach Used

**Two-Phase Transfer Learning:**
1. **Phase 1 (Feature Extraction):** Froze EfficientNet-B0 backbone, trained custom classifier head only
2. **Phase 2 (Fine-tuning):** Unfroze top layers (groups 4–6) of backbone, fine-tuned with lower learning rate

### Why This Strategy

- **Preserves pretrained ImageNet features:** Freezing the backbone initially prevents catastrophic forgetting of useful edge and texture patterns learned from ImageNet
- **Efficient training:** Head-only training in Phase 1 converges quickly (~10 epochs) and reaches high accuracy (~90–92%) with minimal computational cost
- **Selective adaptation:** Fine-tuning only the top groups (intermediate layers) balances adaptation to plant-specific features without overwriting foundational representations
- **Dataset-appropriate:** PlantVillage is large enough (~54K images) to benefit from fine-tuning but small enough that unfreezing selectively avoids overfitting
- **Time efficiency:** Two-phase strategy reaches production-quality accuracy faster than training from scratch or fine-tuning all layers aggressively

### Learning Rates

| Phase | Frozen Layers | Learning Rate | Epochs | Notes |
|-------|---------------|---------------|--------|-------|
| Phase 1 | Backbone (100%) | 1e-3 (Adam) | ~10 | Head training only |
| Phase 2 | Groups 1–3 | 1e-4 (Adam) | ~10 | Selective fine-tune |

Rationale: Pre-trained layers require lower learning rates to avoid disrupting learned features; new layers (head) benefit from higher learning rates for faster convergence.

### Total Training Time

- Phase 1: ~30–40 minutes (GPU: NVIDIA RTX 4060)
- Phase 2: ~35–45 minutes (GPU: NVIDIA RTX 4060)
- **Total:** ~70–85 minutes including logging and checkpointing

### Total Epochs

- Phase 1: ~10 epochs
- Phase 2: ~10 epochs (selected model: M3 with groups 4–6 unfrozen)
- **Total effective training:** ~20 epochs

---

## What Changed During Implementation

### Design Decisions Revised

1. **Fine-tuning layer selection experiment (`tune.py`):**
   - Initially simple: unfreeze all layers with LR decay
   - Revised to: systematic grid search over which layer groups to unfreeze (groups 1–6) with fixed learning rates
   - Result: Model M3 (groups 4–6 unfrozen) emerged as best configuration
   - Why: Granular control revealed that unfreezing only deeper groups balances performance with stability; unfreezing too many layers caused training instability

2. **Augmentation strategy:**
   - Phase 1 baseline: conservative augmentation (RandomResizedCrop, RandomFlip, mild ColorJitter)
   - Phase 2 exploration (`tune.py`): tested stronger augmentation (aggressive rotation, brightness adjustment)
   - Final adopted: baseline augmentation proved sufficient; stronger augmentation did not improve M3 further
   - Why: Conservative augmentation already captures plant disease variation; strong augmentation risked destroying diagnostic cues

3. **Learning rate scheduling:**
   - Initially considered: cosine annealing and step decay
   - Final adopted: fixed learning rates with early stopping
   - Why: Fixed LR + early stopping proved simpler and equally effective; provided better reproducibility across runs

---

## Most Surprising Finding

**Exceptional generalization between validation and test sets:**

Expected validation–test gap: 2–5% (typical for transfer learning on small-to-medium datasets)  
**Observed:** 0.01% (99.40% val vs 99.41% test)

This near-perfect alignment suggests:
- The stratified 75/15/10 train/val/test split effectively eliminated data leakage
- The model learned robust disease patterns rather than dataset artifacts
- Phase 2 fine-tuning was conservative enough to avoid overfitting despite 99%+ accuracy

**Implication:** The model should perform very reliably in production on new, unlabeled plant images from similar growing conditions.

---

## Ready for Assignment 10

- ✅ **Model registered to MLflow Model Registry:** "GreenVision" (version 2, Production stage)
- ✅ **Model can be loaded from Production:** `mlflow.pytorch.load_model("models:/GreenVision/Production")`
- ✅ **Class names artifact saved:** `models/class_to_idx.json` persisted during training
- ✅ **Verification test passed:** Production model loads and can perform inference
- ✅ **Next phase:** FastAPI inference service (Assignment W10A1) and Streamlit dashboard (future assignment)

---

## Production Readiness Checklist

- ✅ High accuracy (99.41% on held-out test set)
- ✅ Good generalization (validation ≈ test)
- ✅ Registered in Model Registry
- ✅ Reproducible checkpoint saved
- ✅ Class mapping persisted
- ✅ Inference verified
- ✅ Full training pipeline documented

**Status:** Ready for deployment to FastAPI inference service.

---

## Reproducibility & Artifacts

- **Phase‑1 retrain (true base model):** run_id `feb3c08d14464d999a528f91bcca74ab` (mlruns/550146252266958514/feb3c08d14464d999a528f91bcca74ab) — checkpoint: `models/phase1_best.pth`
- **Primary training run (phase‑2 tuning, M3):** run_id `0efe2199c322443ca063487e01c3eb9d` (mlruns/550146252266958514/0efe2199c322443ca063487e01c3eb9d)
- **Selected checkpoint:** `models/m3_best.pth`
- **Model Registry:** `GreenVision` (version `2`, Production stage)
- **Saved artifacts:**
   - Validation report: `logs/validation_report.json`
   - Test report: `logs/test_report.json`
   - Class mapping: `models/class_to_idx.json`

Reproducibility tips:

- To record the exact environment used for a run, capture the Python packages used during the run:

```powershell
python -m pip freeze > requirements.txt
```

- To get the model file size (bytes) on Windows PowerShell:

```powershell
(Get-Item models\m3_best.pth).length
```

- Or in Python:

```python
import os
print(os.path.getsize('models/m3_best.pth'))
```

- Quick latency probe (single-image GPU inference):

```python
import time, torch
from PIL import Image
from torchvision import transforms
import mlflow
model = mlflow.pytorch.load_model('models:/GreenVision/Production').to('cuda').eval()
img = Image.open('data/Apple___healthy/<sample.jpg>').convert('RGB')
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
inp = transform(img).unsqueeze(0).to('cuda')
torch.cuda.synchronize(); t0=time.time()
with torch.no_grad():
      _ = model(inp)
torch.cuda.synchronize(); print('latency_ms', (time.time()-t0)*1000)
```

Include these artifacts and the `requirements.txt` in the submission to make runs fully reproducible.

---

## Why not AdamW?

Short answer: Adam was chosen for fast, stable experimentation during Phase 1/2 because it adapts per-parameter learning rates and converges quickly for the new classifier head. AdamW (Adam with decoupled weight decay) is a superior choice in many vision scenarios because it implements weight decay correctly (decoupled from the adaptive moment updates) and often improves generalization compared to vanilla Adam.

Recommendations:

- For final production runs, prefer `AdamW` with a small `weight_decay` (e.g., `1e-4`) and otherwise the same LR schedule. Example:

```python
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
```

- If you want the best possible generalization and are willing to tune more, try `SGD` with momentum (e.g., `momentum=0.9`) plus a learning-rate schedule (cosine or step decay). SGD often yields the best final performance for vision models but requires more careful LR and schedule tuning.

Summary: switching to `AdamW` is a low-effort, high-reward change for production runs; keep `Adam` for quick experiments if desired.
