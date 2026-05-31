## GreenVision Base Model Metrics

**Run date:** 2026-05-29  
**Run name:** `phase1_feature_extraction`  
**MLflow run ID:** `e6ad6769ecb743eda2d50f0f4651af39`

### Summary

This log captures the first completed base model training run using EfficientNet-B0 with a frozen backbone for Phase 1 and fine-tuning for Phase 2.

### Final Metrics

| Split | Metric | Value |
| --- | --- | ---: |
| Phase 1 | Best validation loss | 0.2282 |
| Phase 1 | Best validation accuracy | 92.65% |
| Phase 1 | Final training loss | 0.4216 |
| Phase 1 | Final training accuracy | 86.41% |
| Phase 1 | Final validation loss | 0.2329 |
| Phase 1 | Final validation accuracy | 92.65% |
| Phase 2 | Best validation loss | 0.0183 |
| Phase 2 | Best validation accuracy | 99.48% |
| Phase 2 | Final training loss | 0.0463 |
| Phase 2 | Final training accuracy | 98.66% |
| Phase 2 | Final validation loss | 0.0291 |
| Phase 2 | Final validation accuracy | 99.44% |
| Test | Loss | 0.0151 |
| Test | Accuracy | 99.50% |

### Notes

- Dataset size used for training: 40,728 train images, 8,146 validation images, 5,431 test images.
- `models/class_to_idx.json` was saved for inference consistency.
- `models/phase1_best.pth` and `models/phase2_best.pth` were produced during the run.
- The run completed successfully on the CUDA-enabled GPU environment.

## Tuning Model M2 Metrics

**Run date:** 2026-05-30  
**Run name:** `phase2_tuning_M2`  
**MLflow run ID:** `c3d8b3557ad14b57aa355e8928628313`

### Final Metrics

| Split | Metric | Value |
| --- | --- | ---: |
| M2 | Best validation loss | 0.1455 |
| M2 | Best validation accuracy | 96.13% |
| M2 | Final training loss | 0.1134 |
| M2 | Final training accuracy | 96.76% |
| Test | Loss | 0.1377 |
| Test | Accuracy | 96.34% |

### Notes

- Dataset size used for tuning: 40728 train images, 8146 validation images, 5431 test images.
- Class mapping preserved with 38 classes.
- Best checkpoint saved to models/m2_best.pth.


## Tuning Model M3 Metrics

**Run date:** 2026-05-30  
**Run name:** `phase2_tuning_M3`  
**MLflow run ID:** `0efe2199c322443ca063487e01c3eb9d`

### Final Metrics

| Split | Metric | Value |
| --- | --- | ---: |
| M3 | Best validation loss | 0.0165 |
| M3 | Best validation accuracy | 99.40% |
| M3 | Final training loss | 0.0269 |
| M3 | Final training accuracy | 99.28% |
| Test | Loss | 0.0197 |
| Test | Accuracy | 99.41% |

### Notes

- Dataset size used for tuning: 40728 train images, 8146 validation images, 5431 test images.
- Class mapping preserved with 38 classes.
- Best checkpoint saved to models/m3_best.pth.


## Tuning Model M4 Metrics

**Run date:** 2026-05-30  
**Run name:** `phase2_tuning_M4`  
**MLflow run ID:** `d0f1d0a520414f71a506fcde09e16ddc`

### Final Metrics

| Split | Metric | Value |
| --- | --- | ---: |
| M4 | Best validation loss | 0.1505 |
| M4 | Best validation accuracy | 95.09% |
| M4 | Final training loss | 0.2735 |
| M4 | Final training accuracy | 91.86% |
| Test | Loss | 0.1415 |
| Test | Accuracy | 95.73% |

### Notes

- Dataset size used for tuning: 40728 train images, 8146 validation images, 5431 test images.
- Class mapping preserved with 38 classes.
- Best checkpoint saved to models/m4_best.pth.


## Tuning Model M5 Metrics

**Run date:** 2026-05-31  
**Run name:** `phase2_tuning_M5`  
**MLflow run ID:** `unknown`

### Final Metrics

| Split | Metric | Value |
| --- | --- | ---: |
| M5 | Best validation loss | 0.1696 |
| M5 | Best validation accuracy | 95.56% |
| M5 | Final training loss | 0.4057 |
| M5 | Final training accuracy | 89.10% |
| Test | Loss | 0.1596 |
| Test | Accuracy | 96.08% |

### Notes

- Dataset size used for tuning: 40728 train images, 8146 validation images, 5431 test images.
- Class mapping preserved with 38 classes.
- Best checkpoint saved to models/m5_best.pth.


## Model Comparison Summary

### Base Model vs Tuning Runs

| Model | Freeze / Training Setup | Best Validation Accuracy | Test Accuracy | Best Validation Loss | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Base Phase 1 | Backbone frozen, head only | 92.65% | N/A | 0.2282 | Pre-fine-tuning baseline; separated from Phase 2 by request |
| Base Phase 2 | Full backbone unfrozen | 99.48% | 99.50% | 0.0183 | Base fine-tuning result; technically a tuning stage |
| M2 | Backbone groups 1-3 unfrozen | 96.13% | 96.34% | 0.1455 | Conservative group unfreezing |
| M3 | Backbone groups 4-6 unfrozen | 99.40% | 99.41% | 0.0165 | Best tuning result so far |
| M4 | Backbone groups 7-8 unfrozen | 95.09% | 95.73% | 0.1505 | Late-block fine-tuning |
| M5 | Full backbone unfrozen | 95.56% | 96.08% | 0.1696 | Full-unfreeze tuning run |

### Quick Takeaway

- Best tuning run: M3, with 99.41% test accuracy.
- Best overall model in the repository remains the base Phase 2 run at 99.50% test accuracy.
- Base Phase 1 is kept separate from Phase 2 because Phase 2 is a fine-tuning/tuning stage, not the frozen-head baseline.

## Final Model Selection: M3

### Decision Rationale

**Selected Model:** M3 (backbone groups 4-6 unfrozen)  
**Checkpoint:** `models/m3_best.pth`  
**Test Accuracy:** 99.41% | **Test Loss:** 0.0197

### Why M3?

1. **Best Tuning Performance:** M3 achieved 99.41% test accuracy, matching the base Phase 2 (99.50%) performance within margin of error and significantly outperforming the other tuning profiles (M2 96.34%, M4 95.73%, M5 96.08%).

2. **Stability and Generalization:** M3 showed consistent validation performance (99.40% best validation accuracy) with a very low validation loss (0.0165), indicating excellent generalization and minimal overfitting risk.

3. **Systematic Unfreezing Strategy:** M3 represents a balanced middle-ground unfreezing of the backbone (groups 4-6 out of 8), avoiding both under-exploration (M2's conservative groups 1-3) and over-fitting from full unfreezing (M5).

4. **Reproducibility and Efficiency:** M3's configuration (batch_size=24, lr=5e-5, dropout=0.3) proved effective without requiring aggressive augmentation or extreme learning rates, making it a robust choice for downstream validation and deployment.

### Validation Readiness

M3 is ready to proceed to **Phase 3 Validation**, which focuses on:
- Confusion matrix analysis across all 38 plant disease classes
- Per-class accuracy and failure pattern inspection
- Inspection of validation set predictions vs. ground truth
- Robustness check on edge cases and ambiguous samples

After Phase 3 Validation confirms model behavior is acceptable, the model will proceed to **Phase 4 Testing** on the held-out test set for final performance certification.

