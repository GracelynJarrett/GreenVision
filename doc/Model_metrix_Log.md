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