# Test Workflow

This note describes the planned Phase 4 testing entrypoint, `src/test.py`, and how it fits into the GreenVision pipeline for final model certification.

## Purpose

`test.py` will be the Phase 4 certification script for the selected checkpoint, currently M3. Its job is to load the trained model, run deterministic evaluation on the held-out test split, and produce the final performance report needed to certify the model before deployment.

This file exists so final testing stays separate from validation. The goal is to keep Phase 4 focused on model certification, not exploratory analysis or additional tuning.

## Current Goal

The Phase 4 branch should support the following test outputs:

1. Overall test accuracy and loss across all test samples.
2. Confusion matrix on the held-out test set.
3. Per-class accuracy and performance summary.
4. List of the most common misclassifications on the test set.
5. Saved artifacts or logs for the final certification report.

## Expected Data Flow

```text
config.yaml -> load class mapping -> build deterministic test dataset -> load selected checkpoint -> run predictions -> compute test reports -> save final artifacts
```

The test path should reuse the same preprocessing rules as inference and validation:

- Image size: 224x224
- ImageNet normalization mean: [0.485, 0.456, 0.406]
- ImageNet normalization std: [0.229, 0.224, 0.225]
- Stable class ordering from the saved `class_to_idx` artifact

## Configuration Contract

`test.py` should read the shared configuration and expect these settings:

- `paths.data_dir` for the PlantVillage dataset root
- `paths.class_to_idx` for loading the saved class mapping
- `paths.checkpoint_phase2` or the chosen M3 checkpoint path
- `paths.logs_dir` for test logs
- `project.image_size` for resizing
- `project.num_classes` for the classifier output size
- `testing.batch_size` or fallback to `validation.batch_size`
- `testing.num_workers` or fallback to `validation.num_workers`

## Planned Responsibilities

The test entrypoint should do the following:

1. Load configuration and set up logging.
2. Load the saved class mapping for consistent label ordering.
3. Build the test split with deterministic transforms.
4. Load the chosen checkpoint into `DiseaseClassifier`.
5. Run inference over the test set without gradient tracking.
6. Aggregate predictions into confusion-matrix and per-class metrics.
7. Save the final test report in a readable format.

## Artifact Contract

Phase 4 should preserve the same class mapping artifact and preprocessing rules used by training, validation, and serving.

- Reuse `class_to_idx` instead of rebuilding class order manually.
- Keep the 38-class assumption unchanged.
- Keep the test preprocessing identical to the project normalization rules.

All test outputs should be treated as final certification artifacts.

## Planned Expansion

The next additions for this file should follow the implementation of `src/test.py`:

- Test confusion matrix generation
- Per-class test accuracy calculation
- Test-set misclassification sampling and reporting
- Optional visualization helpers for final review
- Tests for the test helpers and report generation

## Notes

This note intentionally mirrors the structure of `validate.md` so the Phase 4 branch stays consistent with Phase 3 and the rest of the repository documentation. Once `test.py` exists, this file should be updated to match the actual code paths and artifact names.
