# Validation Workflow

This note describes the planned Phase 3 validation entrypoint, `src/validate.py`, and how it fits into the GreenVision pipeline after model selection.

## Purpose

`validate.py` will be the Phase 3 analysis script for the selected checkpoint, currently M3. Its job is to load the trained model, run deterministic evaluation on the validation split, and produce the detailed reports needed to inspect model behavior before final testing.

This file exists so validation stays separate from training. The goal is to keep Phase 3 focused on analysis, not optimization or additional tuning.

## Current Goal

The Phase 3 branch should support the following validation outputs:

1. Confusion matrix across all 38 PlantVillage classes.
2. Per-class accuracy summary.
3. List or table of the most common misclassifications.
4. A small sample of prediction-vs-ground-truth comparisons for manual review.
5. Saved artifacts or logs that make the results easy to inspect later.

## Expected Data Flow

```text
config.yaml -> load class mapping -> build deterministic validation dataset -> load selected checkpoint -> run predictions -> compute validation reports -> save analysis artifacts
```

The validation path should reuse the same preprocessing rules as inference:

- Image size: 224x224
- ImageNet normalization mean: [0.485, 0.456, 0.406]
- ImageNet normalization std: [0.229, 0.224, 0.225]
- Stable class ordering from the saved `class_to_idx` artifact

## Configuration Contract

`validate.py` should read the shared configuration and expect these settings:

- `paths.data_dir` for the PlantVillage dataset root
- `paths.class_to_idx` for loading the saved class mapping
- `paths.checkpoint_phase2` or the chosen M3 checkpoint path, depending on branch setup
- `paths.logs_dir` for validation logs
- `project.image_size` for resizing
- `project.num_classes` for the classifier output size
- `validation.batch_size`
- `validation.num_workers`
- `validation.pin_memory`
- `validation.save_confusion_matrix`
- `validation.evaluate_each_phase`

## Planned Responsibilities

The validation entrypoint should do the following:

1. Load configuration and set up logging.
2. Load the saved class mapping so class indices can be translated back to class names.
3. Build the validation split with deterministic transforms.
4. Load the chosen checkpoint into `DiseaseClassifier`.
5. Run inference over the validation set without gradient tracking.
6. Aggregate predictions into confusion-matrix and per-class metrics.
7. Save or print the analysis results in a readable format.

## Artifact Contract

Phase 3 should preserve the same class mapping artifact used by training and serving.

- Reuse `class_to_idx` instead of rebuilding class order manually.
- Keep the 38-class assumption unchanged.
- Keep the validation preprocessing identical to the project normalization rules.

If the branch saves derived reports, those outputs should be treated as validation artifacts, not model checkpoints.

## Planned Expansion

The next additions for this file should follow the implementation of `src/validate.py`:

- Confusion matrix generation
- Per-class accuracy calculation
- Misclassification sampling and reporting
- Optional visualization helpers for validation review
- Tests for the validation helpers and report generation

## Notes

This note intentionally mirrors the structure of `train.md` so the Phase 3 branch stays consistent with the rest of the repository documentation. Once `validate.py` exists, this file should be updated to match the actual code paths and artifact names.