# Train Workflow

This note tracks the current training entrypoint in `src/train.py` and how it fits the GreenVision training pipeline.

## Purpose

`train.py` is the Phase 1 training orchestrator. Its job is to load configuration, prepare the data pipeline, build the model, and set up the optimization objects that will be used by the training loop.

This file is intentionally kept small so the training logic can be expanded in later steps without turning the entrypoint into a monolithic script.

## Current Responsibilities

The current implementation does the following:

1. Loads YAML configuration from `config.yaml` or a user-supplied path.
2. Sets up logging, including optional file logging in the configured logs directory.
3. Builds train, validation, and test `DataLoader` objects from the PlantVillage dataset.
4. Persists the class-to-index mapping for use during inference.
5. Instantiates the `DiseaseClassifier`, moves it to the active device, and prepares the Phase 1 optimizer and loss function.
6. Starts an MLflow run, logs configuration and metrics, and stores checkpoints as artifacts.

## Data Flow

The expected training flow is:

```text
config.yaml -> dataset loading -> class mapping save -> model setup -> optimizer + loss setup -> training loop
```

The dataset split is expected to stay consistent with the project guide:

- Training: 75%
- Validation: 15%
- Test: 10%

## Configuration Contract

`train.py` currently expects the following configuration sections:

- `paths.data_dir` for the PlantVillage root directory
- `paths.logs_dir` for log output
- `paths.class_to_idx` for the saved class mapping artifact
- `project.image_size` for image resizing
- `project.num_classes` for the classifier output size
- `project.seed` for reproducible splitting
- `training.batch_size`
- `training.num_workers`
- `training.phase1.learning_rate`
- `training.phase1.weight_decay`
- `logging.level`
- `logging.log_to_file`

## Artifact Contract

The training step must preserve the class mapping artifact so inference stays consistent with training.

- Save `class_to_idx` during dataset preparation.
- Reuse the saved mapping during serving.
- Keep the class order stable across training, validation, and inference.

## Planned Expansion

The next training-related additions should extend this file as they are implemented:

- Phase 1 training loop
- Checkpoint saving
- Validation metrics and early stopping
- Phase 2 fine-tuning

## Notes

The current code already reflects the project-level constraints from `IMPLEMENTATION_GUIDE.md`:

- EfficientNet-B0 backbone
- 224x224 input size
- 38 output classes
- ImageNet normalization
- Two-phase training strategy

When `train.py` grows, this note should be updated alongside the code so the documentation continues to match the actual orchestration flow.