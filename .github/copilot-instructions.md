## GreenVision Copilot Instructions

### Project Description
GreenVision is a plant leaf disease classification project for the Applied Machine Learning course. It predicts disease labels from PlantVillage images using transfer learning with EfficientNet-B0, a custom classifier head, and a FastAPI inference service.

### Dataset Details
- The dataset follows a folder-based `ImageFolder` structure.
- PlantVillage contains 38 classes of plant diseases and healthy leaves.
- Class names must remain consistent across training, validation, and inference.
- The label mapping must be saved during training and reused during serving.

### Critical Constants
- Image size: 224x224
- Number of classes: 38
- EfficientNet feature dimension: 1280
- ImageNet normalization mean: [0.485, 0.456, 0.406]
- ImageNet normalization std: [0.229, 0.224, 0.225]

### Code Conventions
- Do not rewrite or edit an entire file in one step.
- Work one section or function at a time when making changes.
- Add docstrings for every public method, function, and class.
- Add comments to new or edited code when the intent is not obvious.
- Use clear, consistent import ordering.
- Prefer small, readable functions over large monolithic blocks.
- Handle errors explicitly; do not fail silently.
- Keep training, tuning, validation, serving, and utility code separated by responsibility.

### Architecture Notes
- The training workflow uses two phases: feature extraction first, then fine-tuning.
- Phase 1 freezes the EfficientNet backbone and trains the classifier head.
- Phase 2 unfreezes deeper layers and fine-tunes with a lower learning rate.
- Keep the ImageNet normalization values unchanged.
- Keep the class-name artifact structure unchanged so inference stays consistent.
- Keep the training sequence consistent unless we explicitly agree to change it.

### Working Rules
- Do not create new files without asking first.
- Before creating a new markdown note, discuss what it should contain, why it is needed, and how it connects to the previous step.
- If it is unclear what to do next, ask before making changes.
- Each major code file may have its own companion markdown note, but only create the note after we discuss and agree on the scope.

### What to Prefer
- Small, incremental edits.
- Reusable helpers for repeated logic.
- Explicit validation steps after changes.
- Consistent handling of image preprocessing across training and inference.

### Implementation Logging
- When asked to log today's events or update the implementation log, create or append an entry to `doc/Implementation_Log.md`.
- Each log entry must have: name, date, branch, overview (1–2 sentences), explanation (3+ sentences), files touched, and next step.
- Newest entries appear at the top of the log file.
- The overview should concisely summarize what was accomplished.
- The explanation should detail the reasoning, changes made, and how the work connects to previous steps.
- Use clear, professional language; treat the log as project documentation.

