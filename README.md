# GreenVision

GreenVision is a plant leaf disease classification project created for the Applied Machine Learning course (Spring 2026). It predicts disease labels from PlantVillage images using transfer learning with EfficientNet-B0, a custom classifier head, and a FastAPI inference service.

## Features

- Transfer learning with EfficientNet-B0 backbone (ImageNet pretrained)
- Two-phase training strategy: feature extraction (frozen backbone) followed by fine-tuning
- Data augmentation and validation pipeline with ImageNet normalization
- MLflow experiment tracking for reproducibility
- FastAPI service for model inference with confidence scoring
- Comprehensive implementation guide with architectural decisions and code snippets

## Repository Structure

- `IMPLEMENTATION_GUIDE.md` — detailed architecture decisions and design rationale
- `.github/copilot-instructions.md` — AI assistant context
- `.github/agent.md` — autonomous agent guardrails
- `doc/` — project documentation
  - `Implementation_Log.md` — progress tracking
  - `Research_Log.md` — research notes
- `data/` — PlantVillage dataset (ignored)
- `models/` — saved model checkpoints (ignored)
- `src/` — source code modules
- `mnist_cnn.ipynb` — reference notebook (legacy)

## Setup

1. Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.\.venv\Scripts\Activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

Required packages: PyTorch, torchvision, EfficientNet, FastAPI, MLflow, Pillow, scikit-learn, and numpy.

## Usage

### Training (Phase 1 + Phase 2)

```bash
python train.py --config configs/train.yaml
```

### Hyperparameter Tuning

```bash
python tune.py --config configs/tune.yaml
```

### Model Validation

```bash
python validate.py --model models/best.pth
```

### Running the Inference Service

```bash
uvicorn app:app --reload --port 8000
```

See `IMPLEMENTATION_GUIDE.md` for detailed architecture, training strategy, and hyperparameter defaults.

## Data and Models

- PlantVillage dataset (54,306 images, 38 classes) is ignored by `.gitignore`.
- Model checkpoints and MLflow artifacts saved to `models/` (also ignored).
- Class name mappings persisted as JSON artifacts for reproducible inference.

## Documentation

- **IMPLEMENTATION_GUIDE.md** — Eight architecture decisions with code snippets and citations.
- **.github/copilot-instructions.md** — Project context and code conventions.
- **.github/agent.md** — Guardrails for autonomous tool behavior.
- **doc/Implementation_Log.md** — Progress tracking and session notes.

## Contributing

This repository is intended for coursework. Follow conventions in `.github/copilot-instructions.md`. For questions, open an issue or contact the project owner.

## License

This project does not include an explicit license. Add one (for example, MIT) if you intend to share the code publicly.

Contact
- For questions about the code or assignments, reach out to the course instructor or project owner.