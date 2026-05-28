# GreenVision

GreenVision is a small image-classification project created for the Applied Machine Learning course (Spring 2026). It contains experiments and notebooks for training and evaluating convolutional neural networks on example datasets (see `mnist_cnn.ipynb`).

Features
- Example CNN training and evaluation in `mnist_cnn.ipynb`
- Lightweight project structure suitable for coursework and experiments

Repository structure
- `mnist_cnn.ipynb` — main notebook with model definition, training, and evaluation
- `data/` — (ignored) dataset files and downloads
- `models/` — (ignored) saved model checkpoints and artifacts
- `scr/` — source code (if present)
- `models/` — saved model artifacts (ignored by .gitignore)
- `IMPLEMENTATION_GUIDE.md` — project notes and implementation plan

Setup
1. Create and activate a Python virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

2. Install dependencies (if a `requirements.txt` exists):

```powershell
pip install -r requirements.txt
```

If no `requirements.txt` is provided, a typical environment includes `numpy`, `torch` or `tensorflow` (depending on the notebook), `scikit-learn`, and `jupyter`.

Usage
- To run the experiments, open `mnist_cnn.ipynb` in Jupyter Notebook or JupyterLab and follow the cells.
- Training and evaluation code is included in the notebook; adapt hyperparameters and dataset paths as needed.

Data and models
- Large datasets and model files are ignored by `.gitignore`. Keep dataset downloads outside version control or add small sample data tracked in the repo.

Contributing
- This repository is intended for coursework. For contributions or questions, please open an issue or contact the project owner.

License
- This project does not include an explicit license. Add one (for example, MIT) if you intend to share the code publicly.

Contact
- For questions about the code or assignments, reach out to the course instructor or project owner.