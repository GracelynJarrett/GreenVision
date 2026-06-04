# 🌿 GreenVision

GreenVision is a plant leaf disease classification system developed for the Applied Machine Learning course (Spring 2026). It combines deep learning, a FastAPI backend, and a Streamlit dashboard to provide real-time disease predictions and treatment recommendations.

---

## 🔍 Overview

GreenVision predicts plant diseases from leaf images using transfer learning with EfficientNet-B0. The system supports real-time inference through an API and an interactive user interface, making it a complete end-to-end machine learning application.

---

## 🚀 Features

- Transfer learning with EfficientNet-B0 (ImageNet pretrained)
- Two-phase training strategy:
  - Feature extraction (frozen backbone)
  - Fine-tuning (unfrozen layers)
- Data augmentation and ImageNet normalization
- MLflow experiment tracking for reproducibility
- FastAPI inference service with validation and confidence scoring
- Interactive Streamlit dashboard for real-time predictions
- Top-5 disease predictions and plant-level probability analysis
- Background image detection (non-leaf filtering)
- Treatment recommendations from YAML configuration

---

## 🧠 Architecture

### Inference Pipeline (Deployed System)

User → Streamlit Dashboard → FastAPI API →  
Validate → Preprocess → Predict → Post-process → Response → Display  

### Training Pipeline

Data → dataset.py → model.py → train.py → tune.py → validate.py → test.py → register.py  

---

## 📁 Repository Structure

- `IMPLEMENTATION_GUIDE.md` — architecture decisions and design rationale
- `.github/copilot-instructions.md` — AI assistant context
- `.github/agent.md` — guardrails
- `doc/` — documentation
  - `Implementation_Log.md` — development history
  - `Research_Log.md` — research notes
- `data/` — dataset (ignored)
- `models/` — model checkpoints (ignored)
- `src/` — source code
  - `app.py` — Streamlit dashboard
  - `serve.py` — FastAPI backend
  - `dataset.py` — data loading & preprocessing
  - `model.py` — model architecture
  - `utils.py` — helper functions
- `config/`
  - `config.yaml` — training configuration
  - `treatments.yaml` — treatment recommendations

---

## ⚙️ Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate

---
### 2. Install dependencies
pip install -r requirements.txt
---

---
### 3.Running the Application
  Start FastAPI: uvicorn src.serve:app --reload --port 8001
  Start Streamlit: streamlit run src/app.py
---

---
## Usage
  Upload a plant leaf image in the Streamlit dashboard
  Click Predict
  View:
    Plant type classification
    Disease prediction
    Confidence score
    Treatment recommendation
    Top-5 prediction graphs
  ---

  ---
## Data and Models
    PlantVillage dataset (~54,000 images, 38 classes)
    Dataset and models are excluded from version control via .gitignore
    Class mappings stored as JSON for reproducible inference
    ---

---
## Documentation
  IMPLEMENTATION_GUIDE.md — architecture decisions and code examples
  doc/Implementation_Log.md — development progress
  doc/Research_Log.md — supporting research
  ---

---
## Fututre improvements
  Improve generalization with more diverse real-world images
  Expand support to additional plant species
  Deploy application to a cloud platform
  Enhance UI/UX for production use
  --

# Author
  Gracelyn Jarrett