# Implementation Log

This log tracks progress across branches and days. Each entry includes a summary of work completed since the last log. Newest entries appear at the top.

---
## 2026-06-02 — Gracelyn — branch: Phase3_serve_streamlit
Overview: Completed the FastAPI inference endpoint, aligned preprocessing between training and inference, and built a fully functional Streamlit dashboard with improved robustness, explainability, and user interaction.
Explanation: I implemented the /predict endpoint in serve.py, enabling real-time inference by handling image uploads, validating input, preprocessing images, running model predictions, and returning structured JSON output. A key improvement during this phase was ensuring that the inference preprocessing pipeline (resize, center crop, and ImageNet normalization) matched the training pipeline, which resolved inconsistencies between model performance during training and real-world usage.
During testing, I discovered that the model achieved high accuracy on validation and test datasets but performed poorly on real-world images. Further investigation showed that this was due to dataset bias rather than normalization issues—the training, validation, and test data shared very similar distributions, causing the model to learn dataset-specific patterns instead of generalizable features.
To improve robustness, I added a confidence threshold (40%) in the backend to flag low-confidence predictions as “Uncertain.” I also implemented detection for non-plant images using the background class, allowing the system to return a “Not a plant” response instead of forcing incorrect classifications. This effectively introduced a basic open-set handling mechanism.
On the frontend, I developed a Streamlit dashboard that connects to the FastAPI service and provides an interactive interface for uploading images and viewing results. The UI includes image preview, structured diagnosis output (plant, disease, confidence), and conditional status messaging for valid predictions, low confidence cases, and non-plant inputs. I redesigned the treatment display using styled components for better readability and added guidance messages to improve usability.
For model explainability, I extended the backend to return top-5 predictions and aggregated plant-type probabilities. These were visualized in the Streamlit app using bar charts, along with additional confidence visualizations such as a confidence meter and threshold comparison. I also implemented a prediction history feature to track user interactions within the session.
The final system integrates model training assumptions, preprocessing alignment, API serving, and a user-facing dashboard, demonstrating a complete end-to-end machine learning pipeline with improved real-world reliability and interpretability.
Files touched:

src/serve.py (implemented /predict, aligned preprocessing, added confidence threshold, background detection, top-5 predictions, and structured outputs)
src/app.py (built Streamlit dashboard, added UI enhancements, conditional messaging, graphs, and history tracking)
config/treatments.yaml (used for mapping predictions to disease names and treatment recommendations)

Next step: Prepare presentation materials by documenting model behavior differences between dataset and real-world inputs, collecting representative test cases (correct, incorrect, and non-plant), and finalizing slides for project presentation.
---



---
## 2026-06-02 — Gracelyn — branch: FastAPI_Serve

**Overview:** Completed the FastAPI inference endpoint and built a fully functional Streamlit dashboard with enhanced model explainability and user experience improvements.

**Explanation:** I implemented the `/predict` endpoint in `serve.py`, enabling real-time inference by accepting uploaded images, validating input types and sizes, preprocessing images, running model predictions, and returning structured JSON results. I added logic to detect non-leaf images using the background class and introduced confidence-based validation to flag low-confidence predictions. I also cleaned the treatment output by extracting only the relevant recommendation text from the YAML configuration.

On the frontend, I developed a Streamlit dashboard that connects to the FastAPI service and provides an interactive user interface for uploading images and viewing results. The UI includes image preview, structured diagnosis output (plant type, disease, confidence, and health status), and a redesigned treatment display using styled components for improved readability. I added user guidance messaging to improve usability and ensured that results and visualizations only appear after prediction is triggered.

To improve model transparency and interpretability, I extended the backend to return top-5 disease predictions and aggregated plant-type probabilities. These were visualized in the Streamlit app using bar charts, alongside additional confidence visualizations such as a confidence meter and threshold comparison. Layout improvements were made to separate sections clearly and enhance overall readability.

The final system integrates model inference, API serving, and a user-facing dashboard, providing a complete end-to-end machine learning application.

**Files touched:**
- `src/serve.py` (implemented `/predict`, added validation, top-5 predictions, plant aggregation, treatment formatting)
- `src/app.py` (created Streamlit dashboard, added UI improvements, graphs, and API integration)
- `config/treatments.yaml` (used for treatment recommendations, no structural changes required)

**Next step:** Prepare final testing and documentation by selecting representative test and real-world images, capturing example predictions, and finalizing project submission materials.

---


## 2026-05-31 — Gracelyn — branch: phase1-cleanup & tuning

**Overview:** Trained tuning profiles M3 and M4, added validation and test entrypoints, implemented registry tooling and verification, cleaned `src/train.py` to be Phase‑1-only, and retrained the true Phase‑1 base model. Updated logs and docs to reflect final metrics and provenance.

**Detailed explanation:**

- Training: executed tuning runs for M3 and M4 (M3 remains the best tuning candidate). Saved checkpoints to `models/m3_best.pth` and `models/m4_best.pth`, with M3 showing superior validation/test performance in the tuning experiments.

- New utilities: created `src/validate.py` (performs validation suite: confusion matrix, per-class accuracy, calibration plots), `src/test.py` (reproducible test-set evaluation runner), `src/register.py` (register + promote a checkpoint to MLflow Model Registry), and `src/verify.py` (load Production model and run a smoke inference). These scripts centralize validation, testing, and registry operations for reproducibility.

- Cleanup & retrain: refactored `src/train.py` and `config.yaml` to remove Phase‑2 behavior from the Phase‑1 entrypoint, retrained the Phase‑1 base model to produce MLflow run `feb3c08d14464d999a528f91bcca74ab` and checkpoint `models/phase1_best.pth`. Logged metrics: phase1 train acc 86.57 (loss 0.4270), best val acc 92.46 (loss 0.2218), test acc 92.38 (loss 0.2248). Inserted these results into `doc/Model_metrix_Log.md` and added the Phase‑1 run ID to `doc/Training_Report.md`.

- Documentation & provenance: updated `doc/Model_metrix_Log.md` (warning about earlier accidental Phase‑2 inclusion, True Base Model section), `doc/Training_Report.md` (added Phase‑1 run id), and `doc/Implementation_Log.md` (this consolidated entry).

**Files touched:**

- `src/train.py` (cleaned Phase‑1 entrypoint)
- `config.yaml` (removed Phase‑2 keys for Phase‑1 entrypoint)
- `src/validate.py`, `src/test.py`, `src/register.py`, `src/verify.py` (new validation/test/registry scripts)
- `doc/Model_metrix_Log.md`, `doc/Training_Report.md`, `doc/Implementation_Log.md` (updated)
- `models/phase1_best.pth`, `models/m3_best.pth`, `models/m4_best.pth` (checkpoints)
- `logs/train.log`, `logs/validation_report.json`, `logs/test_report.json`, `logs/tune_*.log` (artifacts)

**Key run IDs:**

- Phase‑1 retrain (true base): `feb3c08d14464d999a528f91bcca74ab`
- M3 tuning (selected): `0efe2199c322443ca063487e01c3eb9d`

**Next step:** Create a FastAPI inference service and server to serve the selected model (M3) and the true Phase‑1 baseline, and prepare deployment artifacts (requirements, Dockerfile, simple API routes for predict/health/status).

---

## 2026-05-30 — Gracelyn — branch: phase2-tuning

**Overview:** Added a profile-driven tuning runner, tuning presets in `config.yaml`, and selective backbone-group unfreezing; executed and recorded the first tuning run (M2).

**Explanation:** Implemented a dedicated tuning entrypoint `src/tune.py` that reads profile definitions from `config.yaml`, applies augmentation presets, and unfreezes specific backbone block groups (M2–M5). Added a helper to selectively unfreeze backbone groups in `src/model.py` and config-driven transform/dataloader helpers in `src/dataset.py`. Fixed a duplicate `tuning:` block in the config that prevented profile lookup, then ran profile `M2` to completion on GPU (checkpoint saved to `models/m2_best.pth` and metrics appended to `doc/Model_metrix_Log.md`). MLflow tracking uses the `phase2_tuning_<profile>` run name prefix and tuning enforces GPU-only runs.

**Files touched:**
- `src/tune.py` (new)
- `src/model.py` (added grouped unfreeze helper)
- `src/dataset.py` (config-driven transforms + dataset helper)
- `config.yaml` (tuning profiles and MLflow tuning prefix)
- `doc/Model_metrix_Log.md` (M2 metrics appended by run)
- `logs/tune_m2.log` (runtime log)
- `models/m2_best.pth` (generated checkpoint)

**Next step:** Run remaining tuning profiles (`M3`, `M4`, `M5`), compare validation/test metrics in MLflow, select the best checkpoint, then create `phase3-validation` branch from that commit for formal validation.

---

## 2026-05-29 — Gracelyn — branch: Phase1_train_test

---

## 2026-05-28 — Gracelyn — branch: Phase1_train_test

**Overview:** Refreshed the repository README to better document the GreenVision project and its workflow.

**Explanation:** I updated the top-level README so the project description, feature list, repository structure, setup steps, and usage examples are easier to follow. The revised documentation now reflects the two-phase EfficientNet-B0 training workflow, the MLflow tracking setup, and the FastAPI inference service more clearly. I also clarified how the dataset and model artifacts are organized so the repository is easier to navigate for future implementation work. This keeps the public-facing project documentation aligned with the implementation guide and the current codebase direction.

**Files touched:**
- `README.md` (updated)

**Next step:** Continue Phase 1 implementation in `src/train.py` and the related training utilities, then validate the training pipeline against the documented workflow.

## 2026-05-27 — Gracelyn — branch: main

**Overview:** Completed the implementation guide and set up Copilot guardrails and documentation files.

**Explanation:** Today I finished the IMPLEMENTATION_GUIDE.md with all eight required architecture and design decisions, including code snippets, reasoning, and citations. I also created the .github/copilot-instructions.md file with project-specific context (dataset details, critical constants, code conventions, and architecture notes) and guardrails for working in small increments, commenting code, and asking before creating new files. I set up this implementation log to track future progress by branch and day, with each entry following a consistent format: overview, explanation, files touched, and next steps. This documentation establishes a clear baseline for the GreenVision project before moving to code implementation.

**Files touched:**
- `IMPLEMENTATION_GUIDE.md` (formatting and citations)
- `.github/copilot-instructions.md` (created)
- `.github/agent.md` (created)
- `doc/Implementation_Log.md` (created)

**Next step:** Begin implementation — start with `train.py` for Phase-1 head-only training, following the two-phase strategy and all guardrails defined in the documentation.

---
