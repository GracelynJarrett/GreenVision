# Implementation Log

This log tracks progress across branches and days. Each entry includes a summary of work completed since the last log. Newest entries appear at the top.
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
