# Implementation Log

This log tracks progress across branches and days. Each entry includes a summary of work completed since the last log. Newest entries appear at the top.
---

## 2026-05-31 — Gracelyn — branch: phase2-tuning

**Overview:** Completed all remaining tuning profiles (M3, M4, M5), selected M3 as the best model, and documented final model selection rationale in Model_metrix_Log.md.

**Explanation:** Executed tuning profiles M3 (groups 4-6, best result at 99.41% test accuracy), M4 (groups 7-8, 95.73% test), and M5 (full backbone, 96.08% test) sequentially on GPU, with one interruption during M2 due to laptop power loss (successfully restarted). All runs completed with metrics automatically appended to `doc/Model_metrix_Log.md` by the tuning runner. Created a Model Comparison Summary table showing all models side-by-side and added a new "Final Model Selection: M3" section documenting why M3 was chosen: best tuning performance (99.41% test), excellent validation stability (99.40% best val, 0.0165 loss), and balanced backbone unfreezing strategy. Updated the metrics log to clarify that Base Phase 2 (99.50% test) is kept separate from tuning comparison per user decision, since Phase 2 included unplanned hyperparameter tuning not part of the original baseline.

**Files touched:**
- `logs/tune_m3.log`, `logs/tune_m4.log`, `logs/tune_m5.log` (runtime logs)
- `models/m3_best.pth`, `models/m4_best.pth`, `models/m5_best.pth` (generated checkpoints)
- `doc/Model_metrix_Log.md` (M3/M4/M5 metrics appended; added comparison table and Final Model Selection section)

**Next step:** Create `phase3-validation` branch from current commit and set up formal validation workflow for M3 (confusion matrix, per-class accuracy, edge case inspection, prediction visualization).

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
