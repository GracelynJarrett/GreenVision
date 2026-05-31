Tuning Guide for GreenVision

Purpose
- Describe how to run profile-driven tuning (M2–M5) using `src/tune.py`.

Profiles
- M2: Unfreeze backbone groups 1–3 (conservative, light augmentation)
- M3: Unfreeze backbone groups 4–6 (balanced)
- M4: Unfreeze backbone groups 7–8 (stronger augmentation)
- M5: Unfreeze full backbone (equivalent to base Phase 2 unfreeze)

Important notes
- Runs require a CUDA-enabled GPU. `src/tune.py` will raise an error if CUDA is unavailable.
- Tuning profiles are defined in `config.yaml` under the `tuning.profiles` section. Update `config.yaml` for different hyperparameters or augmentation presets.
- MLflow run names use the prefix `phase2_tuning_` and then the profile name, e.g. `phase2_tuning_M2`.
- Checkpoints are saved to the `models/` directory as `<profile>_best.pth` (e.g. `m2_best.pth`).
- Runtime logs are written to `logs/tune_<profile>.log`.
- After a successful run, a summary entry is appended to `doc/Model_metrix_Log.md` by the runner.

Quick start

1. Activate your project virtual environment (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

2. (Optional) Start MLflow UI to monitor runs:

```powershell
mlflow ui --port 5000
```

3. Run a profile (example: M3):

```powershell
python -m src.tune --config config.yaml --profile M3
```

4. After completion, verify:
- `models/m3_best.pth` exists
- `logs/tune_m3.log` contains final epoch and test metrics
- `doc/Model_metrix_Log.md` has the `## Tuning Model M3 Metrics` section

Troubleshooting
- If the run fails immediately with a GPU error, verify CUDA is available and PyTorch detects it:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

- If profiles are not found, ensure `config.yaml` contains the `tuning.profiles` entries (M2–M5) and there is no duplicate top-level `tuning:` block.

Contact
- If you need changes to the profiles or want automated orchestration of all profiles, open an issue or request the change in the `phase2-tuning` branch.