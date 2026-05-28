# GreenVision Agent Guardrails

This file defines what Copilot Agent can and cannot do autonomously in this repository.

---

## What Agent CAN Do

Agent is allowed to work autonomously on these tasks:

- Generate boilerplate code for DataLoader setup and training loop structure.
- Write docstrings for methods, functions, and classes.
- Suggest augmentation transforms (within the conservative baseline specified in `copilot-instructions.md`).
- Scaffold FastAPI endpoints and helper functions.
- Add comments to clarify intent in new or modified code.
- Refactor code for readability or performance (within existing architecture).
- Create test stubs or validation functions.

---

## What Agent MUST NOT Do

Agent is forbidden from making these changes without explicit approval:

- **Change ImageNet normalization values** — they must always be mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
- **Modify the class names artifact structure** — the `idx_to_class` mapping and persistence mechanism must remain intact.
- **Alter the two-phase training sequence** — Phase 1 (freeze backbone, train head) and Phase 2 (unfreeze, fine-tune) must stay as designed.
- **Remove error handling** — all try/except blocks, validation steps, and explicit error checks must be preserved.
- **Change the model checkpoint or MLflow artifact structure** — where and how models/metadata are saved is fixed.
- **Modify the 38-class assumption** — the number of classes is fixed to the PlantVillage dataset.

---

## Files Agent Should Not Modify Without Confirmation

Agent must ask before editing these files:

- `.github/copilot-instructions.md` — defines project conventions and context.
- `models/` (any checkpoint files like `*.pth`) — these are trained artifacts.
- Any MLflow artifacts or metadata.
- `doc/Implementation_Log.md` — only update when explicitly asked to log today's work.
- `IMPLEMENTATION_GUIDE.md` — the architecture and design decisions are locked in unless we discuss changes.

---

## Communication Protocol

If Agent encounters ambiguity or wants to make a change outside these bounds, it MUST ask the user for permission before proceeding. Example:

> "I want to refactor the normalization pipeline for clarity, but that touches the ImageNet normalization logic. Should I proceed, or would you like to review first?"

---

## Updates

This guardrail document is living. If the project direction changes or new rules are needed, update this file and notify the user.
