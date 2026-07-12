# SpectraLite notebooks

## Main file: `works.ipynb`

Each **PHASE N RUN** cell is self-contained (git sync → deps → model → run → `results/`).

You can run **only the phase you need**. Set `FORCE_RERUN_PHASEN = True` inside that cell to remeasure.

### Current progress (in `results/phase_status.json`)

| Phase | Status |
|-------|--------|
| 0 | Complete |
| 1 | Complete (optional FORCE rerun for WT2/PTB PPL) |
| 2 | Complete — vanilla SVD hurts PPL, no speedup |
| 3+ | Not implemented |

### After Colab

Download `SpectraLite/results/` to your PC and tell Cursor **uploaded** (Colab push often needs a PAT).
