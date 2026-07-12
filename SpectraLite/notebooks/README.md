# SpectraLite notebooks

## Main lab file

Use **[`works.ipynb`](works.ipynb)** for the whole research project.

### Persistence (important)

After each phase, artifacts are written under `SpectraLite/results/` and should be **committed to git**:

| File | Purpose |
|------|---------|
| `results/phase_status.json` | Which phases are complete |
| `results/phase0_summary.json` | Phase 0 env/model summary |
| `results/phase0_linear_layers.json` | Full `nn.Linear` inventory |
| `results/phase1_dense_baselines.csv` | Dense FLOP/latency/PPL row(s) |
| `results/phase1_summary.json` | Phase 1 cached metrics |

**New Colab session:** `git pull` → Stage 0 → **Session Restore** → only run incomplete phases.  
Model weights still reload from Hugging Face (not stored in git). Metrics are **not** recomputed if marked complete.

Open via: `File → Open notebook → GitHub → works.ipynb` (never Files double-click).

## Legacy

`Colab_Bootstrap.ipynb` / `Phase0_Setup.ipynb` — superseded by `works.ipynb`.
