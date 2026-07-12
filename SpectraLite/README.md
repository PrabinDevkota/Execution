# SpectraLite

**Training-free, post-training SVD compression for decoder-only Transformer LLMs.**

SpectraLite allocates per-matrix rank from the spectral structure of activation-whitened
weights (effective rank / normalized spectral entropy) under a global FLOP budget, fused
with task-sensitivity weighting and numerical-stability safeguards, and constrains
allocation with a hardware latency-feasibility gate so theoretical FLOP reductions convert
to measured wall-clock decode speedup.

> **Current status:** Phase 0 — project structure and environment verification only.
> Compression, SVD, whitening, rank allocation, and latency engineering are intentionally
> **not** implemented yet.

---

## Repository layout

```
SpectraLite/
├── notebooks/
│   └── Phase0_Setup.ipynb      # Colab / local environment smoke test
├── spectralite/                # Core Python package (Phase 0 utilities)
│   ├── config.py               # Experiment configuration
│   ├── system.py               # Python / CUDA / Torch introspection
│   ├── gpu.py                  # GPU memory helpers
│   ├── model_loader.py         # Tokenizer + CausalLM loading
│   ├── model_analysis.py       # Architecture / Linear-layer inventory
│   └── utils.py                # Seeds, logging, formatting
├── results/                    # Experiment tables / CSVs
├── checkpoints/                # Compressed / intermediate weights (later)
├── figures/                    # Plots for papers / reports
├── logs/                       # Run logs
├── requirements.txt
└── README.md
```

---

## Phase 0 objectives

1. Create a clean, modular research codebase.
2. Verify Python, PyTorch, CUDA, and GPU visibility.
3. Load `facebook/opt-125m` in FP16 (when CUDA is available).
4. Inventory every `nn.Linear` layer (names, shapes) for later SVD targeting.
5. Run one short generation smoke test.
6. Report GPU memory before load / after load / after inference.

---

## Quick start (local or Colab)

```bash
cd SpectraLite
pip install -r requirements.txt          # local (includes torch)
# pip install -r requirements-colab.txt  # Colab only — never reinstalls torch
```

### Colab workflow (recommended)

1. Runtime → **GPU (A100)**  
2. Open and run **`notebooks/Colab_Bootstrap.ipynb` once** per new runtime  
   (clone/pull + Colab-safe deps + CUDA check)  
3. Then open the phase you need (`Phase0_Setup.ipynb`, later `Phase1_…`, etc.)

Or one cell at the top of any phase notebook:

```python
from spectralite.colab_setup import ensure_environment
ensure_environment()
```

Dependency files:

| File | Use when |
|------|----------|
| `requirements-colab.txt` | Google Colab (no torch — keeps CUDA wheel) |
| `requirements.txt` | Local machine / non-Colab |

---

## Primary development model (Phase 0)

| Setting        | Value                 |
|----------------|-----------------------|
| Model          | `facebook/opt-125m`   |
| Dtype          | `float16` on CUDA     |
| Device map     | `auto`                |
| Smoke prompt   | SVD one-sentence Q    |
| Max new tokens | 50                    |

Larger models (OPT-1.3B, Pythia, LLaMA-3.2-1B) belong to later phases.

---

## Package overview

| Module              | Responsibility                                      |
|---------------------|-----------------------------------------------------|
| `config`            | Central hyperparameters and paths                   |
| `system`            | Environment report (Python / Torch / CUDA / GPU)    |
| `gpu`               | Allocated / reserved / free VRAM helpers            |
| `model_loader`      | Load tokenizer + causal LM; count parameters        |
| `model_analysis`    | Blocks, attention/MLP counts, full Linear inventory |
| `utils`             | Seeding, logging, pretty printing                   |

---

## Roadmap (not implemented in Phase 0)

| Phase | Focus                                              |
|-------|----------------------------------------------------|
| 0     | Environment + project skeleton (**this release**)  |
| 1     | Baseline FLOP / latency / PPL harness              |
| 2     | Vanilla truncated SVD                              |
| 3     | Activation-aware baselines (ASVD / SVD-LLM)        |
| 4     | Spectral-entropy rank allocation (core novelty)    |
| 5     | Ledoit-Wolf + condition-number safeguards          |
| 6     | Latency gate + factor fusion + CUDA-graph decode   |
| 7     | Ablations                                          |
| 8     | Full lm-eval / paper tables                        |

---

## Citation / license

Research code under active development. Model weights are subject to their upstream
licenses (e.g. OPT on Hugging Face).
