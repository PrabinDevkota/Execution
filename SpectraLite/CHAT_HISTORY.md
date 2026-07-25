# SpectraLite — Chat / Progress Record

Last updated: 2026-07-25 (IST)

**GitHub (public):** https://github.com/PrabinDevkota/Execution  
**Project path in repo:** `SpectraLite/`  
**Clone / Colab:** open notebooks from that GitHub tree (not local Files double-click).

Dev model: `facebook/opt-125m` · Hardware: Colab NVIDIA A100  
Primary notebook: `notebooks/works.ipynb`  
Paper draft (local, gitignored): `latex_code.tex` — IEEE journal class; Phases 0–8 results.

This file records decisions and outcomes from Cursor chat sessions so work can resume without re-deriving context.

---

## 1. Project goal (unchanged)

**SpectraLite**: training-free, post-training SVD compression for **small decoder-only** LLMs.

Core ideas:
1. Activation whitening before truncated SVD  
2. Rank allocation from Roy–Vetterli effective rank \(\rho_{\mathrm{eff}}\) under a global FLOP budget  
3. **Latency feasibility gate**: compress only if \(r < \kappa_{\mathrm{speed}} \cdot mn/(m+n)\)  
4. Optional stability modules (Ledoit–Wolf, \(\kappa\)) — studied; ridge preferred on OPT-125M  

Flagship later: LLaMA-3.2-1B (after OPT-1.3B).

---

## 2. Phase status

| Phase | What | Status | Headline |
|------|------|--------|----------|
| 0–8 | OPT-125M full ladder | **Done** | Gate dual/triple win; Spec-ρ competitive; zero-shot gated 38.3% |
| **9** | Spec-ρ **+** latency gate (OPT-125M) | **Done** | Spec-ρ gated C4≈141, decode≈9.77ms, zs≈37.9%; ActSVD gated still stronger (C4≈111, 8.44ms) |
| **10** | OPT-1.3B scale ladder | **Code ready — run in Colab (memory-lean)** | Same-family step toward absolute speedup |
| 11+ | LLaMA-3.2-1B + runtime co-design | Planned | Packed MLP / CUDA-graph / FlashSVD handoff |

### Phase 8 zero-shot averages (keep ≈0.75) — recorded

| Method | Avg | Retention vs dense |
|--------|-----|--------------------|
| Dense | 41.8% | 100% |
| ActSVD gated | **38.3%** | **91.7%** |
| SpectraLite-ρ | 37.5% | 89.7% |
| ActSVD ungated | 36.7% | 87.8% |

---

## 3. Main highlight (agreed framing)

**Headline novelty:** the **latency feasibility gate** — improves decode latency, C4 perplexity, *and* zero-shot accuracy vs ungated ActSVD by refusing break-even attention factorization and compressing MLPs instead.

Supporting pillars:
- Whitening: vanilla C4 922 → ActSVD 123 (7.5×)  
- SpectraLite-ρ: competitive spectral alternative; slightly better WT2; beats ungated on zero-shot avg  
- Protect design: ρ-only works; ρ×stable-rank fails  
- Honest limit: no absolute >1× decode vs dense on OPT-125M batch=1 yet  

---

## 4. Next-stage plan (active)

1. **Phase 9 (OPT-125M):** run `works.ipynb` Phase 9 cell → Spec-ρ ± gate vs ActSVD ± gate + Spec-ρ+gate zero-shot → commit `results/phase9_*`.  
2. **Phase 10 (OPT-1.3B):** run Phase 10 cell (loads `facebook/opt-1.3b`) → dense / ActSVD / Spec-ρ ± gate + zero-shot → commit `results/phase10_*`.  
3. **Then:** LLaMA-3.2-1B (HF token), update `latex_code.tex` tables in place.  
4. **If still ≤1× vs dense:** stricter \(\kappa_{\mathrm{speed}}\), packed MLP, CUDA-graph decode.

Code added 2026-07-25:
- `svd_spectralite.py`: `latency_gate` / `kappa_speed` on allocate path  
- `phase9.py`, `phase10.py`  
- `config.config_for_model()` presets  
- Notebook cells Phase 9 & 10 in `works.ipynb`

---

## 5. Paper decisions

IEEE journal (`IEEEtran`). Related Work = user’s 22-paper survey (verbatim).  
`latex_code.tex` is **gitignored**. Future Work already lists Spec-ρ+gate and larger models.

---

## 6. Workflow notes

- Colab: open notebooks via GitHub  
- Phase cells: fetch/reset → deps → load → run → `results/`  
- After Phase 9/10: commit results from Cursor or Colab  
- Default protect: **`rho`**  
- Dense peak MFU notes: A100 FP16 Tensor Core 312 TFLOPS  

---

## 7. Key code modules

`spectralite/config.py`, `calibration.py`, `whitening.py`, `svd_vanilla.py`, `svd_activation.py`, `svd_spectralite.py`, `spectral.py`, `rank_alloc.py`, `latency_gate.py`, `stability.py`, `downstream.py`, `phase2.py`–`phase10.py`, `phase_runner.py`, `lowrank.py`

---

*Update this file when a new phase completes or the paper framing changes.*
