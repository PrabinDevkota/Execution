# SpectraLite — Chat / Progress Record

Last updated: 2026-07-25 (IST)

**GitHub (public):** https://github.com/PrabinDevkota/Execution  
**Project path in repo:** `SpectraLite/`  
Primary notebook: `notebooks/works.ipynb`  
Paper draft (local, **gitignored**): `IEEE_Access_LaTeX_template/access.tex`  
(old `latex_code.tex` is only a pointer; edit Access folder from now on).  
Paper figures: `IEEE_Access_LaTeX_template/figures/` (regenerate via `fig_*/generate.py` or `generate_figure.ipynb`).

Dev models: `facebook/opt-125m` · `facebook/opt-1.3b` · Hardware: Colab A100  
Next flagship: LLaMA-3.2-1B.

---

## Phase status

| Phase | What | Status | Headline |
|------|------|--------|----------|
| 0–8 | OPT-125M full ladder | **Done** | Gate dual/triple win; zs gated 38.3% (91.8% dense) |
| 9 | Spec-ρ + gate (125M) | **Done** | ActSVD+gate still best compressed |
| 10 | OPT-1.3B PPL/latency | **Done** | Gate: decode −20%, thr +24%, C4 88→81 |
| **10b** | OPT-1.3B zero-shot | **Done** | ActSVD+gate **44.5% (88.1% dense)**; BoolQ 61.1% > dense |
| 11+ | LLaMA-3.2-1B + runtime | Planned | Absolute speedup / modern arch |

### OPT-1.3B Phase 10b zero-shot

| Method | Avg | Retention |
|--------|-----|-----------|
| Dense | 50.4% | 100% |
| ActSVD gated | **44.5%** | **88.1%** |
| Spec-ρ+gate | 40.3% | 80.0% |

### Deploy default on OPT ladder
**ActSVD + latency gate** (best C4 + latency + zero-shot on both sizes).

---

## Next steps
1. LLaMA-3.2-1B (memory-lean Phase 10/10b recipe; HF token)  
2. Runtime co-design if still ≤1× vs dense  
3. Optional: official ASVD/SVD-LLM baselines  

Commits as PrabinDevkota (not Cursor Agent).

---

*Update this file when a new phase completes.*
