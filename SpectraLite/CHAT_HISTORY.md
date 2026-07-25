# SpectraLite — Chat / Progress Record

Last updated: 2026-07-25 (IST)

**GitHub (public):** https://github.com/PrabinDevkota/Execution  
**Project path in repo:** `SpectraLite/`  
**Clone / Colab:** open notebooks from that GitHub tree (not local Files double-click).

Dev models: `facebook/opt-125m` (Phases 0–9) · `facebook/opt-1.3b` (Phase 10)  
Hardware: Colab NVIDIA A100  
Primary notebook: `notebooks/works.ipynb`  
Paper draft (local, gitignored): `latex_code.tex` — IEEE journal; Phases 0–10 results.

---

## 1. Project goal (unchanged)

**SpectraLite**: training-free, post-training SVD compression for **small decoder-only** LLMs.

Core ideas:
1. Activation whitening before truncated SVD  
2. Rank allocation from Roy–Vetterli effective rank \(\rho_{\mathrm{eff}}\) under a global FLOP budget  
3. **Latency feasibility gate**: compress only if \(r < \kappa_{\mathrm{speed}} \cdot mn/(m+n)\)  
4. Optional stability modules (Ledoit–Wolf, \(\kappa\)) — studied; ridge preferred on OPT-125M  

Next flagship: LLaMA-3.2-1B.

---

## 2. Phase status

| Phase | What | Status | Headline |
|------|------|--------|----------|
| 0–8 | OPT-125M full ladder | **Done** | Gate dual/triple win; Spec-ρ competitive; zero-shot gated 38.3% |
| **9** | Spec-ρ + gate (OPT-125M) | **Done** | Spec-ρ+gate zs≈37.9%; ActSVD+gate still best (C4≈111) |
| **10** | OPT-1.3B scale ladder | **Done** (PPL/latency; no zs) | Gate: decode −20%, thr +24%, C4 88→81; 0.86× dense |
| 11+ | LLaMA-3.2-1B + runtime / zs | Planned | Absolute speedup + modern arch |

### Phase 8 zero-shot (OPT-125M, keep ≈0.75)

| Method | Avg | Retention |
|--------|-----|-----------|
| Dense | 41.8% | 100% |
| ActSVD gated | **38.3%** | **91.7%** |
| SpectraLite-ρ | 37.5% | 89.7% |
| ActSVD ungated | 36.7% | 87.8% |

### Phase 10 headline (OPT-1.3B, keep ≈0.75, memory-lean)

| Method | C4 ↓ | Decode ms/tok | vs dense decode | #comp / gated-dense |
|--------|------|---------------|-----------------|---------------------|
| Dense | **18.0** | **14.0** | 1.00× | — |
| ActSVD | 88.1 | 20.3 | 0.69× | 144 / 0 |
| **ActSVD + gate** | **80.5** | **16.3** | **0.86×** | **48 / 96** |
| Spec-ρ | 96.7 | 20.0 | 0.70× | 144 / 0 |
| Spec-ρ + gate | 89.9 | 19.1 | 0.73× | 113 / 31 |

Lean protocol: calib 16×256, max_tokens/layer=4096, float32 SVD cache. Zero-shot deferred.

---

## 3. Main highlight (agreed framing)

**Headline novelty:** the **latency feasibility gate** — on both OPT-125M and OPT-1.3B, gated ActSVD improves decode *and* C4 vs ungated by keeping break-even attention dense and compressing MLPs.

Supporting:
- Whitening necessary (125M: vanilla C4 922 → 123)  
- Spec-ρ competitive on 125M; **behind ActSVD on 1.3B** at this keep  
- Spec-ρ+gate only weakly gated on 1.3B (31 layers) because ρ already lowers many attn ranks below break-even  
- Honest: still **no absolute >1× decode vs dense** at batch=1  

---

## 4. Next-stage plan

1. OPT-1.3B zero-shot (ActSVD+gate ± Spec-ρ+gate) — optional second pass  
2. LLaMA-3.2-1B (HF token) with memory-lean Phase-10 recipe  
3. Runtime: packed MLP / CUDA-graph decode if still ≤1× vs dense  
4. Keep paper tables updated in `latex_code.tex` (local)

---

## 5. Paper decisions

IEEE journal (`IEEEtran`). Related Work = 22-paper survey (verbatim).  
`latex_code.tex` is **gitignored**.

---

## 6. Workflow notes

- Phase 10 RAM: never use 50k tokens/layer on 1.3B (OOM); use ≤4096  
- Default protect: **`rho`**  
- Commits as PrabinDevkota (not Cursor Agent)  

---

## 7. Key code modules

`spectralite/config.py`, `calibration.py`, `whitening.py`, `svd_vanilla.py`, `svd_activation.py`, `svd_spectralite.py`, `spectral.py`, `rank_alloc.py`, `latency_gate.py`, `stability.py`, `downstream.py`, `phase2.py`–`phase10.py`, `phase_runner.py`, `lowrank.py`

---

*Update this file when a new phase completes or the paper framing changes.*
