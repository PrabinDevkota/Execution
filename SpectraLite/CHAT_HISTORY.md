# SpectraLite — Chat / Progress Record

Last updated: 2026-07-13 (IST)  
Repo: https://github.com/PrabinDevkota/Execution (path `SpectraLite/`)  
Dev model: `facebook/opt-125m` · Hardware: Colab NVIDIA A100  
Primary notebook: `notebooks/works.ipynb`  
Paper draft (local, gitignored): `latex_code.tex`

This file records decisions and outcomes from Cursor chat sessions so work can resume without re-deriving context.

---

## 1. Project goal (unchanged)

**SpectraLite**: training-free, post-training SVD compression for **small decoder-only** LLMs.

Core ideas:
1. Activation whitening before truncated SVD  
2. Rank allocation from Roy–Vetterli effective rank \(\rho_{\mathrm{eff}}\) under a global FLOP budget  
3. **Latency feasibility gate**: compress only if \(r < \kappa_{\mathrm{speed}} \cdot mn/(m+n)\)  
4. Optional stability modules (Ledoit–Wolf, \(\kappa\)) — studied; ridge preferred on OPT-125M  

Flagship later: LLaMA-3.2-1B (not run yet).

---

## 2. Phase status (0–8 complete)

| Phase | What | Status | Headline |
|------|------|--------|----------|
| 0 | Smoke / load OPT-125M | Done | 125.2M params, 73 Linears, A100 OK |
| 1 | Dense baselines | Done | C4 ≈28.7, WT2 ≈44, prefill ≈7.3 ms, decode ≈7.2 ms/tok |
| 2 | Vanilla SVD | Done | C4 922 / 2953 / 6256 at r=0.5/0.4/0.3 — motivating negative |
| 3 | ActSVD (ridge whitening) | Done | C4 123 / 555 / 2287 — whitening is necessary |
| 4 | SpectraLite-ρ | Done | C4 141 / 757 / 2501; WT2 82.8 beats ActSVD 87.4 at keep 0.75 |
| 5 | LW + κ stability | Done | LW hurt ActSVD (C4 1472); κ mostly idle |
| 6 | Latency gate | Done | **Dual win**: decode 10.1→8.3 ms, C4 123→111 (48 attn dense, 24 MLP compressed) |
| 7 | Ablations | Done | Whitening, gate, MLP, ρ-only good; full/sr protect collapse; calib 16≈32 |
| 8 | Zero-shot lm-eval | Done | Gated best compressed (38.3%, 91.7% of dense); Spec-ρ 37.5% > ungated 36.7% |

Phase 8 commit on `main`: `bf1e94d` — `results/phase8_*.json` + `phase_status.json`.

### Phase 8 zero-shot averages (keep ≈0.75)

| Method | Avg | Retention vs dense |
|--------|-----|--------------------|
| Dense | 41.8% | 100% |
| ActSVD gated | **38.3%** | **91.7%** |
| SpectraLite-ρ | 37.5% | 89.7% |
| ActSVD ungated | 36.7% | 87.8% |

BoolQ: ungated 41.5% → gated 50.2% / Spec-ρ 49.8%.

---

## 3. Main highlight (agreed framing)

**Headline novelty:** the **latency feasibility gate** — improves decode latency, C4 perplexity, *and* zero-shot accuracy vs ungated ActSVD by refusing break-even attention factorization and compressing MLPs instead.

Supporting pillars:
- Whitening: vanilla C4 922 → ActSVD 123 (7.5×)  
- SpectraLite-ρ: competitive spectral alternative; slightly better WT2; beats ungated on zero-shot avg  
- Protect design: ρ-only works; ρ×stable-rank fails  
- Honest limit: no absolute >1× decode vs dense on OPT-125M batch=1 yet  

---

## 4. Paper decisions (this chat)

IEEE conference format (`IEEEtran`).

Section order:
1. Abstract  
2. Introduction (novelty + Phase 0–8 scope)  
3. **Related Work** = user’s 22-paper literature survey (**kept verbatim**; only section title renamed from “Literature Survey”)  
4. Method  
5. Experimental Setup (incl. phased Colab protocol)  
6. Results (Phases 2–8 tables)  
7. Discussion  
8. Conclusion  
9. Future Work  
10. Bibliography `ref1`–`ref22` (user’s keys, unchanged)

User instruction: **do not rewrite/remove the literature survey**; put SpectraLite content around it.

`latex_code.tex` is **gitignored** (local draft until ready to publish).

---

## 5. Workflow notes (operational)

- Colab: open notebooks via GitHub, not Files double-click  
- Each phase cell: fetch/reset → deps → load model → run → write `results/`  
- Model weights not in git; metrics JSON/CSV are  
- Prefer Cursor commit/push for `results/` (Colab push often fails)  
- WikiText HF id: `Salesforce/wikitext`  
- ActSVD SVD on CPU when weight/cov devices differ  
- Default protect after Phase 7: **`rho`** (not `full`)  
- Dense peak for MFU notes: A100 FP16 Tensor Core 312 TFLOPS  

---

## 6. Future stages (explicitly left open)

1. SpectraLite-ρ **+ latency gate** as default deployed config  
2. LLaMA-3.2-1B (absolute speedup headroom)  
3. Runtime co-design: packed MLP, CUDA-graph decode, FlashSVD handoff  
4. Broader eval / few-shot / longer prefill / optional SVD+quant  
5. Revisit LW/κ at larger scale  

---

## 7. Key code modules

`spectralite/config.py`, `calibration.py`, `whitening.py`, `svd_vanilla.py`, `svd_activation.py`, `svd_spectralite.py`, `spectral.py`, `rank_alloc.py`, `latency_gate.py`, `stability.py`, `downstream.py`, `phase2.py`–`phase8.py`, `phase_runner.py`, `lowrank.py`

---

## 8. Chat sessions covered by this record

- SpectraLite implementation Phases 0–8 (prior + this thread)  
- Paper drafting in IEEE format with preserved literature survey  
- Phase 8 analysis integrated into paper; results pushed (`bf1e94d`)  
- Agreed main highlight: latency gate dual/triple win  

Agent transcript (Cursor): `agent-transcripts/b9b00821-1c73-4194-9ab5-112ace42a6aa`

---

*Update this file when a new phase completes or the paper framing changes.*
