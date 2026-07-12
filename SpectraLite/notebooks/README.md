# SpectraLite notebooks

## How to open notebooks in Google Colab (all phases)

**Never double-click** an `.ipynb` in Colab’s left Files panel.  
That opens a raw text editor, not a runnable notebook. This affects **every** phase file.

### Correct workflow

1. New runtime → GPU (A100)
2. Open `Colab_Bootstrap.ipynb` via **File → Open notebook → GitHub**
3. Run Bootstrap (clone/pull + deps)
4. Open **any** phase the same way: **File → Open notebook → GitHub**  
   (or use the Colab links printed by Bootstrap’s launcher cell)
5. When Colab asks about runtime: connect to the **existing** one

### Files

| Notebook | Role |
|----------|------|
| `Colab_Bootstrap.ipynb` | Once per runtime: repo + deps + launcher links |
| `Phase0_Setup.ipynb` | Environment / model / Linear inventory smoke test |
| `Phase1_…` (later) | Baseline harness |
| … | Later research phases |

### Same-runtime reminder

Opening another notebook via GitHub does **not** wipe GPU memory installs or `/content/Execution` if you connect to the existing runtime.  
Deleting the runtime does.
