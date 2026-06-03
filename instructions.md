# Build & Run Instructions

This document tells a new user exactly how to go from a bare machine to reproducing every result in the paper.

---

## 0. What You Will Get

After following these steps you will have:

| Step | Output | Time |
|---|---|---|
| Environment + deps | Working Python venv with all packages | ~10 min |
| Quick verification | EA-NPS module imports OK | ~30 sec |
| Generate figures (fast) | 6 PNG figures in `vip_res/figures/` | ~2 min |
| Proxy validation | `proxy_validation.png` + `proxy_validation.csv` | ~5 min |
| PermutedMNIST experiments | 6 CSV files with all results | ~25 min (GPU) |
| CORe50 experiments | `core50_results.csv` | ~45 min (GPU) |

---

## 1. Prerequisites

**Hardware:**
- Any machine with Python 3.10–3.13 (CPU-only OK for figures + proxy)
- NVIDIA GPU with CUDA 12+ recommended for experiments (T4, V100, A10G, etc.)
  - Experiments will run on CPU but will be ~10× slower
- 4 GB free disk space

**Software:**
- Python 3.10+ ([python.org](https://python.org) or system package)
- `uv` (fast package manager — recommended):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # restart shell after install
  ```
- OR `pip` (comes with Python)

---

## 2. Get the Code

```bash
git clone <repository-url> vvip_r
cd vvip_r
```

---

## 3. Create Environment & Install Dependencies

### With uv (recommended — 3× faster)

```bash
uv venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

uv pip install -r requirements.txt
```

### With pip

```bash
python3 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### What gets installed

All versions are pinned exactly to prevent future breakage:
```
avalanche-lib==0.6.0     # Continual learning framework (auto-downloads datasets)
torch==2.12.0            # Deep learning backend
torchvision==0.27.0      # Image transforms
numpy==2.4.6             # Numerical computing
pandas==3.0.3            # CSV processing
matplotlib==3.10.9       # Figure generation
scikit-learn==1.9.0      # Metrics
pillow==12.2.0           # Image handling
wandb==0.27.0            # Experiment logging (optional, never called)
codecarbon==3.2.7        # Carbon tracking (optional, never called)
thop==0.1.1              # FLOP counting (optional, never called)
```

---

## 4. Verify Installation

```bash
# Check imports work
python3 -c "
import torch, avalanche, matplotlib, pandas, numpy, sklearn, PIL
print(f'torch {torch.__version__}, avalanche {avalanche.__version__}')
print(f'matplotlib {matplotlib.__version__}, pandas {pandas.__version__}')
print(f'numpy {numpy.__version__}, sklearn {sklearn.__version__}')
print('All imports OK')
"

# Check EA-NPS module loads
python3 -c "
from ea_nps_strategy import EANPS, NPSComputer, EnergyProfiler
print('EA-NPS module: NPSComputer, EnergyProfiler, EANPS loaded OK')
"

# Check if GPU is available
python3 -c "
import torch; print(f'GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

**Expected output:**
```
torch 2.12.0, avalanche 0.6.0
matplotlib 3.10.9, pandas 3.0.3
...
EA-NPS module: NPSComputer, EnergyProfiler, EANPS loaded OK
GPU available: True
GPU: Tesla T4  (or similar)
```

---

## 5. Datasets (Auto-Downloaded)

All datasets download **automatically** the first time you run an experiment:

| Dataset | Triggered by | Download location | Size |
|---|---|---|---|
| MNIST | Any `experiments_permuted_mnist.py` run | `~/.avalanche/data/mnist/` | ~10 MB |
| CORe50 | `experiments_core50.py` run | `~/.avalanche/data/core50/` | ~300 MB |

**No manual download needed.** Avalanche handles everything. First run will show download progress bars.

**CORe50 note:** First download takes ~5 minutes (300 MB from Google Drive). Cached afterward. If download fails (rare), see [Avalanche CORe50 docs](https://avalanche.continualai.org/how-to#core50).

---

## 6. Fast Path: Reproduce Figures Only (~2 min)

If you only want to regenerate the figures from existing CSVs (no GPU needed):

```bash
# All 6 main figures
python3 generate_figures.py

# Also the proxy validation figure (requires ~5 min on CPU)
python3 validate_proxy.py
```

**Output:** 7 PNG files in `vip_res/figures/` + `vip_res/proxy_validation.csv`:
- `pareto_frontier.png`
- `learning_curves.png`
- `battery_routes.png`
- `per_task_accuracy.png`
- `accuracy_matrix_forgetting.png`
- `layerwise_heatmap.png`
- `proxy_validation.png`

All at 200 DPI, publication quality.

---

## 7. Full Path: Run Experiments from Scratch (GPU required)

### 7a. PermutedMNIST — All Experiments (Expts 1–5)

This single script runs:
1. **Expt 1:** 5 strategies × 3 seeds — main benchmark
2. **Expt 2–3:** Battery-accuracy tradeoff (SplitMNIST + PermutedMNIST)
3. **Expt 4–5:** Ablation (default decay + fast decay)

```bash
python3 experiments_permuted_mnist.py
```

**What happens step by step:**
1. Avalanche installs inline (first run only)
2. MNIST downloads automatically (~10 MB, once)
3. Run 5 strategies × 3 seeds on PermutedMNIST (15 runs)
4. Run EA-NPS on SplitMNIST with default decay
5. Run EA-NPS on SplitMNIST with fast decay (0.25/task)
6. Run EA-NPS on PermutedMNIST with fast decay
7. Run 3 ablation variants × 3 seeds with default decay
8. Run 3 ablation variants × 3 seeds with fast decay

**Output CSVs** (in current directory):

| File | Rows | Contents |
|---|---|---|
| `permuted_mnist_multiseed.csv` | 15 | 5 strats × 3 seeds, full accuracy matrix |
| `battery_full.csv` | 1 | SplitMNIST + default decay |
| `battery_fast.csv` | 1 | SplitMNIST + fast decay (0.25/task) |
| `permuted_battery.csv` | 1 | PermutedMNIST + fast decay |
| `ablation.csv` | 9 | 3 ablations × 3 seeds, default decay |
| `ablation_fast.csv` | 9 | 3 ablations × 3 seeds, fast decay |

**To verify results match the paper:**
```bash
python3 -c "
import pandas as pd
pm = pd.read_csv('permuted_mnist_multiseed.csv')
print(pm.groupby('strategy')['final_accuracy'].agg(['mean','std']).round(4))
"
```

Expected:
```
            mean     std
strategy                
derpp     0.9717  0.0006
ea_nps    0.9585  0.0023
er        0.9605  0.0010
ewc       0.7621  0.0479
naive     0.7411  0.0389
```

If your numbers differ by more than 0.01, check random seeds and hyperparameters match.

**Expected runtime:** ~25 minutes on Kaggle T4. ~4 hours on CPU.

### 7b. CORe50 — Hard Benchmark

```bash
python3 experiments_core50.py
```

**What happens:**
1. CORe50 downloads (~300 MB, ~5 min first time)
2. All data pre-cached to GPU for fast evaluation
3. Run 5 strategies × 3 seeds on 9 experiences each

**Output:** `core50_results.csv` (15 rows, 5 strats × 3 seeds)

**Verify:**
```bash
python3 -c "
import pandas as pd
c50 = pd.read_csv('core50_results.csv')
print(c50.groupby('strategy')['final_accuracy'].agg(['mean','std']).round(4))
"
```

Expected:
```
            mean     std
strategy                
derpp     0.1055  0.0156
ea_nps    0.0301  0.0072
er        0.0248  0.0063
ewc       0.0200  0.0001
naive     0.0200  0.0000
```

**Important:** CORe50 with a 630K-param CNN is at the edge of learnability. Expect ±0.01 variance. DER++ should always be above 0.07; Naive and EWC should be at 0.02 (chance for 50 classes).

**Expected runtime:** ~45 minutes on T4. ~8+ hours on CPU (not recommended).

### 7c. Fast-Decay Ablation (Standalone)

Skip this if you already ran Section 7a (Expt 5 is included there).

```bash
python3 experiments_ablation.py
```

Output: `ablation_fast.csv` (9 rows, 3 ablations × 3 seeds). ~15 minutes on T4.

### 7d. Proxy Validation (CPU OK)

```bash
python3 validate_proxy.py
```

Outputs `vip_res/proxy_validation.csv` (per-seed data) + `vip_res/figures/proxy_validation.png`. ~5 minutes on any machine.

---

## 8. Move CSVs & Regenerate Figures

After experiments complete, copy CSVs and regenerate figures:

```bash
# Move CSVs to vip_res/
cp permuted_mnist_multiseed.csv vip_res/
cp core50_results.csv vip_res/
cp battery_full.csv battery_fast.csv permuted_battery.csv vip_res/
cp ablation.csv ablation_fast.csv vip_res/

# Regenerate figures from the new data
python3 generate_figures.py
```

---

## 9. Kaggle Deployment (Alternative to Local GPU)

If you don't have a GPU, run experiments on Kaggle:

1. Go to [kaggle.com](https://kaggle.com) → Create → New Notebook
2. Set Accelerator to "GPU T4 x2" (top-right Settings panel)
3. Delete all default cells
4. Create one cell and paste the entire contents of:
   - `experiments_permuted_mnist.py` OR
   - `experiments_core50.py` OR
   - `experiments_ablation.py`
5. Run all (Shift+Enter). Script installs avalanche-lib inline.
6. After completion, go to "/kaggle/working/" in the file browser (right panel)
7. Download the CSV files
8. Place them in `vip_res/` on your local machine
9. Run `python3 generate_figures.py` locally to create figures

**Note:** `proxy_validation.csv` is generated directly in `vip_res/` by `validate_proxy.py` — no copy step needed.

**Note:** Each experiment script is fully self-contained for Kaggle. No external files needed.

---

## 10. Expected Directory State After Full Reproduction

```
vvip_r/
├── ea_nps_strategy.py
├── experiments_permuted_mnist.py
├── experiments_core50.py
├── experiments_ablation.py
├── validate_proxy.py
├── generate_figures.py
├── requirements.txt
├── README.md
├── instructions.md
│
├── vip_res/
│   ├── permuted_mnist_multiseed.csv    # From Expt 1
│   ├── core50_results.csv              # From CORe50
│   ├── battery_full.csv                # From Expt 2a
│   ├── battery_fast.csv                # From Expt 2b
│   ├── permuted_battery.csv            # From Expt 3
│   ├── ablation.csv                    # From Expt 4
│   ├── ablation_fast.csv               # From Expt 5
│   ├── proxy_validation.csv            # Proxy per-seed agreement
│   │
│   └── figures/
│       ├── pareto_frontier.png          # Fig 1: Accuracy vs time
│       ├── learning_curves.png          # Fig 2: Trajectories + forgetting
│       ├── battery_routes.png           # Fig 3: Route flowcharts
│       ├── per_task_accuracy.png        # Fig 4: Accuracy matrices
│       ├── accuracy_matrix_forgetting.png  # Fig 5: Matrix + forgetting
│       ├── layerwise_heatmap.png        # Fig 6: NPS vs Fisher
│       └── proxy_validation.png         # Fig S1: Proxy agreement
│
├── permuted_mnist_multiseed.csv         # Generated by Expt (copy to vip_res/)
├── core50_results.csv                   # Generated by CORe50 (copy to vip_res/)
├── battery_full.csv                     # Generated by Expt (copy to vip_res/)
├── battery_fast.csv                     # Generated by Expt (copy to vip_res/)
├── permuted_battery.csv                 # Generated by Expt (copy to vip_res/)
├── ablation.csv                         # Generated by Expt (copy to vip_res/)
├── ablation_fast.csv                    # Generated by Expt (copy to vip_res/)
├── proxy_validation.csv                 # Generated by validate_proxy.py (in vip_res/)
```

---

## 11. Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `ImportError: No module named avalanche` | Dependencies not installed | `pip install -r requirements.txt` |
| `ImportError: ... from avalanche.training` | Avalanche version mismatch | `pip install avalanche-lib==0.6.0` |
| `CUDA out of memory` | Batch size too large for GPU | Reduce `batch_size=128` to `64` in the experiment script |
| `FileNotFoundError: vip_res/...` | Figures script before experiments | Run experiments first, OR use bundled CSVs |
| `URLError: ... CORe50` | CORe50 download interrupted | Run script again — resume supported |
| `matplotlib font warnings` | Missing font | Ignore — falls back to DejaVu Sans |
| `DeprecationWarning from avalanche` | Third-party deprecation | Ignore — no functional impact |
| `Results differ from paper by >0.01` | Random seed or hyperparameter mismatch | Check `nps_threshold=0.2`, `mem_size=2000`, `batch_size=128` |
| `RuntimeError: ... torch.set_num_threads` | Kaggle environment quirk | Ignore — harmless |
| `pip install hangs` | Network issue | Use `uv pip install` instead (3× faster) |

---

## 12. Code Overview (What Each File Does)

### `ea_nps_strategy.py` — Core Module

| Class | Purpose |
|---|---|
| `EnergyProfiler` | Estimates MACs for each strategy (SGD, ER, EWC, freeze) |
| `NPSComputer` | Computes gradient-based NPS + activation proxy per layer |
| `EANPSPlugin` | Avalanche plugin: routing policy, buffer management, freeze logic |
| `EANPS` | Avalanche strategy class wrapping EANPSPlugin |

### Experiment Scripts — What They Run

| Script | Strategies | Tasks | Seeds | Output |
|---|---|---|---|---|
| `experiments_permuted_mnist.py` | naive, ewc, er, derpp, ea_nps | 5 permuted MNIST | 42,43,44 | 6 CSVs |
| `experiments_core50.py` | naive, ewc, er, derpp, ea_nps | 9 CORe50 NC | 42,43,44 | 1 CSV |
| `experiments_ablation.py` | ea_nps, ea_nps_nps_only, ea_nps_energy_only | 5 permuted MNIST | 42,43,44 | 1 CSV |
| `validate_proxy.py` | ea_nps (gradient vs activation) | 1 permuted MNIST | 0-19 | 1 PNG + 1 CSV |

### Hyperparameters

| Parameter | Value | Where set |
|---|---|---|
| NPS threshold τ | 0.2 | `experiments_*` `run_strategy()` |
| Memory buffer size | 2000 | All experiment scripts |
| Training batch size | 128 | All experiment scripts |
| Eval batch size | 512 | All experiment scripts |
| Optimizer | Adam, lr=0.001 | All experiment scripts |
| EWC lambda | 1.0 | All experiment scripts |
| DER++ alpha / beta | 0.1 / 0.5 | All experiment scripts |
| Epochs per task (MNIST) | 3 | `experiments_permuted_mnist.py` |
| Epochs per task (CORe50) | 1 | `experiments_core50.py` |
| Battery decay Δ | 0.05 (default) / 0.25 (fast) | `run_strategy()` kwarg |
| Battery critical β | 0.2 | `EANPSPlugin.__init__` |
| Freeze ratio | top 10% of layers | `EANPSPlugin.before_training_exp` |
| Activation proxy | `use_activation_proxy=False` | `validate_proxy.py` enables for comparison |
