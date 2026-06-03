# Build & Run Instructions

Reproduce every experiment and figure in the paper from a bare machine.

---

## 0. What You Will Get

| Step | Output | Time |
|---|---|---|
| Environment + deps | Working Python venv with all packages | ~10 min |
| Quick verification | EA-NPS module imports OK | ~30 sec |
| Generate figures (fast) | 7 PNG figures in `vip_res/figures/` | ~2 min |
| Proxy validation | `proxy_validation.png` + `proxy_validation.csv` | ~5 min |
| PermutedMNIST experiments | 6 CSV files with all results | ~25 min (GPU) |
| CORe50 experiments | `core50_results.csv` | ~45 min (GPU) |
| τ sweep | `tau_sweep.csv` + `tau_sweep_agg.csv` | ~30 min (GPU) |
| Dynamic baselines | `dynamic_baselines.csv` | ~30 min (GPU) |

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
  ```
- OR `pip` (comes with Python)

---

## 2. Get the Code

```bash
git clone https://github.com/dsarkar10/plasticity-router
cd plasticity-router
```

---

## 3. Create Environment & Install Dependencies

### With uv (recommended — 3× faster)

```bash
uv venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

uv pip install -r requirements.txt
```

### With pip

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### What gets installed

All versions pinned:
```
avalanche-lib==0.6.0
torch==2.12.0
torchvision==0.27.0
numpy==2.4.6
pandas==3.0.3
matplotlib==3.10.9
scikit-learn==1.9.0
pillow==12.2.0
wandb==0.27.0
codecarbon==3.2.7
thop==0.1.1
```

---

## 4. Verify Installation

```bash
python3 -c "
import torch, avalanche, matplotlib, pandas, numpy, sklearn, PIL
print(f'torch {torch.__version__}, avalanche {avalanche.__version__}')
print('All imports OK')
"

python3 -c "
from ea_nps_strategy import EANPS, NPSComputer, EnergyProfiler
print('EA-NPS module loaded OK')
"

python3 -c "
import torch; print(f'GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## 5. Datasets (Auto-Downloaded)

| Dataset | Triggered by | Download location | Size |
|---|---|---|---|
| MNIST | Any experiment script | `~/.avalanche/data/mnist/` | ~10 MB |
| CORe50 | `experiments_core50.py` | `~/.avalanche/data/core50/` | ~300 MB |

Avalanche handles all downloads automatically. First run shows progress bars. CORe50 first download takes ~5 minutes (Google Drive, 300 MB).

---

## 6. Fast Path: Reproduce Figures Only (~2 min)

If you just want figures from existing CSVs (no GPU):

```bash
python3 generate_figures.py
python3 validate_proxy.py
```

**Output:** 7 PNG files in `vip_res/figures/` + `vip_res/proxy_validation.csv`:
- `pareto_frontier.png`, `learning_curves.png`, `battery_routes.png`
- `per_task_accuracy.png`, `accuracy_matrix_forgetting.png`, `layerwise_heatmap.png`
- `proxy_validation.png`

All at 200 DPI.

---

## 7. Full Path: Run Experiments from Scratch (GPU required)

### 7a. PermutedMNIST — All Experiments (Expts 1–5)

```bash
python3 experiments_permuted_mnist.py
```

Runs: 5 strategies × 3 seeds main benchmark + battery tradeoff (SplitMNIST + PermutedMNIST) + ablation (default + fast decay).

**Output CSVs:**

| File | Rows | Contents |
|---|---|---|
| `permuted_mnist_multiseed.csv` | 15 | 5 strats × 3 seeds, full accuracy matrix |
| `battery_full.csv` | 1 | SplitMNIST + default decay (0.05/task) |
| `battery_fast.csv` | 1 | SplitMNIST + fast decay (0.25/task) |
| `permuted_battery.csv` | 1 | PermutedMNIST + fast decay |
| `ablation.csv` | 9 | 3 ablations × 3 seeds, default decay |
| `ablation_fast.csv` | 9 | 3 ablations × 3 seeds, fast decay |

**Verify:**
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

Expected runtime: ~25 min on Kaggle T4, ~4 hours on CPU.

### 7b. CORe50 — Hard Benchmark

```bash
python3 experiments_core50.py
```

Runs 5 strategies × 3 seeds on 9 CORe50 NC experiences. Output: `core50_results.csv` (15 rows).

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

Expected runtime: ~45 min on T4.

### 7c. Fast-Decay Ablation (Standalone)

Skip if you already ran 7a (Expt 5 included there).

```bash
python3 experiments_ablation.py
```

Output: `ablation_fast.csv` (9 rows). ~15 min on T4.

### 7d. τ Hyperparameter Sweep

```bash
python3 tau_sweep.py
```

Sweeps τ from 0.0 to 1.0 across 2 seeds on PermutedMNIST with fast decay (0.25/task). Outputs `tau_sweep.csv` (16 rows) + `tau_sweep_agg.csv` (8 rows aggregated). ~30 min on T4.

### 7e. Dynamic Baselines Comparison

Open `dynamic_baselines.ipynb` in Kaggle or Jupyter. Run all cells. Compares EA-NPS (weight-magnitude freeze), random freeze, early stopping, and ER on PermutedMNIST with fast decay. Output: `dynamic_baselines.csv` (12 rows). ~30 min on T4.

### 7f. Proxy Validation (CPU OK)

```bash
python3 validate_proxy.py
```

Outputs `vip_res/proxy_validation.csv` (20 rows, per-seed Jaccard) + `vip_res/figures/proxy_validation.png`. ~5 min on any machine.

---

## 8. Move CSVs & Regenerate Figures

```bash
cp permuted_mnist_multiseed.csv vip_res/
cp core50_results.csv vip_res/
cp battery_full.csv battery_fast.csv permuted_battery.csv vip_res/
cp ablation.csv ablation_fast.csv vip_res/
cp tau_sweep.csv tau_sweep_agg.csv vip_res/
cp dynamic_baselines.csv vip_res/

python3 generate_figures.py
```

---

## 9. Kaggle Deployment

1. Go to [kaggle.com](https://kaggle.com) → Create → New Notebook
2. Set Accelerator to "GPU T4 x2" (Settings panel)
3. Delete all default cells
4. Paste the entire contents of the desired script into one cell
5. `.py` files go directly into a code cell; `.ipynb` files can be uploaded via File → Upload
6. Run all

**Which script for which experiment:**

| Experiment | Script to upload |
|---|---|
| Main benchmark + battery + ablation | `experiments_permuted_mnist.py` |
| CORe50 | `experiments_core50.py` |
| Ablation standalone | `experiments_ablation.py` |
| τ sweep | `tau_sweep.py` |
| Dynamic baselines | `dynamic_baselines.ipynb` (upload as notebook) |

All scripts are self-contained — avalanche-lib installs inline.

---

## 10. Expected Directory State After Full Reproduction

```
plasticity-router/
├── ea_nps_strategy.py              # Core: NPS, energy, routing classes
├── experiments_permuted_mnist.py   # Expts 1-5: main benchmark + battery + ablation
├── experiments_core50.py           # CORe50 stress test
├── experiments_ablation.py         # Standalone fast-decay ablation
├── tau_sweep.py                    # τ hyperparameter sensitivity sweep
├── dynamic_baselines.ipynb         # Dynamic baselines comparison
├── validate_proxy.py               # 20-seed proxy validation
├── generate_figures.py             # Figure generation from CSVs
├── requirements.txt                # Pinned dependencies
├── README.md                       # Research overview
├── instructions.md                 # This file
│
├── vip_res/                        # All research outputs
│   ├── permuted_mnist_multiseed.csv    # Main benchmark (15 rows)
│   ├── core50_results.csv              # CORe50 (15 rows)
│   ├── battery_full.csv                # SplitMNIST default decay
│   ├── battery_fast.csv                # SplitMNIST fast decay
│   ├── permuted_battery.csv            # PermutedMNIST fast decay
│   ├── ablation.csv                    # Default decay ablation (9 rows)
│   ├── ablation_fast.csv               # Fast decay ablation (9 rows)
│   ├── proxy_validation.csv            # Per-seed Jaccard (20 rows)
│   ├── tau_sweep.csv                   # Raw τ sweep (16 rows)
│   ├── tau_sweep_agg.csv               # Aggregated τ sweep (8 rows)
│   ├── dynamic_baselines.csv           # Baselines comparison (12 rows)
│   │
│   └── figures/
│       ├── pareto_frontier.png          # Fig 1: Accuracy vs time
│       ├── learning_curves.png          # Fig 2: Trajectories + forgetting
│       ├── battery_routes.png           # Fig 3: Route flowcharts
│       ├── per_task_accuracy.png        # Fig 4: Accuracy matrices
│       ├── accuracy_matrix_forgetting.png  # Fig 5: Matrix + forgetting
│       ├── layerwise_heatmap.png        # Fig 6: NPS vs Fisher
│       └── proxy_validation.png         # Fig S1: Proxy agreement
```

---

## 11. Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `ImportError: No module named avalanche` | Dependencies not installed | `pip install -r requirements.txt` |
| `ImportError: ... from avalanche.training` | Avalanche version mismatch | `pip install avalanche-lib==0.6.0` |
| `CUDA out of memory` | Batch size too large | Reduce `batch_size=128` to `64` in script |
| `FileNotFoundError: vip_res/...` | Figures before experiments | Run experiments first, or use bundled CSVs |
| `Results differ by >0.01` | Parameter mismatch | Check τ=0.2, mem_size=2000, batch_size=128 |
| `pip install hangs` | Network issue | Use `uv pip install` instead |
| `RuntimeError: torch.set_num_threads` | Kaggle env quirk | Ignore — harmless |

---

## 12. Code Overview

### `ea_nps_strategy.py` — Core Module

| Class | Purpose |
|---|---|
| `EnergyProfiler` | Estimates MACs per strategy (SGD, ER, EWC, freeze) using Horowitz energy model |
| `NPSComputer` | Computes gradient-based NPS + per-layer activation proxy |
| `EANPSPlugin` | Avalanche plugin: routing policy, buffer management, selective freeze |
| `EANPS` | Avalanche strategy class wrapping EANPSPlugin |

### Experiment Scripts

| Script | Strategies | Tasks | Seeds | Output |
|---|---|---|---|---|
| `experiments_permuted_mnist.py` | naive, ewc, er, derpp, ea_nps | 5 PermutedMNIST | 42,43,44 | 6 CSVs |
| `experiments_core50.py` | naive, ewc, er, derpp, ea_nps | 9 CORe50 NC | 42,43,44 | 1 CSV |
| `experiments_ablation.py` | ea_nps, ea_nps_nps_only, ea_nps_energy_only | 5 PermutedMNIST | 42,43,44 | 1 CSV |
| `tau_sweep.py` | ea_nps (8 τ values) | 5 PermutedMNIST | 42,43 | 2 CSVs |
| `dynamic_baselines.ipynb` | ea_nps, random_freeze, early_stop, er | 5 PermutedMNIST | 42,43,44 | 1 CSV |
| `validate_proxy.py` | ea_nps (gradient vs activation proxy) | 1 PermutedMNIST | 0-19 | 1 PNG + 1 CSV |

### Hyperparameters

| Parameter | Value | Where set |
|---|---|---|
| NPS threshold τ | 0.2 (default), swept 0.0–1.0 | `experiments_*` scripts |
| Memory buffer | 2000 | All experiment scripts |
| Batch size (train) | 128 | All experiment scripts |
| Batch size (eval) | 512 | All experiment scripts |
| Optimizer | Adam, lr=0.001 | All experiment scripts |
| EWC lambda | 1.0 | All experiment scripts |
| DER++ alpha / beta | 0.1 / 0.5 | All experiment scripts |
| Epochs per task (MNIST) | 3 | `experiments_permuted_mnist.py` |
| Epochs per task (CORe50) | 1 | `experiments_core50.py` |
| Battery decay Δ | 0.05 (default) / 0.25 (fast) | `run_strategy()` kwarg |
| Battery critical β | 0.2 | `EANPSPlugin.__init__` |
| Freeze ratio | top 10% of layers | `EANPSPlugin.before_training_exp` |
| Early stopping patience | 2 epochs | `dynamic_baselines.ipynb` |
