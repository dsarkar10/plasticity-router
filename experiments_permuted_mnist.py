"""
EA-NPS: Complete Experiments for Kaggle (GPU)
==============================================
1. Multi-seed PermutedMNIST (naive, ewc, er, ea_nps, seeds 42-44)
2. Battery-accuracy tradeoff (SplitMNIST + PermutedMNIST, fast decay)
3. Ablation study (Full vs NPS-Only vs Energy-Only)
4. MACs quantification per route
5. Full accuracy matrix for forgetting curves

Upload to Kaggle and run with "Python + GPU" accelerator.
"""
import subprocess, sys, os, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict

# ── Install avalanche ──
subprocess.check_call([sys.executable, "-m", "pip", "install", "avalanche-lib", "-q"])

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── GPU optimizations ──
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

# ───────────────────────────── IMPORTS ─────────────────────────────
from avalanche.benchmarks.classic import PermutedMNIST, SplitMNIST
from avalanche.training import Naive, EWC, Replay as ExperienceReplay, DER as Derpp
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate

# ─────────────────────── EA-NPS STRATEGY (inline) ──────────────────
class EnergyProfiler:
    def __init__(self, model: nn.Module, input_shape: tuple):
        self.model = model
        self.input_shape = input_shape
        self.macs_cache = {}

    def estimate_macs(self, strategy: str) -> float:
        if strategy in self.macs_cache:
            return self.macs_cache[strategy]
        base_forward = self._count_macs_forward()
        base_backward = base_forward * 2
        costs = {
            "sgd": base_forward + base_backward,
            "er": base_forward + base_backward + base_forward * 0.5,
            "ewc": base_forward + base_backward + base_forward * 1.5,
            "freeze": base_forward + base_backward * 0.4,
        }
        self.macs_cache.update(costs)
        return costs.get(strategy, base_forward + base_backward)

    def estimate_nps_routing_cost(self, num_tasks: int, num_freezes: int = 0) -> float:
        per_task = 2 * self.estimate_macs("sgd")
        per_freeze = 2 * self.estimate_macs("sgd")
        return num_tasks * per_task + num_freezes * per_freeze

    def _count_macs_forward(self) -> float:
        total = 0.0; self.model.eval()
        dev = next(self.model.parameters()).device
        x = torch.zeros(1, *self.input_shape, device=dev)
        hooks = []
        def hook_fn(m, inp, out):
            if isinstance(m, nn.Linear):
                hooks.append(m.in_features * m.out_features)
        handles = [mod.register_forward_hook(hook_fn) for mod in self.model.modules() if isinstance(mod, nn.Linear)]
        with torch.no_grad(): self.model(x)
        for h in handles: h.remove()
        return sum(hooks) if hooks else 1_000_000.0


class NPSComputer:
    def __init__(self, model: nn.Module, buffer: list, device: str = "cpu"):
        self.model = model; self.buffer = buffer
        self.device = device; self.criterion = nn.CrossEntropyLoss()

    def compute_nps(self, x_new, y_new) -> float:
        if len(self.buffer) < 5: return 0.0
        idx = np.random.choice(len(self.buffer), min(32, len(self.buffer)), replace=False)
        old_x = torch.stack([self.buffer[i][0] for i in idx]).to(self.device)
        old_y = torch.tensor([self.buffer[i][1] for i in idx], device=self.device)
        x_new, y_new = x_new.to(self.device), y_new.to(self.device)
        self.model.zero_grad()
        self.criterion(self.model(old_x), old_y).backward()
        grad_old = torch.cat([p.grad.view(-1) for p in self.model.parameters()])
        self.model.zero_grad()
        self.criterion(self.model(x_new), y_new).backward()
        grad_new = torch.cat([p.grad.view(-1) for p in self.model.parameters()])
        cos = nn.functional.cosine_similarity(grad_old.unsqueeze(0), grad_new.unsqueeze(0))
        return float(np.clip(1.0 - cos.item(), 0, 1))

    def compute_layerwise_nps(self, x_new, y_new) -> dict:
        if len(self.buffer) < 5: return {n: 0.0 for n, _ in self.model.named_parameters()}
        idx = np.random.choice(len(self.buffer), min(32, len(self.buffer)), replace=False)
        old_x = torch.stack([self.buffer[i][0] for i in idx]).to(self.device)
        old_y = torch.tensor([self.buffer[i][1] for i in idx], device=self.device)
        self.model.zero_grad()
        self.criterion(self.model(old_x), old_y).backward()
        grads_old = {n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None}
        self.model.zero_grad()
        x_new, y_new = x_new.to(self.device), y_new.to(self.device)
        self.criterion(self.model(x_new), y_new).backward()
        grads_new = {n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None}
        layer_nps = {}
        for name in grads_old:
            g_old, g_new = grads_old[name].view(-1), grads_new[name].view(-1)
            cos = nn.functional.cosine_similarity(g_old.unsqueeze(0), g_new.unsqueeze(0))
            layer_nps[name] = float(np.clip(1.0 - cos.item(), 0, 1))
        return layer_nps


class EANPSPlugin(SupervisedPlugin):
    def __init__(self, input_shape: tuple, nps_threshold: float = 0.5,
                 mem_size: int = 2000, battery_critical: float = 0.2,
                 disable_energy: bool = False, disable_nps: bool = False):
        super().__init__()
        self.nps_threshold = nps_threshold; self.mem_size = mem_size
        self.battery_critical = battery_critical; self.input_shape = input_shape
        self.buffer = []; self.battery = 1.0; self.battery_decay = 0.05
        self.nps_computer = None; self.profiler = None
        self.strategy_history = []; self.current_strategy = "sgd"
        self.exp_counter = 0
        self.disable_energy = disable_energy
        self.disable_nps = disable_nps

    def _group_by_exp(self):
        groups = defaultdict(list)
        for i, entry in enumerate(self.buffer): groups[entry[2]].append(i)
        return groups

    def _sample_balanced(self, n: int):
        if n <= 0 or len(self.buffer) == 0: return []
        groups = self._group_by_exp()
        n_per = max(1, n // len(groups))
        selected = []
        for indices in groups.values():
            k = min(n_per, len(indices))
            selected.extend(np.random.choice(indices, k, replace=False).tolist())
        remaining = n - len(selected)
        if remaining > 0:
            pool = [i for i in range(len(self.buffer)) if i not in selected]
            if pool: selected.extend(np.random.choice(pool, min(remaining, len(pool)), replace=False).tolist())
        return selected

    def before_training(self, strategy, *args, **kwargs):
        if self.nps_computer is None:
            self.nps_computer = NPSComputer(strategy.model, self.buffer, strategy.device)
        if self.profiler is None:
            self.profiler = EnergyProfiler(strategy.model, self.input_shape)

    def before_training_exp(self, strategy, *args, **kwargs):
        self.battery = max(0.05, self.battery - self.battery_decay)
        exp = strategy.experience
        dl = torch.utils.data.DataLoader(exp.dataset, batch_size=128, shuffle=True)
        first_batch = next(iter(dl))
        x_new, y_new = first_batch[0].to(strategy.device), first_batch[1].to(strategy.device)
        nps = self.nps_computer.compute_nps(x_new, y_new)
        high_nps = nps > self.nps_threshold
        low_bat = self.battery < self.battery_critical
        energy = self.profiler.estimate_macs

        if self.disable_nps:
            high_nps = True  # always perceive conflict → energy-routing only

        if self.disable_energy:
            low_bat = False  # never critical → nps-routing only (no freeze)

        if not high_nps:
            strategy_name = "sgd"
        elif high_nps and not low_bat:
            er_cost, ewc_cost = energy("er"), energy("ewc")
            strategy_name = "er" if er_cost <= ewc_cost else "ewc"
        else:
            strategy_name = "freeze"
        self.current_strategy = strategy_name
        self.strategy_history.append(strategy_name)
        print(f"  [EA-NPS] NPS={nps:.3f} Bat={self.battery:.0%} Route={strategy_name}")
        if strategy_name == "freeze":
            layer_nps = self.nps_computer.compute_layerwise_nps(x_new, y_new)
            sorted_layers = sorted(layer_nps.items(), key=lambda x: x[1], reverse=True)
            num_to_freeze = max(1, int(len(sorted_layers) * 0.1))
            top_layers = set(name for name, _ in sorted_layers[:num_to_freeze])
            for name, param in strategy.model.named_parameters():
                param.requires_grad = name not in top_layers

    def before_training_iteration(self, strategy, *args, **kwargs):
        if self.current_strategy in ("er", "freeze") and len(self.buffer) > 0:
            idx = self._sample_balanced(strategy.train_mb_size)
            if not idx: return
            buf_x = torch.stack([self.buffer[i][0] for i in idx]).to(strategy.device)
            buf_y = torch.tensor([self.buffer[i][1] for i in idx], device=strategy.device)
            mbatch = strategy.mbatch
            mbatch[0] = torch.cat([mbatch[0], buf_x], dim=0)
            mbatch[1] = torch.cat([mbatch[1], buf_y], dim=0)
            if len(mbatch) > 2:
                buf_t = torch.zeros(len(idx), dtype=torch.long, device=strategy.device)
                mbatch[2] = torch.cat([mbatch[2], buf_t], dim=0)

    def after_training_exp(self, strategy, *args, **kwargs):
        for p in strategy.model.parameters(): p.requires_grad = True
        exp = strategy.experience; exp_idx = self.exp_counter; self.exp_counter += 1
        dl = torch.utils.data.DataLoader(exp.dataset, batch_size=128, shuffle=True)
        for batch in dl:
            x, y = batch[0], batch[1]
            for i in range(len(x)): self.buffer.append((x[i].cpu(), y[i].item(), exp_idx))
        if len(self.buffer) > self.mem_size:
            groups = self._group_by_exp()
            n_per = self.mem_size // len(groups)
            kept = []
            for indices in groups.values():
                if len(indices) > n_per: kept.extend(np.random.choice(indices, n_per, replace=False).tolist())
                else: kept.extend(indices)
            remaining = self.mem_size - len(kept)
            if remaining > 0:
                pool = [i for i in range(len(self.buffer)) if i not in kept]
                if pool: kept.extend(np.random.choice(pool, min(remaining, len(pool)), replace=False).tolist())
            self.buffer = [self.buffer[i] for i in sorted(kept)]

    def after_training(self, strategy, *args, **kwargs): pass


class EANPS(SupervisedTemplate):
    def __init__(self, *, model, optimizer, criterion=nn.CrossEntropyLoss(),
                 input_shape=(1, 28, 28), nps_threshold=0.5, mem_size=2000,
                 battery_critical=0.2, train_mb_size=128, train_epochs=1,
                 eval_mb_size=None, device="cpu", plugins=None, evaluator=None,
                 eval_every=-1, disable_energy=False, disable_nps=False, **base_kwargs):
        ea_plugin = EANPSPlugin(input_shape=input_shape, nps_threshold=nps_threshold,
                                mem_size=mem_size, battery_critical=battery_critical,
                                disable_energy=disable_energy, disable_nps=disable_nps)
        all_plugins = list(plugins or []) + [ea_plugin]
        self.ea_plugin = ea_plugin
        super().__init__(model=model, optimizer=optimizer, criterion=criterion,
                         train_mb_size=train_mb_size, train_epochs=train_epochs,
                         eval_mb_size=eval_mb_size, device=device, plugins=all_plugins,
                         evaluator=evaluator, eval_every=eval_every, **base_kwargs)

# ─────────────────────── EXPERIMENT ENGINE ─────────────────────────
def make_model():
    return nn.Sequential(OrderedDict([
        ("flatten", nn.Flatten()),
        ("fc1", nn.Linear(28*28, 256)), ("relu1", nn.ReLU()),
        ("fc2", nn.Linear(256, 128)), ("relu2", nn.ReLU()),
        ("fc3", nn.Linear(128, 10)),
    ]))

DATASETS = {
    "permuted_mnist": lambda seed: PermutedMNIST(n_experiences=5, seed=seed),
    "split_mnist": lambda seed: SplitMNIST(n_experiences=5, seed=seed),
}

def run_strategy(strategy_name, benchmark, seed, n_experiences=5, epochs_per_task=3,
                 mem_size=2000, batch_size=128, ea_initial_battery=None, ea_battery_decay=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model().to(device)
    common_kwargs = dict(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
                         criterion=nn.CrossEntropyLoss(), train_epochs=epochs_per_task,
                         train_mb_size=batch_size, device=device, evaluator=None)
    if strategy_name == "naive":
        strategy = Naive(**common_kwargs)
    elif strategy_name == "ewc":
        strategy = EWC(**common_kwargs, ewc_lambda=1.0)
    elif strategy_name == "er":
        strategy = ExperienceReplay(**common_kwargs, mem_size=mem_size)
    elif strategy_name == "derpp":
        strategy = Derpp(**common_kwargs, mem_size=mem_size, alpha=0.1, beta=0.5)
    elif strategy_name == "ea_nps":
        strategy = EANPS(**common_kwargs, input_shape=(1, 28, 28), nps_threshold=0.2, mem_size=mem_size)
    elif strategy_name == "ea_nps_nps_only":
        strategy = EANPS(**common_kwargs, input_shape=(1, 28, 28), nps_threshold=0.2, mem_size=mem_size, disable_energy=True)
    elif strategy_name == "ea_nps_energy_only":
        strategy = EANPS(**common_kwargs, input_shape=(1, 28, 28), nps_threshold=0.2, mem_size=mem_size, disable_nps=True)
    else:
        raise ValueError(strategy_name)
    if strategy_name.startswith("ea_nps") and ea_initial_battery is not None:
        strategy.ea_plugin.battery = ea_initial_battery
    if strategy_name.startswith("ea_nps") and ea_battery_decay is not None:
        strategy.ea_plugin.battery_decay = ea_battery_decay

    results = {"final_accuracy": 0.0, "forgetting": 0.0, "time": 0.0, "routes": [], "acc_matrix": []}
    t0 = time.time()
    for exp_idx, experience in enumerate(benchmark.train_stream):
        strategy.train(experience)
        model.eval()
        accs = []
        for te_idx in range(exp_idx + 1):
            correct, total = 0, 0
            dl = torch.utils.data.DataLoader(benchmark.test_stream[te_idx].dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
            with torch.no_grad():
                for x, y, *_ in dl:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += y.size(0)
            accs.append(correct / max(total, 1))
        avg_acc = np.mean(accs)
        results.setdefault("task_accuracies", []).append(avg_acc)
        results["acc_matrix"].append(accs)
        print(f"  Task {exp_idx}: avg_acc={avg_acc:.3f}  per_task={[round(a,3) for a in accs]}")

    results["time"] = time.time() - t0
    results["final_accuracy"] = results["task_accuracies"][-1]
    if len(results["task_accuracies"]) > 1:
        results["forgetting"] = results["task_accuracies"][0] - results["task_accuracies"][-1]

    # ── Energy quantification ──
    if hasattr(strategy, 'ea_plugin'):
        plugin = strategy.ea_plugin
        results["routes"] = plugin.strategy_history
        if plugin.profiler is not None:
            macs_per_route = [plugin.profiler.estimate_macs(r) for r in plugin.strategy_history]
            results["macs_per_route"] = macs_per_route
            results["total_training_macs"] = sum(macs_per_route)
            # NPS routing overhead (critical for fair accounting)
            num_freezes = sum(1 for r in plugin.strategy_history if r == "freeze")
            routing_macs = plugin.profiler.estimate_nps_routing_cost(len(plugin.strategy_history), num_freezes)
            results["routing_macs"] = routing_macs
            results["total_macs"] = results["total_training_macs"] + routing_macs
            # Compare to always-ER baseline (with NPS routing overhead for fair comparison)
            er_macs_per = plugin.profiler.estimate_macs("er")
            er_routing = plugin.profiler.estimate_nps_routing_cost(len(plugin.strategy_history), 0)
            results["er_baseline_macs"] = er_macs_per * len(plugin.strategy_history) + er_routing
            results["macs_saved_pct"] = round(
                (1 - results["total_macs"] / results["er_baseline_macs"]) * 100, 1
            ) if results["er_baseline_macs"] > 0 else 0.0

    return results


def run_experiment_set(label, strategies, seeds, dataset_name, epochs=3, ea_battery=None, ea_decay=None):
    print(f"\n{'#'*70}")
    print(f"# {label}")
    print(f"{'#'*70}")
    all_rows = []
    for seed in seeds:
        benchmark = DATASETS[dataset_name](seed)
        for strat in strategies:
            print(f"\n  {strat.upper()} | {dataset_name} | seed={seed}")
            res = run_strategy(strat, benchmark, seed, epochs_per_task=epochs,
                               ea_initial_battery=ea_battery, ea_battery_decay=ea_decay)
            print(f"  >> Acc={res['final_accuracy']:.4f}, Forgetting={res['forgetting']:.4f}, Time={res['time']:.1f}s")
            row = {"strategy": strat, "dataset": dataset_name, "seed": seed,
                   "final_accuracy": round(res["final_accuracy"], 4),
                   "forgetting": round(res["forgetting"], 4),
                   "time_seconds": round(res["time"], 1)}
            if res.get("routes"):
                row["routes"] = "→".join(res["routes"])
            if res.get("macs_saved_pct") is not None:
                row["macs_saved_pct"] = res["macs_saved_pct"]
            if res.get("routing_macs") is not None:
                row["routing_macs"] = res["routing_macs"]
            if res.get("task_accuracies"):
                row["task_accuracies"] = str(res["task_accuracies"])
            if res.get("acc_matrix"):
                row["acc_matrix"] = str(res["acc_matrix"])
            all_rows.append(row)
    df = pd.DataFrame(all_rows)
    print(f"\n{df.groupby('strategy')[['final_accuracy','time_seconds']].agg(['mean','std']).to_string()}")
    return df


# ════════════════════════ EXPERIMENT 1 ══════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 1: Multi-seed PermutedMNIST (naive, ewc, er, ea_nps)")
print("="*70)
df1 = run_experiment_set(
    "PermutedMNIST — All Strategies, 3 seeds",
    strategies=["naive", "ewc", "er", "derpp", "ea_nps"],
    seeds=[42, 43, 44],
    dataset_name="permuted_mnist",
    epochs=3
)
df1.to_csv("permuted_mnist_multiseed.csv", index=False)
print(f"\nSaved to permuted_mnist_multiseed.csv  (copy to vip_res/)")
print(f"\nSummary:")
summary = df1.groupby("strategy").agg(
    accuracy_mean=("final_accuracy", "mean"),
    accuracy_std=("final_accuracy", "std"),
    time_mean=("time_seconds", "mean"),
    time_std=("time_seconds", "std")
).round(4)
print(summary.to_string())


# ════════════════════════ EXPERIMENT 2 ══════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 2: Battery-Accuracy Tradeoff (SplitMNIST)")
print("="*70)

print("\n--- High battery (no decay, default 1.0) ---")
df2a = run_experiment_set(
    "EA-NPS on SplitMNIST — Full battery (no freeze expected)",
    strategies=["ea_nps"],
    seeds=[42],
    dataset_name="split_mnist",
)
# Note: default decay is 0.05, so battery goes: 1.0, 0.95, 0.90, 0.85, 0.80 (never below 0.2)

print("\n\n--- Fast decay (0.25/task) to trigger freeze ---")
df2b = run_experiment_set(
    "EA-NPS on SplitMNIST — Fast decay 0.25/task",
    strategies=["ea_nps"],
    seeds=[42],
    dataset_name="split_mnist",
    ea_battery=1.0,
    ea_decay=0.25
)
# Battery trajectory: 1.0 → 0.75 → 0.50 → 0.25 → 0.05 (< 0.2 critical trigger at task 3)

df2a.to_csv("battery_full.csv", index=False)
df2b.to_csv("battery_fast.csv", index=False)
print(f"\nSaved to battery_full.csv and battery_fast.csv  (copy to vip_res/)")

# ════════════════════════ EXPERIMENT 3 ══════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 3: PermutedMNIST Battery Demo")
print("="*70)
df3 = run_experiment_set(
    "EA-NPS on PermutedMNIST — Fast decay 0.25/task",
    strategies=["ea_nps"],
    seeds=[42],
    dataset_name="permuted_mnist",
    ea_battery=1.0,
    ea_decay=0.25,
)
df3.to_csv("permuted_battery.csv", index=False)

# ════════════════════════ EXPERIMENT 4 ══════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 4: Ablation — NPS-Only vs Energy-Only vs Full EA-NPS")
print("="*70)
df4 = run_experiment_set(
    "PermutedMNIST — Ablation (Full, NPS-Only, Energy-Only), 3 seeds",
    strategies=["ea_nps", "ea_nps_nps_only", "ea_nps_energy_only"],
    seeds=[42, 43, 44],
    dataset_name="permuted_mnist",
    epochs=3,
)
df4.to_csv("ablation.csv", index=False)
print(f"\nSaved to ablation.csv  (copy to vip_res/)")

# ════════════════════════ EXPERIMENT 5 ══════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 5: Ablation with Fast Decay (meaningful comparison)")
print("="*70)
df5 = run_experiment_set(
    "PermutedMNIST — Ablation with fast decay 0.25/task, 3 seeds",
    strategies=["ea_nps", "ea_nps_nps_only", "ea_nps_energy_only"],
    seeds=[42, 43, 44],
    dataset_name="permuted_mnist",
    epochs=3,
    ea_battery=1.0,
    ea_decay=0.25,
)
df5.to_csv("ablation_fast.csv", index=False)
print(f"\nSaved to ablation_fast.csv  (copy to vip_res/)")
print(f"\nExpected route divergence:")
print(f"  Full EA-NPS:      sgd→er→er→freeze→freeze  (NPS-driven SGD T0 + freeze at low bat)")
print(f"  NPS-Only:         sgd→er→er→er→er          (no freeze, always ER when NPS>threshold)")
print(f"  Energy-Only:      er→er→er→freeze→freeze  (always ER when bat>20%, freeze when low)")

# ════════════════════════ FINAL SUMMARY ═════════════════════════════
print("\n" + "="*70)
print("FINAL SUMMARY — All Experiments")
print("="*70)
print(f"\n--- PermutedMNIST Multi-Seed ---")
print(summary.to_string())
print(f"\n--- Battery-Accuracy Tradeoff Table ---")
batt_rows = []
for name, df in [("SplitMNIST+FullBat", df2a), ("SplitMNIST+FastBat", df2b), ("PermutedMNIST+FastBat", df3)]:
    for _, r in df.iterrows():
        batt_rows.append({
            "experiment": name,
            "final_accuracy": r["final_accuracy"],
            "routes": r.get("routes", ""),
            "macs_saved_pct": r.get("macs_saved_pct", "N/A"),
        })
batt_table = pd.DataFrame(batt_rows)
print(batt_table.to_string(index=False))

print(f"\n\n--- Ablation Study (Default Decay) ---")
abl_summary = df4.groupby("strategy").agg(
    accuracy_mean=("final_accuracy", "mean"),
    forgetting_mean=("forgetting", "mean"),
).round(4)
print(abl_summary.to_string())

print(f"\n--- Ablation Study (Fast Decay 0.25/task) ---")
abl_fast_summary = df5.groupby("strategy").agg(
    accuracy_mean=("final_accuracy", "mean"),
    accuracy_std=("final_accuracy", "std"),
    forgetting_mean=("forgetting", "mean"),
    routes=("routes", "first"),
).round(4)
print(abl_fast_summary.to_string())

print(f"\nCSVs saved: permuted_mnist_multiseed.csv, battery_full.csv, battery_fast.csv, permuted_battery.csv, ablation.csv, ablation_fast.csv")

print(f"\n\nInterpretation:")
print(f"  - Full EA-NPS: NPS-thresholded routing + energy-aware strategy selection + freeze fallback.")
print(f"  - NPS-Only: selects ER whenever NPS > threshold; no freeze (ignores battery).")
print(f"  - Energy-Only: freezes when battery < critical; otherwise ER (ignores NPS).")
print(f"  - Fast decay triggers freeze at <20% battery, saving energy at accuracy cost.")
print(f"  - MACs saved = cumulative MACs(EA-NPS) vs always-ER baseline.")

print(f"\n{'='*70}")
print(f"All CSVs saved. Download them from Kaggle output.")
print(f"{'='*70}")
