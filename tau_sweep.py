import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.benchmarks.classic import PermutedMNIST
import sys
import subprocess
import time
import warnings
from collections import OrderedDict

# ==========================================
# 1. ROBUST INSTALLATION CHECK
# ==========================================


def ensure_avalanche_installed():
    try:
        import avalanche
        print("✅ Avalanche is already installed.")
        return
    except ImportError:
        print("⏳ Avalanche not found. Installing 'avalanche-lib'...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
                "avalanche-lib", "-q", "--no-warn-conflicts"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "avalanche-lib", "-q"],
                capture_output=True,
                text=True
            )
        if result.returncode != 0:
            print("❌ Installation failed! Error output:")
            print(result.stderr)
            raise RuntimeError("Failed to install avalanche-lib.")
        print("✅ Installation successful.")


ensure_avalanche_installed()

# ==========================================
# 2. IMPORTS
# ==========================================

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"System ready. Using Device: {device}")

# ---------- Model & Energy Profiler Constants ----------


def make_model():
    return nn.Sequential(OrderedDict([
        ("flatten", nn.Flatten()),
        ("fc1", nn.Linear(28*28, 256)), ("relu1", nn.ReLU()),
        ("fc2", nn.Linear(256, 128)), ("relu2", nn.ReLU()),
        ("fc3", nn.Linear(128, 10)),
    ]))


FWD_MACS = 234_752
COSTS = {
    "sgd": FWD_MACS * 3,
    "er": FWD_MACS * 3.5,
    "freeze": FWD_MACS * 1.5
}

# ---------- Optimized Plugin for Kaggle T4 ----------


class EANPSPlugin(SupervisedPlugin):
    def __init__(self, nps_threshold=0.2, mem_size=2000):
        super().__init__()
        self.nps_threshold = nps_threshold
        self.mem_size = mem_size
        self.battery = 1.0
        self.buffer = []
        self.strategy_history = []
        self.current_strat = "sgd"

    def before_training_exp(self, strategy, **kwargs):
        self.battery = max(0.05, self.battery - 0.25)

        if len(self.buffer) < 50:
            nps = 1.0
        else:
            idx = np.random.choice(len(self.buffer), min(
                64, len(self.buffer)), replace=False)
            old_x = torch.stack([self.buffer[i][0]
                                for i in idx]).to(strategy.device).detach()
            old_y = torch.tensor([self.buffer[i][1]
                                 for i in idx], device=strategy.device)

            dl = torch.utils.data.DataLoader(
                strategy.experience.dataset, batch_size=64, shuffle=True)
            batch = next(iter(dl))
            x_new = batch[0].to(strategy.device).detach()
            y_new = batch[1].to(strategy.device)

            strategy.model.zero_grad()
            nn.CrossEntropyLoss()(strategy.model(old_x), old_y).backward()
            g_old = torch.cat(
                [p.grad.view(-1) for p in strategy.model.parameters() if p.grad is not None])

            strategy.model.zero_grad()
            nn.CrossEntropyLoss()(strategy.model(x_new), y_new).backward()
            g_new = torch.cat(
                [p.grad.view(-1) for p in strategy.model.parameters() if p.grad is not None])

            cos = nn.functional.cosine_similarity(
                g_old.unsqueeze(0), g_new.unsqueeze(0))
            nps = float(torch.clamp(1.0 - cos, 0, 1).item())

        if nps <= self.nps_threshold:
            strat = "sgd"
        elif self.battery < 0.2:
            strat = "freeze"
        else:
            strat = "er"

        self.strategy_history.append(strat)
        self.current_strat = strat

        if strat == "freeze":
            for name, param in strategy.model.named_parameters():
                if "fc1" in name:
                    param.requires_grad = False

    def before_training_iteration(self, strategy, **kwargs):
        if self.current_strat in ("er", "freeze") and len(self.buffer) > 0:
            sample_size = min(strategy.train_mb_size, len(self.buffer))
            idx = np.random.choice(
                len(self.buffer), sample_size, replace=False)

            bx = torch.stack([self.buffer[i][0]
                             for i in idx]).to(strategy.device)
            by = torch.tensor([self.buffer[i][1]
                              for i in idx], device=strategy.device)

            new_x = torch.cat([strategy.mbatch[0], bx], dim=0)
            new_y = torch.cat([strategy.mbatch[1], by], dim=0)

            # FIX: Use * unpacking to safely handle both lists and tuples
            strategy.mbatch = (new_x, new_y, *strategy.mbatch[2:])

    def after_training_exp(self, strategy, **kwargs):
        for p in strategy.model.parameters():
            p.requires_grad = True

        ds = strategy.experience.dataset
        idx = np.random.choice(len(ds), min(400, len(ds)), replace=False)
        for i in idx:
            img, label, *_ = ds[i]
            self.buffer.append((img.detach().cpu(), label))

        if len(self.buffer) > self.mem_size:
            self.buffer = self.buffer[-self.mem_size:]

# ---------- Sweep Execution Logic ----------


def run_trial(tau, seed, benchmark):
    torch.manual_seed(seed)
    model = make_model().to(device)
    plugin = EANPSPlugin(nps_threshold=tau)

    strategy = SupervisedTemplate(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        criterion=nn.CrossEntropyLoss(),
        train_mb_size=128,
        train_epochs=2,
        device=device,
        plugins=[plugin]
    )

    t0 = time.time()
    for exp in benchmark.train_stream:
        strategy.train(exp)

    model.eval()
    correct, total = 0, 0
    for exp in benchmark.test_stream:
        dl = torch.utils.data.DataLoader(
            exp.dataset, batch_size=1024, num_workers=0, pin_memory=True)
        with torch.no_grad():
            for x, y, *_ in dl:
                correct += (model(x.to(device)).argmax(1)
                            == y.to(device)).sum().item()
                total += y.size(0)

    macs_used = sum(COSTS[s] for s in plugin.strategy_history)
    macs_baseline = COSTS["er"] * 5

    return {
        "tau": tau,
        "seed": seed,
        "acc": round(correct/total, 4),
        "saved_pct": round((1 - macs_used/macs_baseline)*100, 1),
        "routes": "→".join(plugin.strategy_history),
        "time": round(time.time()-t0, 1)
    }


# ===== KAGGLE RUNNER =====
if __name__ == "__main__":
    print("Loading PermutedMNIST Dataset...")
    bm = PermutedMNIST(n_experiences=5, seed=42)

    tau_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    seeds = [42, 43]
    results = []

    print(f"Starting GPU sweep...")
    for tau in tau_values:
        for seed in seeds:
            res = run_trial(tau, seed, bm)
            results.append(res)
            print(f"Tau: {tau:<3} | Seed: {seed} | Acc: {res['acc']:.4f} | MACs Saved: {
                  res['saved_pct']:>4}% | Route: {res['routes']:<25} | {res['time']}s")

    df = pd.DataFrame(results)
    df.to_csv("tau_sweep.csv", index=False)

    agg = df.groupby("tau").agg({
        "acc": ["mean", "std"],
        "saved_pct": "mean"
    }).reset_index()
    agg.to_csv("tau_sweep_agg.csv", index=False)

    print("\n✅ Sweep Complete. Results saved to 'tau_sweep.csv' and 'tau_sweep_agg.csv'")
    print(agg.to_string())
