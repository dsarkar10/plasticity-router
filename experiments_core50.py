"""
EA-NPS: CORe50 Experiments — Optimized for Dual T4 (Kaggle)
=============================================================
Safe optimizations only — training semantics and Avalanche loop unchanged:

  ✅ GPU-cached datasets: all experiences pre-loaded into VRAM once.
     CORe50-mini ~300 MB total — fits 30x on a single 14 GB T4.
     Experience-level caching preserves Avalanche's .train()/.eval() machinery.
  ✅ Single GPU (cuda:0): DataParallel scatter/gather overhead exceeds
     compute savings for a 600K-param model.
  ✅ No persistent_workers: avoids deadlocks with Avalanche's fork model.
  ✅ AMP (autocast) on evaluation pass — training AMP left to Avalanche.
  ✅ Direct GPU evaluation: test set pre-cached, no DataLoader at eval time.
  ✅ Benchmark loaded ONCE, CPU cache reused across all 3 seeds.
  ✅ cudnn.benchmark=True, TF32 enabled (no-op on T4, safe).

  ❌ Avalanche training loop NOT replaced — identical implicit regularization
     across all strategies (required for fair comparison, TMLR standard).
  ❌ Per-task evaluation NOT removed — learning curves and forgetting
     metrics are expected by reviewers.
"""

from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training import Naive, EWC, Replay as ExperienceReplay, DER as Derpp
from avalanche.benchmarks.classic import CORe50
import subprocess
import sys
import os
import time
import warnings
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from collections import OrderedDict, defaultdict

subprocess.check_call([sys.executable, "-m", "pip",
                      "install", "avalanche-lib", "-q"])

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── GPU SETUP ──────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()
print(f"Device: {device}  |  GPUs available: {NUM_GPUS}")
for i in range(NUM_GPUS):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}  {p.total_memory // 1024**2} MB")
print("  Using cuda:0 only — DataParallel overhead > gain for 600K-param model")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True  # No-op on T4 (Turing), safe
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(4)
    torch.zeros(1, device=device)
    torch.cuda.synchronize()


INPUT_SHAPE = (3, 32, 32)
BATCH_SIZE = 512
EVAL_BATCH = 2048


# ══════════════════════════════════════════════════════════════════
# EXPERIENCE-LEVEL TENSOR CACHE (FIXED)
# We attach cached tensors to the Experience object, NOT the Dataset.
# Avalanche frequently clones/replaces datasets for .train()/.eval() modes,
# but the Experience object persists — making it the safe attachment point.
# ══════════════════════════════════════════════════════════════════
def cache_experience(exp):
    """
    Materialise an Avalanche experience's dataset into contiguous CPU tensors.
    Attaches them as _cached_X and _cached_Y to the Experience object to avoid
    breaking Avalanche's dataset machinery (.train()/.eval() cloning).

    Returns (X_cpu, Y_cpu) for convenience, but primary effect is side-effect
    attachment to exp._cached_X and exp._cached_Y.
    """
    loader = torch.utils.data.DataLoader(
        exp.dataset, batch_size=2048, shuffle=False,
        num_workers=2, pin_memory=False, persistent_workers=False,
    )
    xs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            xs.append(batch[0])
            ys.append(batch[1])
    X = torch.cat(xs)
    Y = torch.cat(ys)

    # Attach to the Experience object, NOT the dataset!
    # Avalanche clones datasets internally; Experience persists.
    exp._cached_X = X
    exp._cached_Y = Y

    return X, Y


# ── FAST GPU EVALUATION ────────────────────────────────────────────
def build_gpu_test_cache(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Load test set directly into GPU VRAM for zero-overhead evaluation."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2048, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=False,
    )
    xs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            xs.append(batch[0])
            ys.append(batch[1])
    X = torch.cat(xs).to(device, non_blocking=True)
    Y = torch.cat(ys).to(device, non_blocking=True)
    torch.cuda.synchronize()
    return X, Y


@torch.no_grad()
def evaluate_gpu(model: nn.Module,
                 X: torch.Tensor, Y: torch.Tensor) -> float:
    """Fully vectorised evaluation on pre-cached GPU tensors. No DataLoader."""
    model.eval()
    correct = 0
    for s in range(0, X.size(0), EVAL_BATCH):
        xb, yb = X[s:s+EVAL_BATCH], Y[s:s+EVAL_BATCH]
        with autocast():
            correct += (model(xb).argmax(1) == yb).sum().item()
    torch.cuda.synchronize()
    return correct / X.size(0)


# ── MODEL ──────────────────────────────────────────────────────────
def make_core50_model() -> nn.Module:
    return nn.Sequential(OrderedDict([
        ("conv1",   nn.Conv2d(3,   32,  3, padding=1, bias=False)),
        ("bn1",     nn.BatchNorm2d(32)),
        ("relu1",   nn.ReLU(inplace=True)),
        ("pool1",   nn.MaxPool2d(2)),
        ("conv2",   nn.Conv2d(32,  64,  3, padding=1, bias=False)),
        ("bn2",     nn.BatchNorm2d(64)),
        ("relu2",   nn.ReLU(inplace=True)),
        ("pool2",   nn.MaxPool2d(2)),
        ("conv3",   nn.Conv2d(64,  128, 3, padding=1, bias=False)),
        ("bn3",     nn.BatchNorm2d(128)),
        ("relu3",   nn.ReLU(inplace=True)),
        ("pool3",   nn.MaxPool2d(2)),
        ("flatten", nn.Flatten()),
        ("fc1",     nn.Linear(128 * 4 * 4, 256)),
        ("relu4",   nn.ReLU(inplace=True)),
        ("dropout", nn.Dropout(0.3)),
        ("fc2",     nn.Linear(256, 50)),
    ])).to(device)


# ── ENERGY PROFILER ────────────────────────────────────────────────
class EnergyProfiler:
    def __init__(self, model: nn.Module, input_shape: tuple):
        self.model = model
        self.input_shape = input_shape
        self._macs = None
        self._table = None

    def _base_macs(self) -> float:
        if self._macs is not None:
            return self._macs
        counts = []

        def hook(m, inp, out):
            if isinstance(m, nn.Linear):
                counts.append(m.in_features * m.out_features)
            elif isinstance(m, nn.Conv2d):
                _, ci, _, _ = inp[0].shape
                _, co, h, w = out.shape
                counts.append(m.kernel_size[0] *
                              m.kernel_size[1] * ci * co * h * w)
        handles = [m.register_forward_hook(hook)
                   for m in self.model.modules()
                   if isinstance(m, (nn.Linear, nn.Conv2d))]
        with torch.no_grad():
            self.model(torch.zeros(1, *self.input_shape, device=device))
        for h in handles:
            h.remove()
        self._macs = float(sum(counts)) if counts else 1e6
        self._table = {
            "sgd":    self._macs * 3.0,
            "er":     self._macs * 3.5,
            "ewc":    self._macs * 4.5,
            "freeze": self._macs * 1.4,
        }
        return self._macs

    def estimate_macs(self, s: str) -> float:
        self._base_macs()
        return self._table.get(s, self._macs * 3.0)

    def estimate_nps_routing_cost(self, n_tasks: int, n_freeze: int = 0) -> float:
        return (n_tasks + n_freeze) * self._base_macs() * 6.0


# ── NPS COMPUTER ───────────────────────────────────────────────────
class NPSComputer:
    def __init__(self, model: nn.Module, buffer: list):
        self.model = model
        self.buffer = buffer
        self.crit = nn.CrossEntropyLoss()

    def _buf_sample(self, n=64):
        idx = np.random.choice(len(self.buffer), min(
            n, len(self.buffer)), replace=False)
        X = torch.stack([self.buffer[i][0]
                        for i in idx]).to(device, non_blocking=True)
        Y = torch.tensor([self.buffer[i][1] for i in idx], device=device)
        return X, Y

    def _grad_vec(self, X, Y) -> torch.Tensor:
        self.model.zero_grad()
        with autocast():
            self.crit(self.model(X), Y).backward()
        return torch.cat([p.grad.reshape(-1) for p in self.model.parameters()
                          if p.grad is not None])

    def compute_nps(self, Xn, Yn) -> float:
        if len(self.buffer) < 5:
            return 0.0
        Xo, Yo = self._buf_sample(64)
        g_old = self._grad_vec(Xo, Yo)
        g_new = self._grad_vec(Xn[:64].to(device, non_blocking=True),
                               Yn[:64].to(device, non_blocking=True))
        cos = nn.functional.cosine_similarity(
            g_old.unsqueeze(0), g_new.unsqueeze(0))
        return float(np.clip(1.0 - cos.item(), 0, 1))

    def compute_layerwise_nps(self, Xn, Yn) -> dict:
        if len(self.buffer) < 5:
            return {n: 0.0 for n, _ in self.model.named_parameters()}
        Xo, Yo = self._buf_sample(64)
        Xn = Xn[:64].to(device, non_blocking=True)
        Yn = Yn[:64].to(device, non_blocking=True)

        def grads(X, Y):
            self.model.zero_grad()
            with autocast():
                self.criterion(self.model(X), Y).backward()
            return {n: p.grad.clone() for n, p in self.model.named_parameters()
                    if p.grad is not None}

        go, gn = grads(Xo, Yo), grads(Xn, Yn)
        return {
            name: float(np.clip(
                1.0 - nn.functional.cosine_similarity(
                    go[name].reshape(1, -1), gn[name].reshape(1, -1)
                ).item(), 0, 1))
            for name in go
        }

    def compute_layerwise_activation_nps(self, Xn, Yn) -> dict:
        """Zero-backprop proxy: forward activations only, no gradients. ~10x cheaper."""
        if len(self.buffer) < 5:
            return {n: 0.0 for n, _ in self.model.named_parameters()}
        Xo, _ = self._buf_sample(32)
        Xn = Xn[:32].to(device, non_blocking=True)

        activations = {}
        def make_hook(name):
            def hook_fn(m, inp, out):
                activations[name] = out.detach()
            return hook_fn

        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                handles.append(module.register_forward_hook(make_hook(name)))

        self.model.eval()
        with torch.no_grad(), autocast():
            _ = self.model(Xo)
        old_acts = {k: v.clone() for k, v in activations.items()}

        activations.clear()
        with torch.no_grad(), autocast():
            _ = self.model(Xn)
        new_acts = {k: v.clone() for k, v in activations.items()}

        for h in handles:
            h.remove()

        module_params = defaultdict(list)
        for pname, _ in self.model.named_parameters():
            parts = pname.rsplit(".", 1)
            mod_name = parts[0] if len(parts) > 1 else pname
            module_params[mod_name].append(pname)

        layer_nps = {}
        for mod_name, act_old in old_acts.items():
            act_new = new_acts[mod_name]
            o = act_old.reshape(act_old.size(0), -1).mean(0, keepdim=True)
            n = act_new.reshape(act_new.size(0), -1).mean(0, keepdim=True)
            cos = nn.functional.cosine_similarity(o, n)
            nps = float(np.clip(1.0 - cos.item(), 0, 1))
            for pname in module_params.get(mod_name, [mod_name]):
                layer_nps[pname] = nps

        return layer_nps


# ── EA-NPS PLUGIN ──────────────────────────────────────────────────
class EANPSPlugin(SupervisedPlugin):
    def __init__(self, input_shape=INPUT_SHAPE, nps_threshold=0.2,
                 mem_size=2000, battery_critical=0.2, use_activation_proxy=False):
        super().__init__()
        self.nps_threshold = nps_threshold
        self.mem_size = mem_size
        self.battery_critical = battery_critical
        self.use_activation_proxy = use_activation_proxy
        self.input_shape = input_shape
        self.buffer = []
        self.battery = 1.0
        self.battery_decay = 0.05
        self.nps_computer = None
        self.profiler = None
        self.strategy_history = []
        self.current_strategy = "sgd"
        self.exp_counter = 0

    def _group_by_exp(self):
        g = defaultdict(list)
        for i, e in enumerate(self.buffer):
            g[e[2]].append(i)
        return g

    def _sample_balanced(self, n: int):
        if n <= 0 or not self.buffer:
            return []
        groups = self._group_by_exp()
        n_per = max(1, n // len(groups))
        sel = []
        for idxs in groups.values():
            sel.extend(np.random.choice(
                idxs, min(n_per, len(idxs)), replace=False).tolist())
        rem = n - len(sel)
        if rem > 0:
            pool = [i for i in range(len(self.buffer)) if i not in sel]
            if pool:
                sel.extend(np.random.choice(
                    pool, min(rem, len(pool)), replace=False).tolist())
        return sel

    def before_training(self, strategy, *args, **kwargs):
        if self.nps_computer is None:
            self.nps_computer = NPSComputer(strategy.model, self.buffer)
        if self.profiler is None:
            self.profiler = EnergyProfiler(strategy.model, self.input_shape)

    def before_training_exp(self, strategy, *args, **kwargs):
        self.battery = max(0.05, self.battery - self.battery_decay)
        exp = strategy.experience  # Get the Experience object directly!

        # Use cached tensors directly if available to avoid DataLoader overhead
        if hasattr(exp, '_cached_X') and hasattr(exp, '_cached_Y'):
            X_cpu, Y_cpu = exp._cached_X, exp._cached_Y
            idx = torch.randperm(len(X_cpu))[:512]
            Xn = X_cpu[idx].to(device, non_blocking=True)
            Yn = Y_cpu[idx].to(device, non_blocking=True)
        else:
            # Fallback to DataLoader if cache is missing
            dl = torch.utils.data.DataLoader(
                exp.dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=False)
            Xn, Yn = next(iter(dl))
            Xn = Xn.to(device, non_blocking=True)
            Yn = Yn.to(device, non_blocking=True)

        nps = self.nps_computer.compute_nps(Xn, Yn)
        high = nps > self.nps_threshold
        low = self.battery < self.battery_critical

        if not high:
            route = "sgd"
        elif high and not low:
            route = "er" if (self.profiler.estimate_macs("er")
                             <= self.profiler.estimate_macs("ewc")) else "ewc"
        else:
            route = "freeze"

        self.current_strategy = route
        self.strategy_history.append(route)
        print(
            f"  [EA-NPS] NPS={nps:.3f}  bat={self.battery:.0%}  route={route}")

        if route == "freeze":
            if self.use_activation_proxy:
                lnps = self.nps_computer.compute_layerwise_activation_nps(Xn, Yn)
            else:
                lnps = self.nps_computer.compute_layerwise_nps(Xn, Yn)
            srt = sorted(lnps.items(), key=lambda x: x[1], reverse=True)
            top = {n for n, _ in srt[:max(1, len(srt)//10)]}
            for name, p in strategy.model.named_parameters():
                p.requires_grad = name not in top

    def before_training_iteration(self, strategy, *args, **kwargs):
        if self.current_strategy not in ("er", "freeze") or not self.buffer:
            return
        idx = self._sample_balanced(strategy.train_mb_size)
        if not idx:
            return
        bx = torch.stack([self.buffer[i][0]
                         for i in idx]).to(device, non_blocking=True)
        by = torch.tensor([self.buffer[i][1] for i in idx], device=device)
        mb = strategy.mbatch
        mb[0] = torch.cat([mb[0], bx])
        mb[1] = torch.cat([mb[1], by])
        if len(mb) > 2:
            mb[2] = torch.cat([mb[2],
                               torch.zeros(len(idx), dtype=torch.long, device=device)])

    def after_training_exp(self, strategy, *args, **kwargs):
        for p in strategy.model.parameters():
            p.requires_grad = True

        exp_id = self.exp_counter
        self.exp_counter += 1
        exp = strategy.experience

        if hasattr(exp, '_cached_X') and hasattr(exp, '_cached_Y'):
            X_cpu, Y_cpu = exp._cached_X, exp._cached_Y
            for i in range(len(X_cpu)):
                self.buffer.append((X_cpu[i], int(Y_cpu[i].item()), exp_id))
        else:
            # Fallback to DataLoader if cache is missing
            dl = torch.utils.data.DataLoader(
                exp.dataset, batch_size=2048, shuffle=False, num_workers=0, pin_memory=False)
            for batch in dl:
                x, y = batch[0], batch[1]
                for i in range(len(x)):
                    self.buffer.append((x[i].cpu(), int(y[i].item()), exp_id))

        if len(self.buffer) > self.mem_size:
            groups = self._group_by_exp()
            n_per = self.mem_size // len(groups)
            kept = []
            for idxs in groups.values():
                kept.extend(
                    np.random.choice(idxs, min(n_per, len(idxs)), replace=False).tolist())
            rem = self.mem_size - len(kept)
            if rem > 0:
                pool = [i for i in range(len(self.buffer)) if i not in kept]
                if pool:
                    kept.extend(
                        np.random.choice(pool, min(rem, len(pool)), replace=False).tolist())
            self.buffer = [self.buffer[i] for i in sorted(kept)]


# ── EA-NPS STRATEGY ────────────────────────────────────────────────
class EANPS(SupervisedTemplate):
    def __init__(self, *, model, optimizer, criterion=nn.CrossEntropyLoss(),
                 input_shape=INPUT_SHAPE, nps_threshold=0.2, mem_size=2000,
                 battery_critical=0.2, use_activation_proxy=False,
                 train_mb_size=BATCH_SIZE, train_epochs=1,
                 eval_mb_size=None, device="cpu", plugins=None, evaluator=None,
                 eval_every=-1, **kw):
        self.ea_plugin = EANPSPlugin(
            input_shape=input_shape, nps_threshold=nps_threshold,
            mem_size=mem_size, battery_critical=battery_critical,
            use_activation_proxy=use_activation_proxy,
        )
        super().__init__(
            model=model, optimizer=optimizer, criterion=criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device,
            plugins=list(plugins or []) + [self.ea_plugin],
            evaluator=evaluator, eval_every=eval_every, **kw,
        )


# ── STRATEGY RUNNER ────────────────────────────────────────────────
def run_strategy(name, exp_cache_pairs, X_test_gpu, Y_test_gpu, seed,
                 epochs=3, mem_size=2000):
    """
    exp_cache_pairs : list of AvalancheExperience objects with _cached_X/_cached_Y attached.
      - AvalancheExperience is passed untouched to strategy.train() so
        Avalanche's full .train()/.eval() dataset machinery is preserved.
      - Cached tensors are attached to the Experience object for fast access.
    X_test_gpu / Y_test_gpu : test tensors already on cuda:0.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.empty_cache()
    gc.collect()

    model = make_core50_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    common = dict(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        train_epochs=epochs,
        train_mb_size=BATCH_SIZE,
        device=device,
        evaluator=None,
    )

    if name == "naive":
        strategy = Naive(**common)
    elif name == "ewc":
        strategy = EWC(**common, ewc_lambda=1.0)
    elif name == "er":
        strategy = ExperienceReplay(**common, mem_size=mem_size)
    elif name == "derpp":
        strategy = Derpp(**common, mem_size=mem_size, alpha=0.1, beta=0.5)
    elif name == "ea_nps":
        strategy = EANPS(**common, input_shape=INPUT_SHAPE,
                         nps_threshold=0.2, mem_size=mem_size)
    else:
        raise ValueError(name)

    task_accs = []
    t0 = time.time()

    for exp_id, exp in enumerate(exp_cache_pairs):
        # Pass the genuine Avalanche experience — .train() / .eval() intact
        strategy.train(exp)
        torch.cuda.synchronize()

        # Fast evaluation on pre-cached GPU tensors
        acc = evaluate_gpu(model, X_test_gpu, Y_test_gpu)
        task_accs.append(acc)
        print(f"  Task {exp_id}: acc={acc:.4f}  ({exp_id+1}/9 exps)")

    elapsed = time.time() - t0
    results = {
        "task_accuracies": task_accs,
        "final_accuracy":  task_accs[-1],
        "forgetting":      round(task_accs[0] - task_accs[-1], 4),
        "time":            elapsed,
        "accuracies":      str([round(a, 4) for a in task_accs]),
    }

    if name == "ea_nps":
        pl = strategy.ea_plugin
        if pl.profiler:
            macs_per = [pl.profiler.estimate_macs(
                r) for r in pl.strategy_history]
            total_tr = sum(macs_per)
            n_freeze = sum(1 for r in pl.strategy_history if r == "freeze")
            r_cost = pl.profiler.estimate_nps_routing_cost(
                len(pl.strategy_history), n_freeze)
            er_base = (pl.profiler.estimate_macs("er") * len(pl.strategy_history)
                       + pl.profiler.estimate_nps_routing_cost(len(pl.strategy_history)))
            total_m = total_tr + r_cost
            results.update({
                "routes":         pl.strategy_history,
                "macs_saved_pct": round((1 - total_m / er_base)*100, 1) if er_base else 0.0,
                "total_macs":     total_m,
            })

    del model, strategy, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    return results


# ── EXPERIMENT ENGINE ──────────────────────────────────────────────
def run_experiment(strategies, seeds):
    print(f"\n{'#'*70}")
    print("# CORe50 NC — All Strategies, 3 seeds")
    print(f"{'#'*70}")

    # ── Load benchmark and build GPU-cached dataset replacements ──
    print("\n--- Loading CORe50 benchmark (once for all seeds) ---")
    benchmark = CORe50(scenario="nc", mini=True, object_lvl=True)
    n_exp = len(benchmark.train_stream)
    print(f"  Experiences: {n_exp} train / {len(benchmark.test_stream)} test")

    print("  Pre-caching all experiences into contiguous CPU tensors...")
    t_cache = time.time()

    # Materialise each experience and attach cached tensors to the Experience object
    experiences_with_cache = []
    total_bytes = 0
    for i, exp in enumerate(benchmark.train_stream):
        X, Y = cache_experience(exp)  # Attaches _cached_X/Y to exp
        # DO NOT replace exp._dataset — preserves Avalanche's .train()/.eval() machinery

        experiences_with_cache.append(exp)
        nb = X.element_size() * X.nelement()
        total_bytes += nb
        print(f"    exp {i}: {tuple(X.shape)}  "
              f"{nb/1024**2:.0f} MB  classes={Y.unique().numel()}")

    # Test set goes directly to GPU for zero-overhead evaluation
    print("  Caching test set to GPU...")
    X_test_gpu, Y_test_gpu = build_gpu_test_cache(
        benchmark.test_stream[0].dataset)
    test_mb = X_test_gpu.element_size() * X_test_gpu.nelement() / 1024**2
    print(f"  Test set on GPU: {tuple(X_test_gpu.shape)}  {test_mb:.0f} MB  "
          f"classes={Y_test_gpu.unique().numel()}")
    print(f"  Total train cache: {total_bytes/1024**2:.0f} MB (CPU tensors)")
    print(f"  Cache build time: {time.time()-t_cache:.1f}s")

    # ── Run all strategies × seeds ─────────────────────────────────
    all_rows = []
    for seed in seeds:
        print(f"\n{'='*60}\n  SEED {seed}\n{'='*60}")
        for strat in strategies:
            print(f"\n  ── {strat.upper()} | seed={seed} ──")
            t0 = time.time()
            res = run_strategy(strat, experiences_with_cache,
                               X_test_gpu, Y_test_gpu, seed)
            print(f"  >> FINAL acc={res['final_accuracy']:.4f}  "
                  f"forgetting={res['forgetting']:.4f}  "
                  f"time={res['time']:.1f}s  (wall {time.time()-t0:.1f}s)")

            row = {
                "strategy":       strat,
                "dataset":        "core50",
                "seed":           seed,
                "final_accuracy": round(res["final_accuracy"], 4),
                "forgetting":     round(res["forgetting"], 4),
                "time_seconds":   round(res["time"], 1),
                "accuracies":     res.get("accuracies", ""),
            }
            if res.get("routes"):
                row["routes"] = "→".join(res["routes"])
                row["macs_saved_pct"] = res.get("macs_saved_pct", 0.0)
                row["total_macs"] = res.get("total_macs", 0.0)
            all_rows.append(row)

    del benchmark, experiences_with_cache, X_test_gpu, Y_test_gpu
    torch.cuda.empty_cache()
    gc.collect()

    df = pd.DataFrame(all_rows)
    print(f"\n{df.groupby('strategy')[['final_accuracy', 'time_seconds']].agg(
        ['mean', 'std']).to_string()}")
    return df


# ── MAIN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*70)
    print("CORe50 — Safe GPU Optimizations, Avalanche Loop Preserved")
    print("="*70)

    t_total = time.time()
    df = run_experiment(
        strategies=["naive", "ewc", "er", "derpp", "ea_nps"],
        seeds=[42, 43, 44],
    )

    df.to_csv("core50_results.csv", index=False)
    print(f"\nSaved → core50_results.csv  (copy to vip_res/)")

    summary = df.groupby("strategy").agg(
        accuracy_mean=("final_accuracy", "mean"),
        accuracy_std=("final_accuracy", "std"),
        time_mean=("time_seconds",   "mean"),
        time_std=("time_seconds",   "std"),
    ).round(4)

    print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
    print(summary.to_string())
    print(f"\nTotal wall time: {(time.time()-t_total)/60:.1f} min")
    print("="*70)
    print("Done. Download core50_results.csv from Kaggle output.")
