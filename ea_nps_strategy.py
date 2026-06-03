import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins import SupervisedPlugin
from typing import Optional, List, Union, Callable
from collections import OrderedDict


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

    def _count_macs_forward(self) -> float:
        total = 0.0
        self.model.eval()
        x = torch.zeros(1, *self.input_shape)
        hooks = []

        def hook_fn(m, inp, out):
            if isinstance(m, nn.Linear):
                total_macs = m.in_features * m.out_features
                hooks.append(total_macs)

        handles = []
        for mod in self.model.modules():
            if isinstance(mod, nn.Linear):
                h = mod.register_forward_hook(hook_fn)
                handles.append(h)

        with torch.no_grad():
            self.model(x)

        for h in handles:
            h.remove()

        macs = sum(hooks)
        return macs if macs > 0 else 1_000_000.0


class NPSComputer:
    def __init__(self, model: nn.Module, buffer: list, device: str = "cpu"):
        self.model = model
        self.buffer = buffer
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def compute_nps(self, x_new: torch.Tensor, y_new: torch.Tensor) -> float:
        if len(self.buffer) < 5:
            return 0.0

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

        cos = F.cosine_similarity(grad_old.unsqueeze(0), grad_new.unsqueeze(0))
        return float(np.clip(1.0 - cos.item(), 0, 1))

    def compute_layerwise_nps(self, x_new: torch.Tensor, y_new: torch.Tensor) -> dict:
        if len(self.buffer) < 5:
            return {n: 0.0 for n, _ in self.model.named_parameters()}

        self.model.zero_grad()
        idx = np.random.choice(len(self.buffer), min(32, len(self.buffer)), replace=False)
        old_x = torch.stack([self.buffer[i][0] for i in idx]).to(self.device)
        old_y = torch.tensor([self.buffer[i][1] for i in idx], device=self.device)

        self.criterion(self.model(old_x), old_y).backward()
        grads_old = {n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None}

        self.model.zero_grad()
        x_new, y_new = x_new.to(self.device), y_new.to(self.device)
        self.criterion(self.model(x_new), y_new).backward()
        grads_new = {n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None}

        layer_nps = {}
        for name in grads_old:
            g_old = grads_old[name].view(-1)
            g_new = grads_new[name].view(-1)
            cos = F.cosine_similarity(g_old.unsqueeze(0), g_new.unsqueeze(0))
            layer_nps[name] = float(np.clip(1.0 - cos.item(), 0, 1))

        return layer_nps

    def compute_layerwise_activation_nps(self, x_new: torch.Tensor, y_new: torch.Tensor) -> dict:
        """
        Zero-backprop proxy: compares forward activations instead of gradients.
        ~10x cheaper than gradient-based layerwise NPS.

        Registers forward hooks on weight-bearing layers (Linear, Conv2d),
        passes old buffer and new batch through model (forward only),
        computes cosine similarity of output activations per layer.

        Returns dict keyed by parameter names (drop-in for compute_layerwise_nps).
        """
        if len(self.buffer) < 5:
            return {n: 0.0 for n, _ in self.model.named_parameters()}

        idx = np.random.choice(len(self.buffer), min(32, len(self.buffer)), replace=False)
        old_x = torch.stack([self.buffer[i][0] for i in idx]).to(self.device)
        x_new = x_new[:32].to(self.device)

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
        with torch.no_grad():
            _ = self.model(old_x)
        old_acts = {k: v.clone() for k, v in activations.items()}

        activations.clear()
        with torch.no_grad():
            _ = self.model(x_new)
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
            cos = F.cosine_similarity(o, n)
            nps = float(np.clip(1.0 - cos.item(), 0, 1))
            for pname in module_params.get(mod_name, [mod_name]):
                layer_nps[pname] = nps

        return layer_nps


class EANPSPlugin(SupervisedPlugin):
    def __init__(
        self,
        input_shape: tuple,
        nps_threshold: float = 0.5,
        mem_size: int = 2000,
        battery_critical: float = 0.2,
        use_activation_proxy: bool = False,
    ):
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
        groups = defaultdict(list)
        for i, entry in enumerate(self.buffer):
            groups[entry[2]].append(i)
        return groups

    def _sample_balanced(self, n: int):
        if n <= 0 or len(self.buffer) == 0:
            return []
        groups = self._group_by_exp()
        n_per = max(1, n // len(groups))
        selected = []
        for indices in groups.values():
            k = min(n_per, len(indices))
            selected.extend(np.random.choice(indices, k, replace=False).tolist())
        remaining = n - len(selected)
        if remaining > 0:
            pool = [i for i in range(len(self.buffer)) if i not in selected]
            if pool:
                selected.extend(np.random.choice(pool, min(remaining, len(pool)), replace=False).tolist())
        return selected

    def before_training(self, strategy: "SupervisedTemplate", *args, **kwargs):
        if self.nps_computer is None:
            self.nps_computer = NPSComputer(strategy.model, self.buffer, strategy.device)
        if self.profiler is None:
            self.profiler = EnergyProfiler(strategy.model, self.input_shape)

    def before_training_exp(self, strategy: "SupervisedTemplate", *args, **kwargs):
        self.battery = max(0.05, self.battery - self.battery_decay)

        exp = strategy.experience
        dl = torch.utils.data.DataLoader(exp.dataset, batch_size=128, shuffle=True)
        first_batch = next(iter(dl))
        x_new, y_new = first_batch[0], first_batch[1]
        x_new = x_new.to(strategy.device) if hasattr(x_new, "to") else torch.tensor(x_new).to(strategy.device)
        y_new = y_new.to(strategy.device) if hasattr(y_new, "to") else torch.tensor(y_new).to(strategy.device)

        nps = self.nps_computer.compute_nps(x_new, y_new)
        high_nps = nps > self.nps_threshold
        low_bat = self.battery < self.battery_critical
        energy = self.profiler.estimate_macs

        if not high_nps:
            strategy_name = "sgd"
        elif high_nps and not low_bat:
            er_cost = energy("er")
            ewc_cost = energy("ewc")
            strategy_name = "er" if er_cost <= ewc_cost else "ewc"
        else:
            strategy_name = "freeze"

        self.current_strategy = strategy_name
        self.strategy_history.append(strategy_name)
        print(f"  [EA-NPS] NPS={nps:.3f} Bat={self.battery:.0%} Route={strategy_name}")

        if strategy_name == "freeze":
            if self.use_activation_proxy:
                layer_nps = self.nps_computer.compute_layerwise_activation_nps(x_new, y_new)
            else:
                layer_nps = self.nps_computer.compute_layerwise_nps(x_new, y_new)
            sorted_layers = sorted(layer_nps.items(), key=lambda x: x[1], reverse=True)
            num_to_freeze = max(1, int(len(sorted_layers) * 0.1))
            top_layers = set(name for name, _ in sorted_layers[:num_to_freeze])
            for name, param in strategy.model.named_parameters():
                param.requires_grad = name not in top_layers

    def before_training_iteration(self, strategy: "SupervisedTemplate", *args, **kwargs):
        if self.current_strategy in ("er", "freeze") and len(self.buffer) > 0:
            idx = self._sample_balanced(strategy.train_mb_size)
            if not idx:
                return
            buf_x = torch.stack([self.buffer[i][0] for i in idx]).to(strategy.device)
            buf_y = torch.tensor([self.buffer[i][1] for i in idx], device=strategy.device)

            mbatch = strategy.mbatch
            mbatch[0] = torch.cat([mbatch[0], buf_x], dim=0)
            mbatch[1] = torch.cat([mbatch[1], buf_y], dim=0)
            if len(mbatch) > 2:
                buf_t = torch.zeros(len(idx), dtype=torch.long, device=strategy.device)
                mbatch[2] = torch.cat([mbatch[2], buf_t], dim=0)

    def after_training_exp(self, strategy: "SupervisedTemplate", *args, **kwargs):
        for p in strategy.model.parameters():
            p.requires_grad = True

        exp = strategy.experience
        exp_idx = self.exp_counter
        self.exp_counter += 1
        dl = torch.utils.data.DataLoader(exp.dataset, batch_size=128, shuffle=True)
        for batch in dl:
            x, y = batch[0], batch[1]
            for i in range(len(x)):
                self.buffer.append((x[i].cpu(), y[i].item(), exp_idx))
        if len(self.buffer) > self.mem_size:
            groups = self._group_by_exp()
            n_per = self.mem_size // len(groups)
            kept = []
            for indices in groups.values():
                if len(indices) > n_per:
                    kept.extend(np.random.choice(indices, n_per, replace=False).tolist())
                else:
                    kept.extend(indices)
            remaining = self.mem_size - len(kept)
            if remaining > 0:
                pool = [i for i in range(len(self.buffer)) if i not in kept]
                if pool:
                    kept.extend(np.random.choice(pool, min(remaining, len(pool)), replace=False).tolist())
            self.buffer = [self.buffer[i] for i in sorted(kept)]

    def after_training(self, strategy: "SupervisedTemplate", *args, **kwargs):
        pass


class EANPS(SupervisedTemplate):
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        input_shape: tuple = (1, 28, 28),
        nps_threshold: float = 0.5,
        mem_size: int = 2000,
        battery_critical: float = 0.2,
        use_activation_proxy: bool = False,
        train_mb_size: int = 128,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: str = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator=None,
        eval_every: int = -1,
        **base_kwargs,
    ):
        ea_plugin = EANPSPlugin(
            input_shape=input_shape,
            nps_threshold=nps_threshold,
            mem_size=mem_size,
            battery_critical=battery_critical,
            use_activation_proxy=use_activation_proxy,
        )
        all_plugins = list(plugins or []) + [ea_plugin]
        self.ea_plugin = ea_plugin

        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=all_plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs,
        )
