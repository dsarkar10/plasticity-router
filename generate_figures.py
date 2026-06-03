import os, re, ast, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
VIP = "vip_res"
OUT = f"{VIP}/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 13, "figure.dpi": 200,
    "axes.labelsize": 14, "axes.titlesize": 15,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
})


C = {
    "naive":  "#6c6c6c",   # gray
    "ewc":    "#e07b39",   # orange
    "er":     "#3b7a9e",   # blue
    "derpp":  "#8b3a8b",   # purple
    "ea_nps": "#3a9e3a",   # green
}
M = {
    "naive":  "o",         # circle
    "ewc":    "s",         # square
    "er":     "v",         # triangle down
    "derpp":  "D",         # diamond
    "ea_nps": "*",         # star
}
ORDER = ["naive", "ewc", "er", "derpp", "ea_nps"]
STRAT_LABELS = {
    "naive": "Naive", "ewc": "EWC", "er": "ER",
    "derpp": "DER++", "ea_nps": "EA-NPS",
}


def parse_list(s):
    if isinstance(s, str) and s.startswith("["):
        try:
            return ast.literal_eval(re.sub(r'np\.float64\(([^)]+)\)', r'\1', s))
        except Exception:
            return []
    return s if isinstance(s, list) else []


pm = pd.read_csv(f"{VIP}/permuted_mnist_multiseed.csv")
pm["task_accuracies"] = pm["task_accuracies"].apply(parse_list)
pm["acc_matrix"] = pm["acc_matrix"].apply(parse_list)

summary = pm.groupby("strategy").agg(
    acc_mean=("final_accuracy", "mean"),
    acc_std=("final_accuracy", "std"),
    time_mean=("time_seconds", "mean"),
    time_std=("time_seconds", "std"),
)

# Battery data
bf = pd.read_csv(f"{VIP}/battery_fast.csv")
bfull = pd.read_csv(f"{VIP}/battery_full.csv")
pb = pd.read_csv(f"{VIP}/permuted_battery.csv")

# Ablation data
abl = pd.read_csv(f"{VIP}/ablation_fast.csv")


def fig_pareto_frontier():
    fig, ax = plt.subplots(figsize=(8, 6))

    for strat in ORDER:
        sub = pm[pm.strategy == strat]
        mu_acc = summary.loc[strat, "acc_mean"]
        sd_acc = summary.loc[strat, "acc_std"]
        mu_t = summary.loc[strat, "time_mean"]
        sd_t = summary.loc[strat, "time_std"]

        ax.errorbar(mu_t, mu_acc, xerr=sd_t, yerr=sd_acc,
                    fmt=M[strat], color=C[strat], capsize=4, capthick=1.5,
                    markersize=11, elinewidth=1.5, label=STRAT_LABELS[strat],
                    markeredgecolor="black", markeredgewidth=0.5)

        for _, r in sub.iterrows():
            ax.scatter(r["time_seconds"], r["final_accuracy"],
                       marker=M[strat], color=C[strat], s=40, alpha=0.3,
                       edgecolors="black", linewidth=0.3, zorder=2)

    frontier_x = [summary.loc["naive", "time_mean"],
                  summary.loc["ea_nps", "time_mean"],
                  summary.loc["derpp", "time_mean"]]
    frontier_y = [summary.loc["naive", "acc_mean"],
                  summary.loc["ea_nps", "acc_mean"],
                  summary.loc["derpp", "acc_mean"]]
    ax.plot(frontier_x, frontier_y, "--", color="black", alpha=0.25, linewidth=1.5, label="Pareto frontier")

    ax.set_xlabel("Wall Time (seconds)")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("PermutedMNIST — Accuracy vs. Energy Cost")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(140, 420)
    ax.set_ylim(0.65, 1.0)

    ea_t = summary.loc["ea_nps", "time_mean"]
    er_t = summary.loc["er", "time_mean"]
    ax.annotate(f"EA-NPS: {ea_t:.0f}s\nER: {er_t:.0f}s\n{((er_t-ea_t)/er_t*100):.0f}% faster",
                xy=(ea_t, summary.loc["ea_nps", "acc_mean"]),
                xytext=(ea_t - 55, summary.loc["ea_nps", "acc_mean"] - 0.04),
                arrowprops=dict(arrowstyle="->", color=C["ea_nps"], lw=1.5),
                fontsize=10, color=C["ea_nps"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C["ea_nps"], alpha=0.8))

    fig.tight_layout()
    fig.savefig(f"{OUT}/pareto_frontier.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ pareto_frontier.png")


def fig_learning_curves():
    n_tasks = 5
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    for strat in ORDER:
        sub = pm[pm.strategy == strat]
        all_trajs = np.array([r["task_accuracies"] for _, r in sub.iterrows()])
        mu = all_trajs.mean(axis=0)
        sd = all_trajs.std(axis=0)
        ax.plot(range(1, n_tasks + 1), mu, color=C[strat], marker=M[strat],
                markersize=7, label=STRAT_LABELS[strat], linewidth=2)
        ax.fill_between(range(1, n_tasks + 1), mu - sd, mu + sd,
                        color=C[strat], alpha=0.12)

    ax.set_xlabel("Task")
    ax.set_ylabel("Average Accuracy (all seen tasks)")
    ax.set_title("Learning Curves", fontweight="bold")
    ax.set_xticks(range(1, n_tasks + 1))
    ax.set_ylim(0.55, 1.0)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--")

    ax = axes[1]
    n_forget = n_tasks - 1
    for strat in ORDER:
        sub = pm[pm.strategy == strat]
        all_forget = []
        for _, r in sub.iterrows():
            mat = r["acc_matrix"]
            forget = [mat[i][i] - mat[-1][i] for i in range(n_forget)]
            all_forget.append(forget)
        all_forget = np.array(all_forget)
        mu_f = all_forget.mean(axis=0)
        sd_f = all_forget.std(axis=0)
        ax.bar(np.arange(n_forget) + ORDER.index(strat) * 0.15 - 0.3,
               mu_f, width=0.12, color=C[strat], label=STRAT_LABELS[strat],
               edgecolor="white", yerr=sd_f, capsize=2)

    ax.set_xlabel("Task")
    ax.set_ylabel("Forgetting")
    ax.set_title("Per-Task Forgetting", fontweight="bold")
    ax.set_xticks(range(n_forget))
    ax.set_xticklabels([f"Task {i+1}" for i in range(n_forget)])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--", axis="y")

    fig.tight_layout()
    fig.savefig(f"{OUT}/learning_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ learning_curves.png")


def fig_battery_routes():
    route_cmap = {"sgd": "#3b7a9e", "er": "#e07b39", "freeze": "#8b3a8b"}
    op_label = {"sgd": "SGD", "er": "ER", "freeze": "FRZ"}
    scenarios = [
        ("SplitMNIST — Full Battery  (Δ=0.05)",   bfull),
        ("SplitMNIST — Fast Decay    (Δ=0.25)",   bf),
        ("PermutedMNIST — Fast Decay (Δ=0.25)",   pb),
    ]

    slot_w = 1.6       # width per task slot (box + gap)
    box_w = 0.8        # box width
    gap = slot_w - box_w  # gap between boxes
    box_h = 0.7
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 5.5))

    for idx, (title, df) in enumerate(scenarios):
        ax = axes[idx]
        route_str = str(df.iloc[0].get("routes", ""))
        route = [r.strip() for r in route_str.split("→")]
        n_tasks = len(route)
        xmax = n_tasks * slot_w

        ax.set_xlim(-0.3, xmax + 2.0)
        ax.set_ylim(-0.1, 1.2)
        ax.set_aspect("auto")
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

        for t, op in enumerate(route):
            color = route_cmap.get(op, "#cccccc")
            label = op_label.get(op, op.upper())

            bx = t * slot_w + gap / 2
            by = 0.2
            rect = mpatches.FancyBboxPatch((bx, by), box_w, box_h,
                                            boxstyle="round,pad=0.12",
                                            facecolor=color, edgecolor="white",
                                            linewidth=2.5)
            ax.add_patch(rect)
            ax.text(bx + box_w / 2, by + box_h / 2, label, ha="center", va="center",
                    fontsize=14, fontweight="bold", color="white")

            ax.text(bx + box_w / 2, 0.05, f"Task {t+1}", ha="center", va="top",
                    fontsize=10, color="#555555", fontstyle="italic")

            if t < n_tasks - 1:
                x1 = bx + box_w
                x2 = (t + 1) * slot_w + gap / 2
                ax.annotate("", xy=(x2, 0.55), xytext=(x1, 0.55),
                            arrowprops=dict(arrowstyle="->", color="#555555",
                                            lw=3, mutation_scale=25))

    acc = df.iloc[0]["final_accuracy"]
    macs = df.iloc[0].get("macs_saved_pct", "N/A")
    ax.text(xmax + 0.3, 0.55,
                f"{title}\nAcc={acc:.4f}\nMACs={macs}%",
                ha="left", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0",
                          edgecolor="#cccccc", alpha=0.8))

    fig.suptitle("EA-NPS Routes under Different Battery Scenarios",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(pad=1.5)
    fig.savefig(f"{OUT}/battery_routes.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ battery_routes.png")



def _jagged_to_square(jagged):
    """Convert jagged acc_matrix list-of-lists to n×n ndarray (NaN for unseen)."""
    n = len(jagged)
    mat = np.full((n, n), np.nan)
    for i, row in enumerate(jagged):
        for j, val in enumerate(row):
            mat[i, j] = val
    return mat

def fig_per_task_accuracy():
    strategies_to_plot = ["naive", "er", "ea_nps"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, strat in enumerate(strategies_to_plot):
        ax = axes[idx]
        sub = pm[pm.strategy == strat]
        jagged = sub.iloc[0]["acc_matrix"]
        mat = _jagged_to_square(jagged)
        n = mat.shape[0]

        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad("white", alpha=0.0)
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white" if val < 0.5 else "black")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"T{t+1}" for t in range(n)])
        ax.set_yticklabels([f"T{t+1}" for t in range(n)])
        ax.set_xlabel("Test Task")
        ax.set_ylabel("Trained Until")
        ax.set_title(STRAT_LABELS[strat], fontweight="bold")

    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.025])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Accuracy")
    fig.suptitle("Per-Task Accuracy Matrices (seed=42)", fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(f"{OUT}/per_task_accuracy.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ per_task_accuracy.png")



def fig_accuracy_matrix_forgetting():
    sub = pm[pm.strategy == "ea_nps"]
    jagged = sub.iloc[0]["acc_matrix"]
    mat = _jagged_to_square(jagged)
    n = mat.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("white", alpha=0.0)
    im = ax.imshow(mat, cmap=cmap, vmin=0.5, vmax=1.0, aspect="equal")
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if val < 0.7 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"T{t+1}" for t in range(n)])
    ax.set_yticklabels([f"T{t+1}" for t in range(n)])
    ax.set_xlabel("Test Task", fontweight="bold")
    ax.set_ylabel("Trained Until", fontweight="bold")
    ax.set_title("EA-NPS Accuracy Matrix", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Accuracy")

    ax = axes[1]
    n_forget = n - 1
    forget = [mat[i][i] - mat[-1][i] for i in range(n_forget)]
    bars = ax.bar(range(n_forget), forget, color="#d62d2d", edgecolor="white", width=0.6)
    for bar, v in zip(bars, forget):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("Task", fontweight="bold")
    ax.set_ylabel("Forgetting", fontweight="bold")
    ax.set_title("Per-Task Forgetting (EA-NPS)", fontweight="bold")
    ax.set_xticks(range(n_forget))
    ax.set_xticklabels([f"Task {t+1}" for t in range(n_forget)])
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")
    ax.set_ylim(0, max(forget) * 1.3 + 0.02)

    fig.tight_layout()
    fig.savefig(f"{OUT}/accuracy_matrix_forgetting.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ accuracy_matrix_forgetting.png")



def fig_layerwise_heatmap():
    import torch, torch.nn as nn
    from collections import OrderedDict
    import sys; sys.path.insert(0, os.getcwd())
    from ea_nps_strategy import NPSComputer

    torch.manual_seed(42)
    np.random.seed(42)

    model = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(28*28, 256)), ("relu1", nn.ReLU()),
        ("fc2", nn.Linear(256, 128)), ("relu2", nn.ReLU()),
        ("fc3", nn.Linear(128, 10)),
    ]))

    all_buffer = []
    layer_nps_by_task = []
    task_names = ["Task 0", "Task 1", "Task 2", "Task 3", "Task 4"]
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    layers = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]

    for task in range(5):
        cls_start = task * 2
        cls_end = cls_start + 2
        task_data, task_labels = [], []
        for c in range(cls_start, cls_end):
            for _ in range(80):
                img = torch.zeros(28*28)
                img[c*28:(c+1)*28] = torch.randn(28) * 0.3 + 1.0
                task_data.append(img); task_labels.append(c)

        X_task = torch.stack(task_data)
        y_task = torch.tensor(task_labels)

        model.train()
        for _ in range(3):
            optim.zero_grad()
            crit(model(X_task), y_task).backward()
            optim.step()

        for i in range(len(X_task)):
            all_buffer.append((X_task[i].cpu(), int(y_task[i].item()), task))

        if len(all_buffer) >= 10:
            idx = np.random.choice(len(all_buffer), min(64, len(all_buffer)), replace=False)
            X_old = torch.stack([all_buffer[i][0] for i in idx])
            computer = NPSComputer(model, all_buffer, "cpu")
            gn = computer.compute_layerwise_nps(X_task[:32], y_task[:32])
            layer_nps_by_task.append(gn)
        else:
            layer_nps_by_task.append({k: 0.0 for k in layers})

    heatmap = np.zeros((len(layers), len(layer_nps_by_task)))
    for t, nps_dict in enumerate(layer_nps_by_task):
        for li, layer in enumerate(layers):
            heatmap[li, t] = nps_dict.get(layer, 0.0)

    model.eval()
    fisher = {}
    for name, p in model.named_parameters():
        fisher[name] = torch.zeros_like(p)
    for _ in range(50):
        idx = np.random.choice(len(all_buffer), min(64, len(all_buffer)), replace=False)
        xb = torch.stack([all_buffer[i][0] for i in idx])
        yb = torch.tensor([all_buffer[i][1] for i in idx])
        model.zero_grad()
        crit(model(xb), yb).backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad ** 2
    for name in fisher:
        fisher[name] /= 50
    fisher_by_layer = {}
    for name, f in fisher.items():
        mod = name.rsplit(".", 1)[0]
        fisher_by_layer[mod] = fisher_by_layer.get(mod, 0.0) + f.mean().item()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    im1 = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=10)
    ax.set_xticks(range(len(layer_nps_by_task)))
    ax.set_xticklabels(task_names, fontsize=9)
    ax.set_xlabel("Training Task", fontweight="bold")
    ax.set_ylabel("Layer", fontweight="bold")
    ax.set_title("EA-NPS: Per-Layer Gradient Conflict (NPS)", fontweight="bold")
    for i in range(len(layers)):
        for j in range(len(task_names)):
            v = heatmap[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if v > 0.5 else "black")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label="NPS (conflict)")

    ax = axes[1]
    layer_names = ["fc1", "fc2", "fc3"]
    fisher_vals = [fisher_by_layer.get(n, 0.0) for n in layer_names]
    f_max = max(fisher_vals) or 1.0
    fisher_norm = [v / f_max for v in fisher_vals]
    bars = ax.barh(layer_names, fisher_norm, color="#e07b39", edgecolor="white", height=0.6)
    ax.set_xlabel("Normalized Fisher Importance", fontweight="bold")
    ax.set_title("EWC: Per-Module Parameter Importance", fontweight="bold")
    ax.grid(True, alpha=0.25, axis="x", linestyle="--")
    ax.set_xlim(0, 1.15)
    for bar, v in zip(bars, fisher_norm):
        ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=11, fontweight="bold")

    fig.tight_layout(pad=2)
    fig.savefig(f"{OUT}/layerwise_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ layerwise_heatmap.png")



if __name__ == "__main__":
    print("Generating all figures from vip_res/ data...\n")

    fig_pareto_frontier()
    fig_learning_curves()
    fig_battery_routes()
    fig_per_task_accuracy()
    fig_accuracy_matrix_forgetting()
    fig_layerwise_heatmap()

    print(f"\nAll figures saved to {OUT}/")
    print("Note: proxy_validation.png is generated by validate_proxy.py")
