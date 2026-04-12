"""
utils/compare_results.py — Transformer vs LSTM comparison plots.

Loads the four result JSON files produced by the training scripts
and generates publication-ready comparison figures.

Run after all four training scripts have completed:
    python run_fedavg.py
    python run_fedvoc.py
    python run_fedavg_lstm.py
    python run_fedvoc_lstm.py
    python utils/compare_results.py

Output:
    results/compare_convergence.png   — loss curves, all four models
    results/compare_perplexity.png    — per-client perplexity bar chart
    results/compare_domain_acc.png    — domain accuracy (FedVoc transformer vs LSTM)
    results/compare_comm_cost.png     — communication cost bar chart
    results/compare_summary.txt       — plain-text summary table
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
DOMAIN_NAMES = ["Shakespeare", "News", "Medical"]
COLORS = {
    "FedAvg\n(Transformer)": "#4C72B0",
    "FedVoc\n(Transformer)": "#DD8452",
    "FedAvg\n(LSTM)":        "#55A868",
    "FedVoc\n(LSTM)":        "#C44E52",
}


def load_json(path):
    if not os.path.exists(path):
        print(f"  [WARN] Missing: {path} — skipping.")
        return None
    with open(path) as f:
        return json.load(f)


def load_all():
    data = {
        "FedAvg\n(Transformer)": load_json(f"{RESULTS_DIR}/fedavg_results.json"),
        "FedVoc\n(Transformer)": load_json(f"{RESULTS_DIR}/fedvoc_results.json"),
        "FedAvg\n(LSTM)":        load_json(f"{RESULTS_DIR}/fedavg_lstm_results.json"),
        "FedVoc\n(LSTM)":        load_json(f"{RESULTS_DIR}/fedvoc_lstm_results.json"),
    }
    return data


# ── Plot helpers ──────────────────────────────────────────────────────────────

def style_ax(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if legend:
        ax.legend(fontsize=9, framealpha=0.7)


# ── Figure 1: Convergence curves ──────────────────────────────────────────────

def plot_convergence(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for label, results in data.items():
        if results is None or "round_losses" not in results:
            continue
        losses = results["round_losses"]
        rounds = list(range(len(losses)))
        group  = "FedAvg" if "FedAvg" in label else "FedVoc"
        ax     = axes[0] if "FedAvg" in label else axes[1]
        ax.plot(rounds, losses, label=label.replace("\n", " "),
                color=COLORS[label], linewidth=2, marker="o", markersize=3)

    style_ax(axes[0], "FedAvg: Transformer vs LSTM", "Round", "Avg Train Loss")
    style_ax(axes[1], "FedVoc: Transformer vs LSTM", "Round", "Avg Train Loss")

    plt.tight_layout()
    out = f"{RESULTS_DIR}/compare_convergence.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 2: Perplexity bar chart ────────────────────────────────────────────

def plot_perplexity(data):
    fig, ax = plt.subplots(figsize=(10, 5))

    n_clients = 3
    n_models  = sum(1 for v in data.values() if v and "eval" in v)
    width     = 0.18
    x         = np.arange(n_clients)
    offset    = 0

    for label, results in data.items():
        if results is None or "eval" not in results:
            continue
        ppls = [e["perplexity"] for e in results["eval"]]
        ax.bar(x + offset * width, ppls, width,
               label=label.replace("\n", " "),
               color=COLORS[label], alpha=0.85, edgecolor="white")
        offset += 1

    ax.set_xticks(x + (n_models - 1) * width / 2)
    ax.set_xticklabels(DOMAIN_NAMES, fontsize=11)
    style_ax(ax, "Test Perplexity by Domain and Model",
             "Domain", "Perplexity (lower = better)")

    out = f"{RESULTS_DIR}/compare_perplexity.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 3: Domain accuracy ─────────────────────────────────────────────────

def plot_domain_accuracy(data):
    """Only FedVoc variants report domain accuracy."""
    fedvoc_tf   = data.get("FedVoc\n(Transformer)")
    fedvoc_lstm = data.get("FedVoc\n(LSTM)")

    if not fedvoc_tf or not fedvoc_lstm:
        print("  [SKIP] Domain accuracy plot — missing FedVoc results.")
        return

    tf_acc   = fedvoc_tf.get("domain_acc",   [0, 0, 0])
    lstm_acc = fedvoc_lstm.get("domain_acc", [0, 0, 0])

    x     = np.arange(len(DOMAIN_NAMES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - width / 2, tf_acc,   width, label="FedVoc (Transformer)",
           color=COLORS["FedVoc\n(Transformer)"], alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, lstm_acc, width, label="FedVoc (LSTM)",
           color=COLORS["FedVoc\n(LSTM)"],        alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(DOMAIN_NAMES, fontsize=11)
    ax.set_ylim(0, 1)
    style_ax(ax, "Domain Accuracy: FedVoc Transformer vs LSTM",
             "Domain", "Top-3 Domain Accuracy (higher = better)")

    out = f"{RESULTS_DIR}/compare_domain_acc.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 4: Communication cost ─────────────────────────────────────────────

def plot_comm_cost(data):
    labels = []
    costs  = []
    colors = []

    for label, results in data.items():
        if results is None or "comm_cost" not in results:
            continue
        labels.append(label.replace("\n", " "))
        costs.append(results["comm_cost"])
        colors.append(COLORS[label])

    if not costs:
        print("  [SKIP] Communication cost plot — no data.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, costs, color=colors, alpha=0.85, edgecolor="white")

    # Annotate bars with human-readable param counts
    for bar, cost in zip(bars, costs):
        label_txt = f"{cost / 1e6:.1f}M" if cost >= 1e6 else f"{cost / 1e3:.0f}K"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02, label_txt,
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x / 1e6:.0f}M" if x >= 1e6 else f"{x / 1e3:.0f}K"
    ))
    style_ax(ax, "Communication Cost per Round (params uploaded to server)",
             "Model", "Parameters", legend=False)

    out = f"{RESULTS_DIR}/compare_comm_cost.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(data):
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append(f"{'Model':<28} {'Avg PPL':>9} {'Domain Acc':>12} {'Comm (params)':>15}")
    lines.append("-" * 70)

    for label, results in data.items():
        if results is None:
            continue
        name = label.replace("\n", " ")

        if "eval" in results:
            ppls    = [e["perplexity"] for e in results["eval"]]
            avg_ppl = sum(ppls) / len(ppls)
        else:
            avg_ppl = float("nan")

        domain_acc = results.get("domain_acc")
        acc_str    = f"{sum(domain_acc) / len(domain_acc):.3f}" if domain_acc else "  n/a"

        comm = results.get("comm_cost", 0)
        comm_str = f"{comm / 1e6:.2f}M" if comm >= 1e6 else f"{comm / 1e3:.1f}K"

        lines.append(f"{name:<28} {avg_ppl:>9.2f} {acc_str:>12} {comm_str:>15}")

    lines.append("=" * 70)
    summary = "\n".join(lines)
    print(summary)

    out = f"{RESULTS_DIR}/compare_summary.txt"
    with open(out, "w") as f:
        f.write(summary + "\n")
    print(f"\n  Summary saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\nLoading result files...")
    data = load_all()

    available = [k for k, v in data.items() if v is not None]
    if not available:
        print("No result files found. Run the training scripts first.")
        sys.exit(1)

    print(f"Found results for: {', '.join(k.replace(chr(10), ' ') for k in available)}\n")

    print("Generating comparison plots...")
    plot_convergence(data)
    plot_perplexity(data)
    plot_domain_accuracy(data)
    plot_comm_cost(data)
    print_summary(data)

    print("\nAll comparison plots saved to results/")


if __name__ == "__main__":
    main()
