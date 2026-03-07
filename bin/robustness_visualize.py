#!/usr/bin/env python
"""
Visualizations for robustness report:
1. Performance vs shift magnitude (line): X = speed or freq multiplier, Y = mean reward.
2. Robustness drop % comparison (bar): grouped by algo (DDQN vs PPO).
"""
import argparse
import json
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_metrics(results_dir):
    """Load robustness_summary.json or robustness_metrics.csv from results_dir."""
    summary_path = os.path.join(results_dir, "robustness_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    metrics_path = os.path.join(results_dir, "robustness_metrics.csv")
    if not os.path.isfile(metrics_path):
        return None
    metrics = []
    with open(metrics_path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return None
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        algo, condition, value_key, mean_r, std_r, n_ep = parts[0], parts[1], parts[2], float(parts[3]), float(parts[4]), int(parts[5])
        drop = float(parts[6]) if len(parts) > 6 and parts[6] else None
        try:
            v = float(value_key)
        except ValueError:
            v = value_key
        metrics.append({
            "algo": algo,
            "condition": condition,
            "value_key": v,
            "mean_reward": mean_r,
            "std_reward": std_r,
            "drop_pct": drop,
        })
    return {"metrics": metrics, "baseline_means": {}}


def plot_performance_vs_shift(summary, out_path):
    """Line plot: X = shift magnitude (speed or freq), Y = mean reward, one line per algo."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return
    metrics = summary.get("metrics", [])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for cond, ax in [("speed", axes[0]), ("freq", axes[1])]:
        for algo in ["ddqn", "ppo"]:
            pts = [(m["value_key"], m["mean_reward"], m["std_reward"]) for m in metrics if m["algo"] == algo and m["condition"] == cond and isinstance(m["value_key"], (int, float))]
            if not pts:
                continue
            pts.sort(key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            errs = [p[2] for p in pts]
            ax.errorbar(xs, ys, yerr=errs, label=algo.upper(), capsize=2)
        ax.set_xlabel(f"{cond} multiplier")
        ax.set_ylabel("Mean reward")
        ax.set_title(f"Performance vs {cond} shift")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_drop_bars(summary, out_path):
    """Bar chart: robustness drop % by algo, grouped by condition/value."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return
    metrics = summary.get("metrics", [])
    seen = set()
    keys = []
    ddqn_drops = []
    ppo_drops = []
    for m in metrics:
        if m["condition"] == "baseline":
            continue
        key = (m["condition"], m["value_key"])
        if key in seen:
            continue
        seen.add(key)
        label = f"{m['condition']}_{m['value_key']}"
        keys.append(label)
        d = next((x["drop_pct"] for x in metrics if x["algo"] == "ddqn" and x["condition"] == m["condition"] and x["value_key"] == m["value_key"]), None)
        p = next((x["drop_pct"] for x in metrics if x["algo"] == "ppo" and x["condition"] == m["condition"] and x["value_key"] == m["value_key"]), None)
        ddqn_drops.append(float(d) if d is not None else 0.0)
        ppo_drops.append(float(p) if p is not None else 0.0)
    if not keys:
        print("No drop data for bar chart")
        return
    x = np.arange(len(keys))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.5), 5))
    ax.bar(x - w/2, ddqn_drops, w, label="DDQN")
    ax.bar(x + w/2, ppo_drops, w, label="PPO")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel("Performance drop %")
    ax.set_title("Robustness drop by algo")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="robustness_results")
    parser.add_argument("--out_line", type=str, default="robustness_performance_vs_shift.png")
    parser.add_argument("--out_bars", type=str, default="robustness_drop_bars.png")
    args = parser.parse_args()

    summary = load_metrics(args.results_dir)
    if summary is None:
        print("Run robustness_metrics.py first or point --results_dir to a dir with robustness_summary.json")
        return
    if HAS_MATPLOTLIB:
        plot_performance_vs_shift(summary, args.out_line)
        plot_drop_bars(summary, args.out_bars)
    else:
        print("Install matplotlib for visualizations.")


if __name__ == "__main__":
    main()
