import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def smooth(values, window=10):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    pad = np.full(window - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


def plot_comparison(feature_metrics=None, cnn_metrics=None, output_dir="."):
    agents = []
    if feature_metrics:
        agents.append(("Feature DQN (MLP)", feature_metrics, "tab:blue"))
    if cnn_metrics:
        agents.append(("Pixel DQN (CNN)", cnn_metrics, "tab:orange"))

    if not agents:
        print("No metrics to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Feature-Based DQN vs Pixel-Based DQN — Chrome Dino", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    for name, m, color in agents:
        rewards = m["rewards"]
        episodes = range(len(rewards))
        ax.plot(episodes, rewards, alpha=0.2, color=color)
        ax.plot(episodes, smooth(rewards), color=color, label=name, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Learning Curve (Smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for name, m, color in agents:
        scores = m["scores"]
        episodes = range(len(scores))
        ax.plot(episodes, scores, alpha=0.2, color=color)
        ax.plot(episodes, smooth(scores), color=color, label=name, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Game Score")
    ax.set_title("Game Score Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for name, m, color in agents:
        times = m["episode_times"]
        ax.plot(range(len(times)), times, color=color, label=name, alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Training Time per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    headers = ["Metric"] + [name for name, _, _ in agents]
    rows = []
    for name, m, _ in agents:
        last_10_rewards = m["rewards"][-10:] if len(m["rewards"]) >= 10 else m["rewards"]
        last_10_scores = m["scores"][-10:] if len(m["scores"]) >= 10 else m["scores"]
        rows.append({
            "name": name,
            "avg_final_reward": f"{np.mean(last_10_rewards):.1f}",
            "avg_final_score": f"{np.mean(last_10_scores):.0f}",
            "max_score": f"{max(m['scores'])}",
            "total_time": f"{sum(m['episode_times']):.0f}s",
            "episodes": f"{len(m['rewards'])}",
        })

    table_data = [
        ["Avg Reward (last 10)"] + [r["avg_final_reward"] for r in rows],
        ["Avg Score (last 10)"] + [r["avg_final_score"] for r in rows],
        ["Max Score"] + [r["max_score"] for r in rows],
        ["Total Training Time"] + [r["total_time"] for r in rows],
        ["Episodes"] + [r["episodes"] for r in rows],
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax.set_title("Summary Statistics", pad=20)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "dqn_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare DQN agents")
    parser.add_argument("--feature-metrics", default="feature_dqn_metrics.json")
    parser.add_argument("--cnn-metrics", default="cnn_dqn_metrics.json")
    parser.add_argument("--feature-only", action="store_true")
    parser.add_argument("--cnn-only", action="store_true")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    feature_m, cnn_m = None, None

    if not args.cnn_only and os.path.exists(args.feature_metrics):
        feature_m = load_metrics(args.feature_metrics)
        print(f"Loaded feature DQN metrics: {len(feature_m['rewards'])} episodes")
    elif not args.cnn_only:
        print(f"Warning: {args.feature_metrics} not found, skipping feature DQN")

    if not args.feature_only and os.path.exists(args.cnn_metrics):
        cnn_m = load_metrics(args.cnn_metrics)
        print(f"Loaded CNN DQN metrics: {len(cnn_m['rewards'])} episodes")
    elif not args.feature_only:
        print(f"Warning: {args.cnn_metrics} not found, skipping CNN DQN")

    plot_comparison(feature_m, cnn_m, args.output_dir)


if __name__ == "__main__":
    main()
