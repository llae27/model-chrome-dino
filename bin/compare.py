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


def plot_comparison(feature_metrics, cnn_metrics, algo="dqn", output="comparison.png"):
    algo_upper = algo.upper()
    agents = []
    if feature_metrics:
        agents.append((f"Feature {algo_upper} (MLP)", feature_metrics, "tab:blue"))
    if cnn_metrics:
        agents.append((f"Pixel {algo_upper} (CNN)", cnn_metrics, "tab:orange"))

    if not agents:
        print("No metrics to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Feature vs Pixel {algo_upper} — Chrome Dino", fontsize=14, fontweight="bold")

    # Use steps as x-axis if available, otherwise fall back to episode index
    def get_x(m):
        if "steps" in m and len(m["steps"]) == len(m["rewards"]):
            return np.array(m["steps"]), "Environment Step"
        return np.arange(len(m["rewards"])), "Episode"

    # Learning curve
    ax = axes[0, 0]
    for name, m, color in agents:
        x, xlabel = get_x(m)
        rewards = m["rewards"]
        ax.plot(x, rewards, alpha=0.2, color=color)
        ax.plot(x, smooth(rewards), color=color, label=name, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Total Reward")
    ax.set_title("Learning Curve (Smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Game score
    ax = axes[0, 1]
    for name, m, color in agents:
        x, xlabel = get_x(m)
        scores = m["scores"]
        ax.plot(x, scores, alpha=0.2, color=color)
        ax.plot(x, smooth(scores), color=color, label=name, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Game Score")
    ax.set_title("Game Score Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode time
    ax = axes[1, 0]
    for name, m, color in agents:
        times = m.get("wall_times", m.get("episode_times", []))
        if times:
            x, xlabel = get_x(m)
            ax.plot(x[:len(times)], times, color=color, label=name, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Episode Duration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary table
    ax = axes[1, 1]
    ax.axis("off")
    headers = ["Metric"] + [name for name, _, _ in agents]
    rows = []
    for name, m, _ in agents:
        last_10_rewards = m["rewards"][-10:] if len(m["rewards"]) >= 10 else m["rewards"]
        last_10_scores = m["scores"][-10:] if len(m["scores"]) >= 10 else m["scores"]
        times = m.get("wall_times", m.get("episode_times", []))
        rows.append({
            "name": name,
            "avg_final_reward": f"{np.mean(last_10_rewards):.1f}",
            "avg_final_score": f"{np.mean(last_10_scores):.0f}",
            "max_score": f"{max(m['scores'])}",
            "total_time": f"{sum(times):.0f}s" if times else "N/A",
            "episodes": f"{len(m['rewards'])}",
        })

    table_data = [
        ["Avg Reward (last 10)"] + [r["avg_final_reward"] for r in rows],
        ["Avg Score (last 10)"] + [r["avg_final_score"] for r in rows],
        ["Max Score"] + [r["max_score"] for r in rows],
        ["Total Training Time"] + [r["total_time"] for r in rows],
        ["Episodes"] + [r["episodes"] for r in rows],
    ]

    table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax.set_title("Summary Statistics", pad=20)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare RL agents (feature vs pixel)")
    parser.add_argument("--algo", default="dqn", help="Algorithm name for titles (dqn/ppo/ddqn)")
    parser.add_argument("--feature-metrics", default="feature_dqn_metrics.json")
    parser.add_argument("--cnn-metrics", default="cnn_dqn_metrics.json")
    parser.add_argument("--output", default=None, help="Output filename (default: {algo}_comparison.png)")
    parser.add_argument("--feature-only", action="store_true")
    parser.add_argument("--cnn-only", action="store_true")
    args = parser.parse_args()

    output = args.output or f"{args.algo}_comparison.png"

    feature_m, cnn_m = None, None

    if not args.cnn_only and os.path.exists(args.feature_metrics):
        feature_m = load_metrics(args.feature_metrics)
        print(f"Loaded feature metrics: {len(feature_m['rewards'])} episodes")
    elif not args.cnn_only:
        print(f"Warning: {args.feature_metrics} not found, skipping feature agent")

    if not args.feature_only and os.path.exists(args.cnn_metrics):
        cnn_m = load_metrics(args.cnn_metrics)
        print(f"Loaded CNN metrics: {len(cnn_m['rewards'])} episodes")
    elif not args.feature_only:
        print(f"Warning: {args.cnn_metrics} not found, skipping CNN agent")

    plot_comparison(feature_m, cnn_m, algo=args.algo, output=output)


if __name__ == "__main__":
    main()
