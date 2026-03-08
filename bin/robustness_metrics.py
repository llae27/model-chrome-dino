#!/usr/bin/env python
"""
Post-process robustness JSONs: Mean ± Std, Performance Drop %, optional t-test.
Reads all JSONs from robustness_results/ (or given dir) and writes metrics CSV + summary.
"""
import argparse
import glob
import json
import os

import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_results(results_dir):
    """Load all *_*.json files; return list of dicts with algo, condition, value_key, mean, std, etc."""
    pattern = os.path.join(results_dir, "*.json")
    files = glob.glob(pattern)
    rows = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        algo = data.get("algo", "unknown")
        condition = data.get("condition", "unknown")
        mean_r = data.get("mean_reward")
        std_r = data.get("std_reward")
        n = data.get("n_episodes")
        per_episode = data.get("per_episode", [])
        rewards = [e["reward"] for e in per_episode]
        value_key = None
        if condition == "speed":
            value_key = data.get("speed_multiplier", "?")
        elif condition == "freq":
            value_key = data.get("obstacle_frequency_multiplier", "?")
        elif condition == "day_night":
            value_key = "invert" if data.get("day_night_toggle") else "normal"
        elif condition == "baseline":
            value_key = "baseline"
        fbase = os.path.basename(path).replace(".json", "")
        rows.append({
            "file": fbase,
            "algo": algo,
            "condition": condition,
            "value_key": value_key,
            "mean_reward": mean_r,
            "std_reward": std_r,
            "n_episodes": n,
            "rewards": rewards,
        })
    return rows


def baseline_mean_by_algo(rows):
    """Per-algo baseline mean (condition=baseline or speed=1.0)."""
    by_algo = {}
    for r in rows:
        if r["condition"] == "baseline" or (r["condition"] == "speed" and r["value_key"] == 1.0):
            by_algo[r["algo"]] = r["mean_reward"]
    for r in rows:
        if r["condition"] == "speed" and r["value_key"] == 1.0 and r["algo"] not in by_algo:
            by_algo[r["algo"]] = r["mean_reward"]
    return by_algo


def performance_drop_pct(baseline_mean, shifted_mean):
    denom = max(abs(baseline_mean), 1e-8)
    return (baseline_mean - shifted_mean) / denom * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="robustness_results")
    parser.add_argument("--output_csv", type=str, default=None, help="Default: results_dir/robustness_metrics.csv")
    parser.add_argument("--output_summary", type=str, default=None, help="Default: results_dir/robustness_summary.json")
    args = parser.parse_args()
    if args.output_csv is None:
        args.output_csv = os.path.join(args.results_dir, "robustness_metrics.csv")
    if args.output_summary is None:
        args.output_summary = os.path.join(args.results_dir, "robustness_summary.json")

    rows = load_results(args.results_dir)
    if not rows:
        print(f"No JSONs found in {args.results_dir}")
        return

    baseline_means = baseline_mean_by_algo(rows)
    if not baseline_means:
        for r in rows:
            if r["condition"] == "speed":
                baseline_means[r["algo"]] = next(
                    (x["mean_reward"] for x in rows if x["algo"] == r["algo"] and x["condition"] == "speed" and x["value_key"] == 1.0),
                    None,
                )
    for r in rows:
        if r["algo"] not in baseline_means and r["condition"] == "baseline":
            baseline_means[r["algo"]] = r["mean_reward"]

    # Build metrics table: algo, condition, value_key, mean, std, drop_pct
    metrics = []
    for r in rows:
        bl = baseline_means.get(r["algo"])
        drop = None
        if bl is not None and r["condition"] != "baseline":
            drop = performance_drop_pct(bl, r["mean_reward"])
        metrics.append({
            "algo": r["algo"],
            "condition": r["condition"],
            "value_key": r["value_key"],
            "mean_reward": r["mean_reward"],
            "std_reward": r["std_reward"],
            "n_episodes": r["n_episodes"],
            "drop_pct": drop,
        })

    # CSV
    with open(args.output_csv, "w") as f:
        f.write("algo,condition,value_key,mean_reward,std_reward,n_episodes,drop_pct\n")
        for m in metrics:
            drop_s = f"{m['drop_pct']:.2f}" if m["drop_pct"] is not None else ""
            f.write(f"{m['algo']},{m['condition']},{m['value_key']},{m['mean_reward']:.4f},{m['std_reward']:.4f},{m['n_episodes']},{drop_s}\n")
    print(f"Wrote {args.output_csv}")

    # Optional: t-test between DDQN and PPO per (condition, value_key)
    comparison = []
    if HAS_SCIPY:
        for cond in ["speed", "freq", "day_night"]:
            for v in set(m["value_key"] for m in metrics if m["condition"] == cond):
                ddqn = next((m for m in metrics if m["algo"] == "ddqn" and m["condition"] == cond and m["value_key"] == v), None)
                ppo = next((m for m in metrics if m["algo"] == "ppo" and m["condition"] == cond and m["value_key"] == v), None)
                if ddqn and ppo and ddqn.get("rewards") is None:
                    continue
                r_ddqn = next((x["rewards"] for x in rows if x["algo"] == "ddqn" and x["condition"] == cond and x["value_key"] == v), None)
                r_ppo = next((x["rewards"] for x in rows if x["algo"] == "ppo" and x["condition"] == cond and x["value_key"] == v), None)
                if r_ddqn and r_ppo and len(r_ddqn) > 1 and len(r_ppo) > 1:
                    t_stat, p_val = stats.ttest_ind(r_ddqn, r_ppo)
                    comparison.append({"condition": cond, "value_key": v, "t_stat": float(t_stat), "p_value": float(p_val)})
    else:
        comparison = []

    summary = {
        "baseline_means": baseline_means,
        "metrics": metrics,
        "comparison_ttest": comparison,
    }
    with open(args.output_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.output_summary}")
    if comparison:
        print("T-test (DDQN vs PPO):", comparison[:5], "..." if len(comparison) > 5 else "")


if __name__ == "__main__":
    main()
