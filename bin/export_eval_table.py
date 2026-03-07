#!/usr/bin/env python
"""
Build the evaluation table (Reward and Score over 100 runs) from baseline JSONs.
Reads robustness_results/ddqn_baseline_baseline.json and ppo_baseline_baseline.json.
No Random baseline. DQN row included only if dqn_baseline_baseline.json exists.
Output: CSV and printed table for pasting into the spreadsheet.
"""
import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Export evaluation table from baseline JSONs")
    parser.add_argument("--results_dir", type=str, default="robustness_results")
    parser.add_argument("--output", type=str, default=None, help="CSV path (default: print only)")
    args = parser.parse_args()

    # Map algo name in file -> display name in table (DQN optional)
    algo_files = [
        ("dqn", "DQN"),
        ("ddqn", "DDQN"),
        ("ppo", "PPO"),
    ]
    rows = []
    for file_algo, display_name in algo_files:
        fname = f"{file_algo}_baseline_baseline.json"
        path = os.path.join(args.results_dir, fname)
        if not os.path.isfile(path):
            if file_algo == "dqn":
                continue  # DQN optional
            continue
        with open(path) as f:
            data = json.load(f)
        mean_reward = data.get("mean_reward")
        mean_score = data.get("mean_score")
        n = data.get("n_episodes", 0)
        reward_str = f"{mean_reward:.2f}" if mean_reward is not None else "N/A"
        score_str = f"{mean_score:.0f}" if mean_score is not None else "N/A"
        rows.append((display_name, reward_str, score_str, n))
    if not rows:
        print(f"No baseline JSONs found in {args.results_dir}")
        return
    # Header
    lines = ["Evaluation,Reward (time),Score"]
    for display_name, reward_str, score_str, n in rows:
        lines.append(f"{display_name},{reward_str},{score_str}")
    table = "\n".join(lines)
    print("Evaluation table (average over 100 runs):")
    print(table)
    if args.output:
        with open(args.output, "w") as f:
            f.write(table)
        print(f"\nWrote {args.output}")

if __name__ == "__main__":
    main()
