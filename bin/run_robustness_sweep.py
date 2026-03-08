#!/usr/bin/env python
"""
Run the full robustness sweep: DDQN and PPO over speed, freq, day_night conditions.
Writes one JSON per (algo, condition, value) to an output directory.
"""
import argparse
import json
import os
import subprocess
import sys

DEFAULT_EVAL_SEEDS = "100,101,102,103,104"
EPISODES_PER_SEED = 20

SPEED_VALUES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
FREQ_VALUES = [1.0, 1.1, 1.25, 1.5, 1.75]
DAY_NIGHT_VALUES = [False, True]  # baseline then inverted


def run_one(algo, condition, value_key, value, out_dir, ddqn_ckpt, ppo_ckpt, eval_seeds, episodes_per_seed, dry_run):
    if algo == "ddqn":
        script = "eval_robustness_ddqn.py"
        ckpt = ddqn_ckpt
    else:
        script = "eval_robustness_ppo.py"
        ckpt = ppo_ckpt
    fname = f"{algo}_{condition}_{value_key}.json"
    out_path = os.path.join(out_dir, fname)
    cmd = [
        sys.executable, script, ckpt,
        "--condition", condition,
        "--eval_seeds", eval_seeds,
        "--episodes_per_seed", str(episodes_per_seed),
        "--output", out_path,
    ]
    if condition == "speed":
        cmd += ["--speed_multiplier", str(value)]
    elif condition == "freq":
        cmd += ["--obstacle_frequency_multiplier", str(value)]
    elif condition == "day_night":
        if value:
            cmd += ["--day_night"]
    bin_dir = os.path.dirname(os.path.abspath(__file__))
    if dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, cwd=bin_dir, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="robustness_results", help="Directory for JSON outputs")
    parser.add_argument("--ddqn_ckpt", type=str, default="ddqn_runs_final/seed_0/ckpt_step_300000.pt")
    parser.add_argument("--ppo_ckpt", type=str, default="ppo_dino.pt")
    parser.add_argument("--eval_seeds", type=str, default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--episodes_per_seed", type=int, default=EPISODES_PER_SEED)
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")
    parser.add_argument("--algos", type=str, default="ddqn,ppo", help="Comma-separated: ddqn,ppo")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    algos = [a.strip() for a in args.algos.split(",")]

    for algo in algos:
        run_one(algo, "baseline", "baseline", 1.0, args.out_dir,
                args.ddqn_ckpt, args.ppo_ckpt, args.eval_seeds, args.episodes_per_seed, args.dry_run)
        for speed in SPEED_VALUES:
            run_one(algo, "speed", f"speed{speed}", speed, args.out_dir,
                    args.ddqn_ckpt, args.ppo_ckpt, args.eval_seeds, args.episodes_per_seed, args.dry_run)
        for freq in FREQ_VALUES:
            run_one(algo, "freq", f"freq{freq}", freq, args.out_dir,
                    args.ddqn_ckpt, args.ppo_ckpt, args.eval_seeds, args.episodes_per_seed, args.dry_run)
        for day_night in DAY_NIGHT_VALUES:
            run_one(algo, "day_night", "invert" if day_night else "normal", day_night, args.out_dir,
                    args.ddqn_ckpt, args.ppo_ckpt, args.eval_seeds, args.episodes_per_seed, args.dry_run)

    print(f"Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
