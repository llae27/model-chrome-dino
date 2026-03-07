#!/usr/bin/env python
"""
Strict robustness evaluation for DDQN (pixel-only). No exploration, no updates.
Deterministic: fixed seeds, fixed episodes, model.eval(), epsilon=0 (argmax).
"""
import argparse
import json
import os
import sys

import gymnasium as gym
import numpy as np
import torch

import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino, make_robustness_dino

# Assume run from bin/ or path includes bin
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_ddqn_fixed import QNet, obs_to_numpy, unwrap_reset, to_torch_obs

DEFAULT_EVAL_SEEDS = [100, 101, 102, 103, 104]
EPISODES_PER_SEED = 20
EARLY_TERMINATION_STEPS = 50  # heuristic: episode ended in very few steps


def make_eval_env(
    speed_multiplier=1.0,
    obstacle_frequency_multiplier=1.0,
    day_night_toggle=False,
):
    env = gym.make("ChromeDinoNoBrowser-v0")
    env = make_robustness_dino(
        env,
        timer=True,
        frame_stack=True,
        day_night_toggle=day_night_toggle,
        speed_multiplier=speed_multiplier if speed_multiplier > 1.0 else 1.0,
    )
    try:
        env.unwrapped.set_acceleration(True)
    except Exception:
        pass
    if obstacle_frequency_multiplier != 1.0:
        try:
            env.unwrapped.set_obstacle_frequency_multiplier(obstacle_frequency_multiplier)
        except Exception:
            pass
    return env


def run_episode(env, model, device, seed, max_steps=50000):
    """Run a single episode; return reward, steps, failure_mode, score."""
    reset_out = env.reset(seed=seed)
    obs = unwrap_reset(reset_out)
    obs = obs_to_numpy(obs)
    total_reward = 0.0
    steps = 0
    done = False
    failure_mode = None
    while not done and steps < max_steps:
        with torch.no_grad():
            action = int(
                torch.argmax(
                    model(to_torch_obs(obs[None], device)), dim=1
                ).item()
            )
        obs_raw, r, terminated, truncated, _ = env.step(action)
        obs = obs_to_numpy(obs_raw)
        total_reward += r
        steps += 1
        done = terminated or truncated
    if steps < EARLY_TERMINATION_STEPS and done:
        failure_mode = "early_termination"
    score = None
    try:
        score = int(env.unwrapped.get_score())
    except Exception:
        pass
    return float(total_reward), steps, failure_mode, score


def evaluate_batch(
    model,
    env,
    eval_seeds,
    episodes_per_seed,
    device="cpu",
    max_episode_steps=50000,
):
    """Evaluates the model. Returns list of dicts with seed, episode, reward, steps, failure_mode."""
    model.eval()
    results = []
    for seed in eval_seeds:
        for ep in range(episodes_per_seed):
            reward, steps, failure_mode, score = run_episode(
                env, model, device, seed, max_episode_steps
            )
            results.append({
                "seed": int(seed),
                "episode": ep,
                "reward": reward,
                "steps": int(steps),
                "failure_mode": failure_mode,
                "score": score,
            })
    return results


def main():
    parser = argparse.ArgumentParser(description="DDQN robustness evaluation (pixel-only)")
    parser.add_argument("checkpoint", nargs="?", default="ddqn_runs/seed_0/ckpt_step_300000.pt", help="Path to DDQN .pt checkpoint")
    parser.add_argument("--condition", choices=["baseline", "speed", "freq", "day_night"], default="baseline")
    parser.add_argument("--speed_multiplier", type=float, default=1.0)
    parser.add_argument("--obstacle_frequency_multiplier", type=float, default=1.0)
    parser.add_argument("--day_night", action="store_true", help="Enable day/night (pixel inversion)")
    parser.add_argument("--eval_seeds", type=str, default=None, help="Comma-separated seeds, e.g. 100,101,102,103,104")
    parser.add_argument("--episodes_per_seed", type=int, default=EPISODES_PER_SEED)
    parser.add_argument("--output", type=str, default=None, help="JSON or CSV output path")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dry_run", action="store_true", help="Run only 2 episodes total then exit")
    parser.add_argument("--max_episodes", type=int, default=None, help="Cap total episodes (for compute estimate)")
    args = parser.parse_args()

    eval_seeds = DEFAULT_EVAL_SEEDS
    if args.eval_seeds:
        eval_seeds = [int(s) for s in args.eval_seeds.split(",")]

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    # make_dino produces (80, 160, 4) with frame_stack=4
    H, W, C = 80, 160, 4

    q = QNet(in_channels=C, n_actions=2, input_hw=(H, W)).to(device)
    q.load_state_dict(ckpt["q_state_dict"])
    q.eval()

    day_night = args.day_night or args.condition == "day_night"
    speed = args.speed_multiplier if args.condition == "speed" else 1.0
    freq = args.obstacle_frequency_multiplier if args.condition == "freq" else 1.0

    env = make_eval_env(
        speed_multiplier=speed,
        obstacle_frequency_multiplier=freq,
        day_night_toggle=day_night,
    )

    total_eps = len(eval_seeds) * args.episodes_per_seed
    if args.max_episodes is not None:
        total_eps = min(total_eps, args.max_episodes)
    if args.dry_run:
        total_eps = min(2, total_eps)
        eval_seeds = eval_seeds[:1]
        args.episodes_per_seed = min(2, args.episodes_per_seed)

    results = []
    count = 0
    for seed in eval_seeds:
        for ep in range(args.episodes_per_seed):
            if count >= total_eps:
                break
            reward, steps, failure_mode, score = run_episode(env, q, device, seed, max_steps=50000)
            results.append({
                "seed": int(seed),
                "episode": ep,
                "reward": reward,
                "steps": int(steps),
                "failure_mode": failure_mode,
                "score": score,
            })
            count += 1
        if count >= total_eps:
            break

    env.close()

    rewards = [r["reward"] for r in results]
    scores = [r["score"] for r in results if r.get("score") is not None]
    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    mean_s = float(np.mean(scores)) if scores else None
    std_s = float(np.std(scores)) if scores else None
    print(f"condition={args.condition} speed={speed} freq={freq} day_night={day_night}")
    print(f"episodes={len(results)} mean_reward={mean_r:.2f} std_reward={std_r:.2f} min={min(rewards):.2f} max={max(rewards):.2f}")
    if scores:
        print(f"  mean_score={mean_s:.1f} std_score={std_s:.1f} min_score={min(scores)} max_score={max(scores)}")

    out = {
        "algo": "ddqn",
        "checkpoint": args.checkpoint,
        "condition": args.condition,
        "speed_multiplier": speed,
        "obstacle_frequency_multiplier": freq,
        "day_night_toggle": day_night,
        "eval_seeds": eval_seeds,
        "episodes_per_seed": args.episodes_per_seed,
        "n_episodes": len(results),
        "mean_reward": mean_r,
        "std_reward": std_r,
        "min_reward": float(min(rewards)),
        "max_reward": float(max(rewards)),
        "per_episode": results,
    }
    if mean_s is not None:
        out["mean_score"] = mean_s
        out["std_score"] = std_s
        out["min_score"] = int(min(scores))
        out["max_score"] = int(max(scores))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.output}")
    return out


if __name__ == "__main__":
    main()
