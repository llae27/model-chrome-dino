#!/usr/bin/env python
"""
Strict robustness evaluation for PPO (pixel-only). No exploration, no updates.
Deterministic: fixed seeds, fixed episodes, model.eval(), argmax action selection.
"""
import argparse
import json
import os
import sys

import gymnasium as gym
import numpy as np
import torch

import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_robustness_dino

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_ppo import ActorCritic, fix_obs

DEFAULT_EVAL_SEEDS = [100, 101, 102, 103, 104]
EPISODES_PER_SEED = 20
EARLY_TERMINATION_STEPS = 50


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
    obs, _ = env.reset(seed=seed)
    obs = fix_obs(obs)
    total_reward = 0.0
    steps = 0
    done = False
    failure_mode = None
    while not done and steps < max_steps:
        obs_t = torch.from_numpy(obs).to(device)
        with torch.no_grad():
            logits, _ = model.forward(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())
        obs_raw, r, terminated, truncated, _ = env.step(action)
        obs = fix_obs(obs_raw)
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


def main():
    parser = argparse.ArgumentParser(description="PPO robustness evaluation (pixel-only)")
    parser.add_argument("checkpoint", nargs="?", default="ppo_dino.pt", help="Path to PPO .pt (state_dict)")
    parser.add_argument("--condition", choices=["baseline", "speed", "freq", "day_night"], default="baseline")
    parser.add_argument("--speed_multiplier", type=float, default=1.0)
    parser.add_argument("--obstacle_frequency_multiplier", type=float, default=1.0)
    parser.add_argument("--day_night", action="store_true", help="Enable day/night (pixel inversion)")
    parser.add_argument("--eval_seeds", type=str, default=None, help="Comma-separated seeds")
    parser.add_argument("--episodes_per_seed", type=int, default=EPISODES_PER_SEED)
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dry_run", action="store_true", help="Run only 2 episodes then exit")
    parser.add_argument("--max_episodes", type=int, default=None, help="Cap total episodes")
    args = parser.parse_args()

    eval_seeds = DEFAULT_EVAL_SEEDS
    if args.eval_seeds:
        eval_seeds = [int(s) for s in args.eval_seeds.split(",")]

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device)
    model = ActorCritic(n_actions=2).to(device)
    model.load_state_dict(state)
    model.eval()

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
            reward, steps, failure_mode, score = run_episode(env, model, device, seed, max_steps=50000)
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
        "algo": "ppo",
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
