# Feature-based DDQN for Chrome Dino
# MLP Q-network using 8-dim engineered features instead of pixels.

import argparse
import json
import os
import time
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

import gym_chrome_dino
from feature_wrapper import FeatureObservationWrapper
from selenium.common.exceptions import (
    InvalidSessionIdException, WebDriverException, NoSuchWindowException,
)


def default_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Config:
    env_id: str = "ChromeDinoNoBrowser-v0"
    seed: int = 0
    device: str = default_device()

    total_steps: int = 250_000
    learning_starts: int = 10_000
    train_every: int = 4

    gamma: float = 0.99
    batch_size: int = 32
    replay_size: int = 100_000

    lr: float = 1e-4
    grad_clip: float = 10.0

    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_steps: int = 200_000

    target_update_every: int = 10_000

    log_every: int = 2_000
    save_every: int = 50_000
    save_dir: str = "ddqn_runs/feature"
    metrics_path: str = "ddqn_feature_metrics.json"

    acceleration: bool = True


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.ns = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, s, a, r, ns, d):
        self.s[self.idx] = s
        self.ns[self.idx] = ns
        self.a[self.idx] = a
        self.r[self.idx] = r
        self.d[self.idx] = d
        self.idx += 1
        if self.idx >= self.capacity:
            self.idx = 0
            self.full = True

    def sample(self, batch_size: int):
        n = len(self)
        idxs = np.random.randint(0, n, size=batch_size)
        return self.s[idxs], self.a[idxs], self.r[idxs], self.ns[idxs], self.d[idxs]


class FeatureQNet(nn.Module):
    def __init__(self, obs_dim: int = 8, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear_epsilon(step: int, cfg: Config) -> float:
    if step >= cfg.eps_decay_steps:
        return cfg.eps_end
    frac = step / float(cfg.eps_decay_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def make_env(cfg: Config, retries: int = 5, wait: float = 5.0):
    for attempt in range(retries):
        try:
            env = gym.make(cfg.env_id)
            env = FeatureObservationWrapper(env)
            env.unwrapped.set_acceleration(cfg.acceleration)
            return env
        except Exception as e:
            if attempt < retries - 1:
                print(f"[WARN] make_env attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def save_ckpt(path, step, q, tgt, opt, cfg, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "q_state_dict": q.state_dict(),
        "tgt_state_dict": tgt.state_dict(),
        "opt_state_dict": opt.state_dict(),
        "cfg": cfg.__dict__,
        "metrics": metrics,
    }, path)


def train_ddqn(cfg: Config, resume_from: str = None):
    set_seed(cfg.seed)
    device = cfg.device
    print(f"Using device: {device}")

    env = make_env(cfg)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f"obs_dim={obs_dim}, n_actions={n_actions}")

    q = FeatureQNet(obs_dim=obs_dim, n_actions=n_actions).to(device)
    tgt = FeatureQNet(obs_dim=obs_dim, n_actions=n_actions).to(device)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    rb = ReplayBuffer(cfg.replay_size, obs_dim=obs_dim)

    metrics = {"rewards": [], "scores": [], "steps": [], "wall_times": [],
               "algorithm": "ddqn", "representation": "feature",
               "total_steps": cfg.total_steps,
               "hyperparams": {"lr": cfg.lr, "gamma": cfg.gamma, "batch_size": cfg.batch_size,
                               "replay_size": cfg.replay_size, "eps_decay_steps": cfg.eps_decay_steps}}

    start_step = 1
    if resume_from is not None:
        print(f"Loading checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        q.load_state_dict(ckpt["q_state_dict"])
        tgt.load_state_dict(ckpt["tgt_state_dict"])
        opt.load_state_dict(ckpt["opt_state_dict"])
        start_step = ckpt["step"] + 1
        if "metrics" in ckpt:
            metrics = ckpt["metrics"]
        print(f"Resuming from step {start_step}, eps={linear_epsilon(start_step, cfg):.3f}")

    obs, _ = env.reset()
    ep_reward = 0.0
    ep_start_time = time.time()
    episodes = 0

    t0 = time.time()
    last_log = start_step - 1

    for step in range(start_step, cfg.total_steps + 1):
        eps = linear_epsilon(step, cfg)

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                action = int(torch.argmax(q(obs_t), dim=1).item())

        try:
            next_obs, reward, terminated, truncated, info = env.step(action)
        except (InvalidSessionIdException, NoSuchWindowException, WebDriverException) as e:
            print(f"[WARN] Selenium/Chrome died at step {step}: {e}. Restarting env...")
            try:
                env.close()
            except Exception:
                pass
            env = make_env(cfg)
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_start_time = time.time()
            continue

        done = float(terminated or truncated)
        rb.add(obs, action, float(reward), next_obs, done)

        obs = next_obs
        ep_reward += float(reward)

        if terminated or truncated:
            episodes += 1
            try:
                score = env.unwrapped.get_score()
            except Exception:
                score = 0
            wall_time = time.time() - ep_start_time
            metrics["rewards"].append(ep_reward)
            metrics["scores"].append(score)
            metrics["steps"].append(step)
            metrics["wall_times"].append(wall_time)

            print(f"[ep {episodes}] step={step} reward={ep_reward:.1f} score={score} time={wall_time:.1f}s")

            obs, _ = env.reset()
            ep_reward = 0.0
            ep_start_time = time.time()

        if step >= cfg.learning_starts and (step % cfg.train_every == 0) and len(rb) >= cfg.batch_size:
            s, a, r, ns, d = rb.sample(cfg.batch_size)

            s_t = torch.from_numpy(s).to(device)
            ns_t = torch.from_numpy(ns).to(device)
            a_t = torch.from_numpy(a).to(device)
            r_t = torch.from_numpy(r).to(device)
            d_t = torch.from_numpy(d).to(device)

            q_sa = q(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                next_a = torch.argmax(q(ns_t), dim=1)
                next_q = tgt(ns_t).gather(1, next_a.view(-1, 1)).squeeze(1)
                target = r_t + (1.0 - d_t) * cfg.gamma * next_q

            loss = F.smooth_l1_loss(q_sa, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip)
            opt.step()

        if step % cfg.target_update_every == 0:
            tgt.load_state_dict(q.state_dict())

        if step - last_log >= cfg.log_every:
            last_log = step
            sps = step / max(1e-6, (time.time() - t0))
            last_r = metrics["rewards"][-1] if metrics["rewards"] else 0.0
            print(f"step={step:7d}  eps={eps:.3f}  replay={len(rb):6d}  "
                  f"episodes={episodes:5d}  ep_r={last_r:7.1f}  sps={sps:7.1f}")

        if step % cfg.save_every == 0:
            save_path = os.path.join(cfg.save_dir, f"ckpt_step_{step}.pt")
            save_ckpt(save_path, step, q, tgt, opt, cfg, metrics)
            with open(cfg.metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved checkpoint: {save_path}")

    env.close()

    final_path = os.path.join(cfg.save_dir, "final_model.pt")
    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.save(q.state_dict(), final_path)

    with open(cfg.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Training complete. Model: {final_path}, Metrics: {cfg.metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=250_000)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    cfg.total_steps = args.total_steps
    train_ddqn(cfg, resume_from=args.resume)
