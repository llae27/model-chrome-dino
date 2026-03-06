# train_ddqn_dino.py
import os
import time
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

import gym_chrome_dino  # registers env ids
from gym_chrome_dino.utils.wrappers import make_dino  # 160x80 grayscale + FrameStack(4)
from gym_chrome_dino.envs.chrome_dino_env import ACTION_MEANING
from selenium.common.exceptions import (
    InvalidSessionIdException, WebDriverException, NoSuchWindowException,
    SessionNotCreatedException,
)


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    env_id: str = "ChromeDinoNoBrowser-v0"

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Training length
    total_steps: int = 300_000
    learning_starts: int = 10_000
    train_every: int = 2
    updates_per_step: int = 1

    # RL
    gamma: float = 0.99
    batch_size: int = 32
    replay_size: int = 200_000

    # Optim
    lr: float = 1e-4
    grad_clip: float = 1.0

    # Epsilon schedule (linear)
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_steps: int = 200_000

    # Target network
    target_update_every: int = 10_000  # steps

    # Saving / logging
    log_every: int = 2_000
    save_every: int = 50_000
    out_dir: str = "ddqn_runs"


@dataclass
class FastConfig(Config):
    """Scaled-down config for quick smoke tests (~10-15 min at 6-15 sps).
    Use this to verify hyperparameter changes before committing to a full run.
    All ratios are preserved: learning_starts/replay_size, eps_decay/total_steps, etc.
    """
    total_steps: int = 12_000
    learning_starts: int = 1_000
    replay_size: int = 5_000
    eps_decay_steps: int = 7_000   # same ratio as 150k/250k
    target_update_every: int = 250
    save_every: int = 4_000
    log_every: int = 200
    out_dir: str = "ddqn_fast_runs"


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_reset(reset_out):
    # Gymnasium returns (obs, info); handle both
    return reset_out[0] if isinstance(reset_out, tuple) else reset_out


def obs_to_numpy(obs):
    x = np.asarray(obs)
    return np.ascontiguousarray(x)


def to_torch_obs(obs_bhwc: np.ndarray, device: str) -> torch.Tensor:
    # (B,H,W,C) uint8/float -> float32 (B,C,H,W) in [0,1]
    if obs_bhwc.dtype != np.float32:
        x = obs_bhwc.astype(np.float32) / 255.0
    else:
        x = obs_bhwc
    x = torch.from_numpy(x).to(device)
    return x.permute(0, 3, 1, 2).contiguous()


# ----------------------------
# Replay Buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, dtype=np.uint8):
        self.capacity = capacity
        self.obs_shape = obs_shape

        self.s = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.ns = np.zeros((capacity, *obs_shape), dtype=dtype)
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
        return (
            self.s[idxs],
            self.a[idxs],
            self.r[idxs],
            self.ns[idxs],
            self.d[idxs],
        )


# ----------------------------
# Q Network (CNN with dynamic flatten)
# Works with 80x160x4 from make_dino().
# ----------------------------
class QNet(nn.Module):
    def __init__(self, in_channels: int, n_actions: int, input_hw):
        super().__init__()
        H, W = input_hw

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # compute conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            n_flat = self.conv(dummy).view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)


# ----------------------------
# Main trainer (DDQN)
# ----------------------------
def linear_epsilon(step: int, cfg: Config) -> float:
    if step >= cfg.eps_decay_steps:
        return cfg.eps_end
    frac = step / float(cfg.eps_decay_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def make_env(cfg: Config, retries: int = 5, wait: float = 5.0):
    # Match worker.py exactly. Retries handle Chrome crashes during restart.
    for attempt in range(retries):
        try:
            env = gym.make(cfg.env_id)
            env = make_dino(env, timer=True, frame_stack=True)
            env.unwrapped.set_acceleration(True)
            return env
        except Exception as e:
            if attempt < retries - 1:
                print(f"[WARN] make_env attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def save_ckpt(path: str, step: int, q: nn.Module, tgt: nn.Module, opt: torch.optim.Optimizer, cfg: Config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "q_state_dict": q.state_dict(),
            "tgt_state_dict": tgt.state_dict(),
            "opt_state_dict": opt.state_dict(),
            "cfg": cfg.__dict__,
        },
        path,
    )


def train_ddqn(cfg: Config, resume_from: str = None):
    set_seed(cfg.seed)
    device = cfg.device

    env = make_env(cfg)

    # Get obs shape from one reset
    obs0 = obs_to_numpy(unwrap_reset(env.reset()))
    H, W, C = obs0.shape
    assert C == 4, f"Expected frame-stacked C=4, got {obs0.shape}"

    # action_space.n == 2: action 0 = NOOP, action 1 = UP (jump)
    n_actions = env.action_space.n

    q = QNet(in_channels=C, n_actions=n_actions, input_hw=(H, W)).to(device)
    tgt = QNet(in_channels=C, n_actions=n_actions, input_hw=(H, W)).to(device)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)

    rb = ReplayBuffer(cfg.replay_size, obs_shape=(H, W, C), dtype=np.uint8)

    start_step = 1
    if resume_from is not None:
        print(f"Loading checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        q.load_state_dict(ckpt["q_state_dict"])
        tgt.load_state_dict(ckpt["tgt_state_dict"])
        opt.load_state_dict(ckpt["opt_state_dict"])
        start_step = ckpt["step"] + 1
        print(f"Resuming from step {start_step}  eps={linear_epsilon(start_step, cfg):.3f}")

    obs = obs0
    ep_reward = 0.0
    ep_len = 0
    episodes = 0
    last_ep_reward = 0.0
    last_loss = float("nan")
    last_mean_q = float("nan")

    t0 = time.time()
    last_log = start_step - 1

    for step in range(start_step, cfg.total_steps + 1):
        eps = linear_epsilon(step, cfg)

        # -------- act (epsilon-greedy) --------
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                x = to_torch_obs(obs[None, ...], device)  # (1,C,H,W)
                action = int(torch.argmax(q(x), dim=1).item())

        # -------- step --------
        try:
            next_obs, reward, terminated, truncated, info = env.step(action)
        except (InvalidSessionIdException, NoSuchWindowException, WebDriverException) as e:
            print(f"[WARN] Selenium/Chrome died at step {step}: {e}. Restarting env...")

            # best-effort cleanup
            try:
                env.close()
            except Exception:
                pass

            # make_env already retries internally (up to 5x with 5s waits)
            env = make_env(cfg)
            obs = obs_to_numpy(unwrap_reset(env.reset()))
            continue

        next_obs = obs_to_numpy(next_obs)

        done = float(terminated or truncated)
        rb.add(obs, action, float(reward), next_obs, done)

        obs = next_obs
        ep_reward += float(reward)
        ep_len += 1

        # -------- episode reset --------
        if terminated or truncated:
            episodes += 1
            last_ep_reward = ep_reward  # save before reset so we can log it
            obs = obs_to_numpy(unwrap_reset(env.reset()))
            ep_reward = 0.0
            ep_len = 0

        # -------- learn --------
        if len(rb) >= cfg.learning_starts and (step % cfg.train_every == 0):
            for _ in range(cfg.updates_per_step):
                s, a, r, ns, d = rb.sample(cfg.batch_size)

                s_t = to_torch_obs(s, device)
                ns_t = to_torch_obs(ns, device)
                a_t = torch.from_numpy(a).to(device)
                r_t = torch.from_numpy(r).to(device)
                d_t = torch.from_numpy(d).to(device)

                # Q(s,a)
                q_sa = q(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)

                with torch.no_grad():
                    # DDQN: a* = argmax_a Q_online(s',a)
                    next_a = torch.argmax(q(ns_t), dim=1)
                    # Q_target(s', a*)
                    next_q = tgt(ns_t).gather(1, next_a.view(-1, 1)).squeeze(1)
                    target = r_t + (1.0 - d_t) * cfg.gamma * next_q

                loss = F.smooth_l1_loss(q_sa, target)
                last_loss = loss.item()
                last_mean_q = q_sa.mean().item()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip)
                opt.step()

        # -------- target update --------
        if step % cfg.target_update_every == 0:
            tgt.load_state_dict(q.state_dict())

        # -------- logging --------
        if step - last_log >= cfg.log_every:
            last_log = step
            steps_per_sec = step / max(1e-6, (time.time() - t0))
            print(
                f"step={step:7d}  eps={eps:.3f}  replay={len(rb):6d}  "
                f"episodes={episodes:5d}  ep_r={last_ep_reward:7.1f}  "
                f"loss={last_loss:8.4f}  mean_q={last_mean_q:7.3f}  sps={steps_per_sec:7.1f}"
            )

        # -------- save --------
        if step % cfg.save_every == 0:
            run_dir = os.path.join(cfg.out_dir, f"seed_{cfg.seed}")
            save_path = os.path.join(run_dir, f"ckpt_step_{step}.pt")
            save_ckpt(save_path, step, q, tgt, opt, cfg)
            print(f"Saved checkpoint: {save_path}")

    env.close()
    return q


if __name__ == "__main__":
    # FastConfig: ~10-15 min smoke test for tuning hyperparameters
    # Config:     full 300k run (~10+ hours)
    # cfg = FastConfig()
    cfg = Config()
    train_ddqn(cfg, resume_from=None)
