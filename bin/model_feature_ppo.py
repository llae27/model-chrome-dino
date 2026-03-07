# Feature-based PPO for Chrome Dino
# MLP actor-critic using 8-dim engineered features instead of pixels.

import argparse
import json
import math
import os
import time
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

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
class CFG:
    env_id: str = "ChromeDinoNoBrowser-v0"
    seed: int = 42
    device: str = default_device()

    num_steps: int = 2048
    total_timesteps: int = 250_000

    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_vloss: bool = True

    acceleration: bool = True

    log_every_updates: int = 1
    ckpt_every_updates: int = 25
    save_dir: str = "ppo_runs/feature"
    metrics_path: str = "ppo_feature_metrics.json"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(cfg: CFG, retries: int = 5, wait: float = 5.0) -> gym.Env:
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


class FeatureActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 8, n_actions: int = 2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        x = self.shared(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value


def compute_gae_1env(rewards, dones, values, last_value, gamma, lam):
    T = rewards.shape[0]
    adv = torch.zeros((T,), dtype=torch.float32)
    last_gae = torch.tensor(0.0, dtype=torch.float32)
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def save_checkpoint(path, update, global_step, model, opt, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "update": update,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "opt_state_dict": opt.state_dict(),
        "metrics": metrics,
    }, path)


def train(cfg: CFG, resume_path: str = None):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    env = make_env(cfg)
    n_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    print(f"obs_dim={obs_dim}, n_actions={n_actions}")

    model = FeatureActorCritic(obs_dim=obs_dim, n_actions=n_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    T = cfg.num_steps
    batch_size = T
    minibatch_size = batch_size // cfg.num_minibatches
    num_updates = cfg.total_timesteps // T

    obs_buf = np.zeros((T, obs_dim), dtype=np.float32)
    actions_buf = np.zeros((T,), dtype=np.int64)
    logprobs_buf = np.zeros((T,), dtype=np.float32)
    rewards_buf = np.zeros((T,), dtype=np.float32)
    dones_buf = np.zeros((T,), dtype=np.float32)
    values_buf = np.zeros((T,), dtype=np.float32)

    metrics = {"rewards": [], "scores": [], "steps": [], "wall_times": [],
               "algorithm": "ppo", "representation": "feature",
               "total_steps": cfg.total_timesteps,
               "hyperparams": {"lr": cfg.lr, "gamma": cfg.gamma, "gae_lambda": cfg.gae_lambda,
                               "clip_coef": cfg.clip_coef, "ent_coef": cfg.ent_coef,
                               "vf_coef": cfg.vf_coef, "num_steps": cfg.num_steps}}

    global_step = 0
    start_update = 1

    if resume_path is not None:
        print(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["opt_state_dict"])
        start_update = ckpt["update"] + 1
        global_step = ckpt["global_step"]
        metrics = ckpt["metrics"]
        print(f"Resuming from update {start_update}, global_step {global_step}")

    start_time = time.time()
    ep_return = 0.0
    ep_start_time = time.time()

    obs, _ = env.reset()

    for update in range(start_update, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        opt.param_groups[0]["lr"] = cfg.lr * frac

        for t in range(T):
            global_step += 1
            obs_buf[t] = obs

            obs_t = torch.from_numpy(obs).float().to(device)
            action_t, logprob_t, value_t = model.act(obs_t)

            action = int(action_t.item())

            try:
                next_obs, reward, terminated, truncated, info = env.step(action)
            except (InvalidSessionIdException, NoSuchWindowException, WebDriverException):
                print(f"[WARN] Selenium crash at step {global_step}. Restarting env...")
                try:
                    env.close()
                except Exception:
                    pass
                env = make_env(cfg)
                next_obs, _ = env.reset()
                reward, terminated, truncated = 0.0, True, False

            done = bool(terminated or truncated)

            actions_buf[t] = action
            logprobs_buf[t] = float(logprob_t.item())
            values_buf[t] = float(value_t.item())
            rewards_buf[t] = float(reward)
            dones_buf[t] = 1.0 if done else 0.0

            ep_return += float(reward)

            if done:
                try:
                    score = env.unwrapped.get_score()
                except Exception:
                    score = 0
                wall_time = time.time() - ep_start_time
                metrics["rewards"].append(ep_return)
                metrics["scores"].append(score)
                metrics["steps"].append(global_step)
                metrics["wall_times"].append(wall_time)

                print(f"[ep {len(metrics['rewards'])}] step={global_step} reward={ep_return:.1f} score={score} time={wall_time:.1f}s")

                ep_return = 0.0
                ep_start_time = time.time()
                next_obs, _ = env.reset()

            obs = next_obs

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(device)
            _, last_value = model.forward(obs_t)
            last_value = last_value.squeeze(0).cpu().float()

        rewards_t = torch.from_numpy(rewards_buf)
        dones_t = torch.from_numpy(dones_buf)
        values_t = torch.from_numpy(values_buf)

        adv_t, rets_t = compute_gae_1env(
            rewards=rewards_t, dones=dones_t, values=values_t,
            last_value=last_value, gamma=cfg.gamma, lam=cfg.gae_lambda,
        )

        adv_np = adv_t.numpy()
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        b_obs = obs_buf
        b_actions = actions_buf
        b_old_logp = logprobs_buf
        b_old_v = values_buf
        b_adv = adv_np
        b_rets = rets_t.numpy()

        inds = np.arange(batch_size)
        clipfracs = []

        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb = inds[start:start + minibatch_size]

                mb_obs = torch.from_numpy(b_obs[mb]).to(device)
                mb_actions = torch.from_numpy(b_actions[mb]).to(device)
                mb_old_logp_t = torch.from_numpy(b_old_logp[mb]).to(device)
                mb_adv_t = torch.from_numpy(b_adv[mb]).to(device)
                mb_rets_t = torch.from_numpy(b_rets[mb]).to(device)
                mb_old_v_t = torch.from_numpy(b_old_v[mb]).to(device)

                logits, new_v = model.forward(mb_obs)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                logratio = new_logp - mb_old_logp_t
                ratio = logratio.exp()

                pg1 = -mb_adv_t * ratio
                pg2 = -mb_adv_t * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                if cfg.clip_vloss:
                    v_clipped = mb_old_v_t + torch.clamp(new_v - mb_old_v_t, -cfg.clip_coef, cfg.clip_coef)
                    v_loss_unclipped = (new_v - mb_rets_t) ** 2
                    v_loss_clipped = (v_clipped - mb_rets_t) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_v - mb_rets_t) ** 2).mean()

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

                clipfrac = (torch.abs(ratio - 1.0) > cfg.clip_coef).float().mean().item()
                clipfracs.append(clipfrac)

        if update % cfg.log_every_updates == 0:
            sps = int(global_step / (time.time() - start_time))
            recent = metrics["rewards"][-20:] if metrics["rewards"] else []
            avg_ret = float(np.mean(recent)) if recent else float("nan")
            print(f"update {update:4d}/{num_updates} | steps {global_step:8d} | "
                  f"SPS {sps:5d} | avg_return(20) {avg_ret:8.2f} | clipfrac {np.mean(clipfracs):.3f}")

        if update % cfg.ckpt_every_updates == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"ckpt_update_{update}.pt")
            save_checkpoint(ckpt_path, update, global_step, model, opt, metrics)
            with open(cfg.metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Checkpoint saved: {ckpt_path}")

    env.close()

    final_path = os.path.join(cfg.save_dir, "final_model.pt")
    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.save(model.state_dict(), final_path)

    with open(cfg.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Training complete. Model: {final_path}, Metrics: {cfg.metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=250_000)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = CFG()
    cfg.total_timesteps = args.total_steps
    train(cfg, resume_path=args.resume)
