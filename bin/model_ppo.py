# ppo_dino_single_env.py
# PPO (clip) for Gymnasium + gym_chrome_dino, SINGLE ENV
# Robust to FrameStack returning lists/LazyFrames and to non-contiguous tensors.

import math
import time
import random
from dataclasses import dataclass
from typing import Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import gymnasium as gym
import gym_chrome_dino  # registers env id
from gym_chrome_dino.utils.wrappers import make_dino

def default_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ----------------------------
# Config
# ----------------------------
@dataclass
class CFG:
    env_id: str = "ChromeDinoNoBrowser-v0"

    seed: int = 42
    device: str = default_device()

    # Rollout collection (single env)
    num_steps: int = 2048
    # total_timesteps: int = 1_000_000
    total_timesteps: int = 250_000

    # PPO hyperparams
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

    # Dino wrappers
    timer: bool = True
    frame_stack: bool = True
    acceleration: bool = True

    # Logging / saving
    log_every_updates: int = 1
    save_path: str = "ppo_dino.pt"


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_reset(env: gym.Env, seed: int | None = None):
    """Try reset(seed=...), fall back to reset() if wrapper doesn't accept seed kwarg."""
    if seed is None:
        return env.reset()
    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def fix_obs(obs: Any) -> np.ndarray:
    """
    Ensure observation is a np.ndarray uint8 with shape (80,160,4).
    Handles cases where FrameStack returns list/tuple/LazyFrames/array with different layout.
    """
    arr = np.asarray(obs)

    # If it's an object array (common when it's a list of frames), stack properly
    if arr.dtype == object:
        frames = list(obs)
        frames = [np.asarray(f) for f in frames]
        # squeeze last dim if (H,W,1)
        frames = [f.squeeze(-1) if (f.ndim == 3 and f.shape[-1] == 1) else f for f in frames]
        arr = np.stack(frames, axis=0)  # (K,H,W) likely

    # Now handle possible shapes
    if arr.shape == (80, 160, 4):
        out = arr
    elif arr.shape == (4, 80, 160):
        out = np.transpose(arr, (1, 2, 0))  # (80,160,4)
    elif arr.ndim == 4 and arr.shape[0] == 4 and arr.shape[1] == 80 and arr.shape[2] == 160:
        # (4,80,160,C?) -> try to squeeze
        out = arr
        if out.shape[-1] == 1:
            out = out.squeeze(-1)            # (4,80,160)
            out = np.transpose(out, (1, 2, 0))
        elif out.shape[-1] == 4:
            # could be (4,80,160,4) from weird stacking; take last frame-stack dimension
            out = out[..., -1]               # (4,80,160)
            out = np.transpose(out, (1, 2, 0))
        else:
            raise ValueError(f"Unexpected obs shape {arr.shape}")
    elif arr.shape == (80, 160):
        # if somehow single frame arrives, expand to 4 channels (fallback)
        out = np.repeat(arr[:, :, None], 4, axis=2)
    else:
        raise ValueError(f"Unexpected obs shape {arr.shape}")

    # Clip + dtype normalize
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)

    return out


# ----------------------------
# Env
# ----------------------------
def make_env(cfg: CFG) -> gym.Env:
    try:
        env = gym.make(cfg.env_id, render=False, accelerate=False)
    except TypeError:
        env = gym.make(cfg.env_id)

    env = make_dino(env, timer=cfg.timer, frame_stack=cfg.frame_stack)

    # Seed spaces (helps reproducibility even if reset(seed=...) doesn't)
    try:
        env.action_space.seed(cfg.seed)
    except Exception:
        pass
    try:
        env.observation_space.seed(cfg.seed)
    except Exception:
        pass

    try:
        env.unwrapped.set_acceleration(cfg.acceleration)
    except Exception:
        pass

    return env


# ----------------------------
# Actor-Critic
# ----------------------------
class ActorCritic(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.n_actions = n_actions

        # obs is (80,160,4) channels-last, uint8
        c, h, w = 4, 80, 160

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flat = self.cnn(dummy).reshape(1, -1).size(1)

        self.shared = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(512, self.n_actions)
        self.value_head = nn.Linear(512, 1)

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

    def preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs:
          - (80,160,4) uint8  or (B,80,160,4) uint8
        -> (B,4,80,160) float32 [0,1]
        """
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        # permute makes it non-contiguous; call contiguous() to be safe
        obs = obs.permute(0, 3, 1, 2).contiguous()
        return obs.float() / 255.0

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.preprocess(obs)
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)  # <-- reshape instead of view
        x = self.shared(x)
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


# ----------------------------
# GAE (single env)
# ----------------------------
def compute_gae_1env(
    rewards: torch.Tensor,      # (T,)
    dones: torch.Tensor,        # (T,) 0/1
    values: torch.Tensor,       # (T,)
    last_value: torch.Tensor,   # scalar
    gamma: float,
    lam: float,
):
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


# ----------------------------
# Training
# ----------------------------
def train(cfg: CFG):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    print(f"Using device: {device}")

    env = make_env(cfg)

    # Check action space only (obs checker warnings can happen due to wrapper returns; we fix_obs anyway)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 2, f"Expected Discrete(2), got {env.action_space}"

    model = ActorCritic(env.action_space.n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    T = cfg.num_steps
    batch_size = T
    minibatch_size = batch_size // cfg.num_minibatches
    assert minibatch_size > 0 and batch_size % cfg.num_minibatches == 0

    num_updates = cfg.total_timesteps // T
    assert num_updates >= 1

    # Rollout buffers
    obs_buf = np.zeros((T, 80, 160, 4), dtype=np.uint8)
    actions_buf = np.zeros((T,), dtype=np.int64)
    logprobs_buf = np.zeros((T,), dtype=np.float32)
    rewards_buf = np.zeros((T,), dtype=np.float32)
    dones_buf = np.zeros((T,), dtype=np.float32)
    values_buf = np.zeros((T,), dtype=np.float32)

    global_step = 0
    start_time = time.time()

    ep_return = 0.0
    completed_returns = []

    obs, _ = safe_reset(env, seed=cfg.seed)
    obs = fix_obs(obs)

    for update in range(1, num_updates + 1):
        # LR anneal
        frac = 1.0 - (update - 1.0) / num_updates
        opt.param_groups[0]["lr"] = cfg.lr * frac

        # Collect rollout
        for t in range(T):
            global_step += 1

            obs_buf[t] = obs

            # Fast + correct: numpy -> torch
            obs_t = torch.from_numpy(obs).to(device)  # uint8 tensor
            action_t, logprob_t, value_t = model.act(obs_t)

            action = int(action_t.item())
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            actions_buf[t] = action
            logprobs_buf[t] = float(logprob_t.item())
            values_buf[t] = float(value_t.item())
            rewards_buf[t] = float(reward)
            dones_buf[t] = 1.0 if done else 0.0

            ep_return += float(reward)

            if done:
                completed_returns.append(ep_return)
                
                print(f"[episode finished] step={global_step} score={completed_returns[-1]:.2f} "
                      f"(episodes={len(completed_returns)})")
                
                ep_return = 0.0
                next_obs, _ = safe_reset(env)

            obs = fix_obs(next_obs)

        # Bootstrap last value
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device)
            _, last_value = model.forward(obs_t)
            last_value = last_value.squeeze(0).cpu().float()

        # GAE/Returns (CPU torch)
        rewards_t = torch.from_numpy(rewards_buf)
        dones_t = torch.from_numpy(dones_buf)
        values_t = torch.from_numpy(values_buf)

        adv_t, rets_t = compute_gae_1env(
            rewards=rewards_t,
            dones=dones_t,
            values=values_t,
            last_value=last_value,
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
        )

        # Advantage normalization
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

                mb_obs = torch.from_numpy(b_obs[mb]).to(device)  # uint8
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
            recent = completed_returns[-20:] if completed_returns else []
            avg_ret = float(np.mean(recent)) if recent else float("nan")
            print(
                f"update {update:4d}/{num_updates} | steps {global_step:8d} | "
                f"SPS {sps:5d} | avg_return(20) {avg_ret:8.2f} | clipfrac {np.mean(clipfracs):.3f}"
            )

    env.close()
    torch.save(model.state_dict(), cfg.save_path)
    print(f"Saved to {cfg.save_path}")


@torch.no_grad()
def evaluate(cfg: CFG, episodes: int = 3):
    device = torch.device(cfg.device)
    env = make_env(cfg)

    model = ActorCritic(env.action_space.n).to(device)
    model.load_state_dict(torch.load(cfg.save_path, map_location=device))
    model.eval()

    for ep in range(episodes):
        obs, _ = safe_reset(env, seed=cfg.seed + 1000 + ep)
        obs = fix_obs(obs)
        done = False
        ret = 0.0
        while not done:
            obs_t = torch.from_numpy(obs).to(device)
            logits, _ = model.forward(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ret += float(reward)
            if not done:
                obs = fix_obs(obs)
        print(f"eval ep {ep+1}: return={ret:.2f}")

    env.close()


if __name__ == "__main__":
    cfg = CFG()
    train(cfg)
    # evaluate(cfg, episodes=3)