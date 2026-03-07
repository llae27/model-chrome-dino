import sys
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import gymnasium as gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino
from gym_chrome_dino.envs.chrome_dino_env import ACTION_MEANING


# ---- Same obs-fixer we used in training ----
def fix_obs(obs):
    """
    Ensure observation is a np.ndarray uint8 with shape (80,160,4).
    Handles cases where FrameStack returns list/tuple/LazyFrames/object arrays.
    """
    arr = np.asarray(obs)

    if arr.dtype == object:
        frames = list(obs)
        frames = [np.asarray(f) for f in frames]
        frames = [f.squeeze(-1) if (f.ndim == 3 and f.shape[-1] == 1) else f for f in frames]
        arr = np.stack(frames, axis=0)  # (K,H,W) likely

    if arr.shape == (80, 160, 4):
        out = arr
    elif arr.shape == (4, 80, 160):
        out = np.transpose(arr, (1, 2, 0))
    elif arr.shape == (80, 160):
        out = np.repeat(arr[:, :, None], 4, axis=2)  # fallback
    else:
        raise ValueError(f"Unexpected obs shape {arr.shape} (dtype={arr.dtype})")

    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)

    return out


# ---- Minimal ActorCritic compatible with your saved PPO weights ----
class ActorCritic(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
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

    def preprocess(self, obs_t: torch.Tensor) -> torch.Tensor:
        # obs_t: (80,160,4) uint8 or (B,80,160,4) uint8 -> (B,4,80,160) float in [0,1]
        if obs_t.ndim == 3:
            obs_t = obs_t.unsqueeze(0)
        obs_t = obs_t.permute(0, 3, 1, 2).contiguous().float() / 255.0
        return obs_t

    def forward(self, obs_t: torch.Tensor):
        x = self.preprocess(obs_t)
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        x = self.shared(x)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


def main():
    # model path from CLI (default)
    model_path = sys.argv[1] if len(sys.argv) >= 2 else "ppo_dino.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Starting PPO worker with model:", model_path)

    with open("dino_log.txt", "w") as log_file:
        sys.stdout = log_file

        # initialize env
        env = gym.make("ChromeDinoNoBrowser-v0")
        env = make_dino(env, timer=True, frame_stack=True)

        # NOTE: if PPO was trained with acceleration=False, turning this on may hurt performance.
        try:
            env.unwrapped.set_acceleration(True)
        except Exception:
            pass

        params = env.unwrapped.game.get_parameters()
        print(f"Current acceleration: {params.get('config.ACCELERATION', 'N/A')}")

        # Print env details for debugging (comment out if you just want to run the agent)
        print("------------------------------")
        print("obs_space:", env.observation_space)
        obs, _ = env.reset()
        obs_fixed = fix_obs(obs)
        print("fixed obs:", obs_fixed.shape, obs_fixed.dtype, obs_fixed.min(), obs_fixed.max())
        print("action_space:", env.action_space)
        print("------------------------------")

        # load model
        model = ActorCritic(n_actions=env.action_space.n).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"Loaded model weights from: {model_path}")

        terminated = False
        truncated = False

        # run experiment
        step = 0
        total_reward = 0.0
        while not (terminated or truncated):
            step += 1

            # choose action from PPO policy (deterministic argmax)
            obs_fixed = fix_obs(obs)
            obs_t = torch.from_numpy(obs_fixed).to(device)  # uint8 tensor

            with torch.no_grad():
                logits, _ = model.forward(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())

            # render (this can be slow; comment out if you only care about stepping)
            frame = env.render()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            # ACTION_MEANING has 0..3 but your env is Discrete(2), so only 0/1 are used.
            meaning = ACTION_MEANING.get(action, str(action))
            print(f"step={step} action={meaning} reward={reward} term={terminated} trunc={truncated} info={info} frame={getattr(frame,'shape',None)}")

        score = env.unwrapped.get_score()
        print(f"Final score: {score} | total_reward: {total_reward:.2f}")

        sys.stdout = sys.__stdout__

    print(f"Experiment completed. Logs saved to dino_log.txt")

if __name__ == "__main__":
    main()