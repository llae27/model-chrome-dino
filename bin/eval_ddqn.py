import torch, numpy as np, gymnasium as gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino
from model_ddqn_fixed import QNet, obs_to_numpy, unwrap_reset, to_torch_obs
import glob, os

N_EPISODES = 20

env = gym.make("ChromeDinoNoBrowser-v0")
env = make_dino(env, timer=True, frame_stack=True)
env.unwrapped.set_acceleration(True)
obs0 = obs_to_numpy(unwrap_reset(env.reset()))
H, W, C = obs0.shape

ckpts = sorted(glob.glob("ddqn_runs/seed_0/ckpt_step_*.pt"),
               key=lambda p: int(p.split("_")[-1].replace(".pt", "")))

for path in ckpts:
    ckpt = torch.load(path, map_location="cpu")
    q = QNet(in_channels=C, n_actions=2, input_hw=(H, W))
    q.load_state_dict(ckpt["q_state_dict"])
    q.eval()

    rewards = []
    for _ in range(N_EPISODES):
        obs = obs_to_numpy(unwrap_reset(env.reset()))
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                action = int(torch.argmax(q(to_torch_obs(obs[None], "cpu")), dim=1).item())
            obs_raw, r, terminated, truncated, _ = env.step(action)
            obs = obs_to_numpy(obs_raw)
            total += r
            done = terminated or truncated
        rewards.append(total)

    step = ckpt["step"]
    print(f"step {step:7d} | mean={np.mean(rewards):6.2f}  std={np.std(rewards):5.2f}  "
          f"min={min(rewards):5.1f}  max={max(rewards):5.1f}")

env.close()
