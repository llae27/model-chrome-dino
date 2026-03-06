"""
Compute and display Q-values from saved DDQN checkpoints.
Runs N_EPISODES per checkpoint, recording Q-values at each step.
Prints mean Q-value per action and overall statistics.
"""
import torch
import numpy as np
import gymnasium as gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino
from model_ddqn_fixed import QNet, obs_to_numpy, unwrap_reset, to_torch_obs
import glob, os

N_EPISODES = 5   # episodes per checkpoint (keep low, each can be long)
CKPT_DIR   = "ddqn_runs/seed_0"

env = gym.make("ChromeDinoNoBrowser-v0")
env = make_dino(env, timer=True, frame_stack=True)
env.unwrapped.set_acceleration(True)
obs0 = obs_to_numpy(unwrap_reset(env.reset()))
H, W, C = obs0.shape

ckpts = sorted(
    [os.path.normpath(p) for p in glob.glob(f"{CKPT_DIR}/ckpt_step_*.pt")],
    key=lambda p: int(p.split("_")[-1].replace(".pt", ""))
)

print(f"{'step':>8}  {'mean_q':>8}  {'q_noop':>8}  {'q_jump':>8}  {'pct_jump':>9}  {'mean_r':>8}")
print("-" * 65)

for path in ckpts:
    ckpt = torch.load(path, map_location="cpu")
    n_actions = ckpt["q_state_dict"]["head.3.weight"].shape[0]
    q_net = QNet(in_channels=C, n_actions=n_actions, input_hw=(H, W))
    q_net.load_state_dict(ckpt["q_state_dict"])
    q_net.eval()

    all_q_noop, all_q_jump, all_rewards = [], [], []

    for _ in range(N_EPISODES):
        obs = obs_to_numpy(unwrap_reset(env.reset()))
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                q_vals = q_net(to_torch_obs(obs[None], "cpu"))[0]  # (n_actions,)
            all_q_noop.append(q_vals[0].item())
            all_q_jump.append(q_vals[1].item())
            action = int(torch.argmax(q_vals).item())
            obs_raw, r, terminated, truncated, _ = env.step(action)
            obs = obs_to_numpy(obs_raw)
            total += r
            done = terminated or truncated
        all_rewards.append(total)

    mean_q     = np.mean(all_q_noop + all_q_jump)
    mean_noop  = np.mean(all_q_noop)
    mean_jump  = np.mean(all_q_jump)
    pct_jump   = 100.0 * np.mean(np.array(all_q_jump) > np.array(all_q_noop))
    mean_r     = np.mean(all_rewards)
    step       = ckpt["step"]

    print(f"{step:>8}  {mean_q:>8.3f}  {mean_noop:>8.3f}  {mean_jump:>8.3f}  {pct_jump:>8.1f}%  {mean_r:>8.2f}")

env.close()
