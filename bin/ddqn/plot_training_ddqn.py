"""
Plot mean Q-value and episode reward vs training step across checkpoints.
Loads each checkpoint, runs N_EPISODES greedy episodes, records both metrics.
Saves plot as training_curves.png
"""
import torch
import numpy as np
import gymnasium as gym
import gym_chrome_dino
import matplotlib.pyplot as plt
import os, glob
from gym_chrome_dino.utils.wrappers import make_dino
from model_ddqn_fixed import QNet, obs_to_numpy, unwrap_reset, to_torch_obs

N_EPISODES = 20
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR   = os.path.join(SCRIPT_DIR, "ddqn_runs_final", "seed_0")

env = gym.make("ChromeDinoNoBrowser-v0")
env = make_dino(env, timer=True, frame_stack=True)
env.unwrapped.set_acceleration(True)
obs0 = obs_to_numpy(unwrap_reset(env.reset()))
H, W, C = obs0.shape

ckpts = sorted(
    [os.path.normpath(p) for p in glob.glob(f"{CKPT_DIR}/ckpt_step_*.pt")],
    key=lambda p: int(p.split("_")[-1].replace(".pt", ""))
)

steps, mean_qs, mean_rewards = [], [], []

for path in ckpts:
    ckpt = torch.load(path, map_location="cpu")
    n_actions = ckpt["q_state_dict"]["head.3.weight"].shape[0]
    q_net = QNet(in_channels=C, n_actions=n_actions, input_hw=(H, W))
    q_net.load_state_dict(ckpt["q_state_dict"])
    q_net.eval()

    ep_rewards, ep_mean_qs = [], []

    for _ in range(N_EPISODES):
        obs = obs_to_numpy(unwrap_reset(env.reset()))
        total, done, q_vals_ep = 0.0, False, []
        while not done:
            with torch.no_grad():
                q_vals = q_net(to_torch_obs(obs[None], "cpu"))[0]
            q_vals_ep.append(q_vals.mean().item())
            action = int(torch.argmax(q_vals).item())
            obs_raw, r, terminated, truncated, _ = env.step(action)
            obs = obs_to_numpy(obs_raw)
            total += r
            done = terminated or truncated
        ep_rewards.append(total)
        ep_mean_qs.append(np.mean(q_vals_ep))

    step = ckpt["step"]
    steps.append(step)
    mean_rewards.append(np.mean(ep_rewards))
    mean_qs.append(np.mean(ep_mean_qs))
    print(f"step={step:7d}  mean_r={np.mean(ep_rewards):7.2f}  mean_q={np.mean(ep_mean_qs):7.3f}")

env.close()

# ---------- plot ----------
fig, ax1 = plt.subplots(figsize=(10, 5))

color_r = "#2196F3"
color_q = "#F44336"

ax1.set_xlabel("Training Step")
ax1.set_ylabel("Mean Episode Reward", color=color_r)
ax1.plot(steps, mean_rewards, color=color_r, marker="o", linewidth=2, label="Mean Reward")
ax1.tick_params(axis="y", labelcolor=color_r)
ax1.set_xticks(steps)
ax1.set_xticklabels([f"{s//1000}k" for s in steps])

ax2 = ax1.twinx()
ax2.set_ylabel("Mean Q-Value", color=color_q)
ax2.plot(steps, mean_qs, color=color_q, marker="s", linewidth=2, linestyle="--", label="Mean Q-Value")
ax2.tick_params(axis="y", labelcolor=color_q)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.title("DDQN Training Progress: Reward and Q-Value vs Training Step")
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("\nSaved plot to training_curves.png")
plt.show()
