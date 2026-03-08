import torch, numpy as np, gymnasium as gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino
from model_ddqn_fixed import QNet, obs_to_numpy, unwrap_reset, to_torch_obs

N_EPISODES = 100

env = gym.make("ChromeDinoNoBrowser-v0")
env = make_dino(env, timer=True, frame_stack=True)
env.unwrapped.set_acceleration(True)
obs0 = obs_to_numpy(unwrap_reset(env.reset()))
H, W, C = obs0.shape

ckpts = ["ddqn_runs_final/seed_0/ckpt_step_300000.pt"]

print(f"{'step':>8} | {'rew_mean':>8} {'rew_std':>8} {'rew_min':>8} {'rew_max':>8} | {'sc_mean':>8} {'sc_std':>8} {'sc_min':>8} {'sc_max':>8}")
print("-" * 95)

for path in ckpts:
    ckpt = torch.load(path, map_location="cpu")
    n_actions = ckpt["q_state_dict"]["head.3.weight"].shape[0]
    q = QNet(in_channels=C, n_actions=n_actions, input_hw=(H, W))
    q.load_state_dict(ckpt["q_state_dict"])
    q.eval()

    rewards, scores = [], []
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
        scores.append(env.unwrapped.get_score())

    step = ckpt["step"]
    print(f"{step:>8} | {np.mean(rewards):>8.2f} {np.std(rewards):>8.2f} {min(rewards):>8.2f} {max(rewards):>8.2f} | "
          f"{np.mean(scores):>8.2f} {np.std(scores):>8.2f} {min(scores):>8.2f} {max(scores):>8.2f}")

env.close()
