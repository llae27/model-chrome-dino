"""
Record a demo video of the trained DDQN agent playing Chrome Dino.
Runs N_ATTEMPTS episodes and saves the best scoring one as an MP4.

Requirements: pip install opencv-python
Usage:        python demo_video.py
"""
import torch
import numpy as np
import gymnasium as gym
import gym_chrome_dino
import cv2
from gym_chrome_dino.utils.wrappers import make_dino
from model_ddqn_fixed import QNet, obs_to_numpy, unwrap_reset, to_torch_obs

CKPT_PATH   = "ddqn_runs_final/seed_0/ckpt_step_300000.pt"
OUTPUT_PATH = "ddqn_demo.mp4"
N_ATTEMPTS  = 10          # run this many episodes, save the best
FPS         = 10          # match env render_fps

# ---------- load model ----------
ckpt = torch.load(CKPT_PATH, map_location="cpu")
n_actions = ckpt["q_state_dict"]["head.3.weight"].shape[0]

env = gym.make("ChromeDinoNoBrowser-v0")
env = make_dino(env, timer=True, frame_stack=True)
env.unwrapped.set_acceleration(True)

obs0 = obs_to_numpy(unwrap_reset(env.reset()))
H_obs, W_obs, C = obs0.shape

q = QNet(in_channels=C, n_actions=n_actions, input_hw=(H_obs, W_obs))
q.load_state_dict(ckpt["q_state_dict"])
q.eval()

# ---------- run episodes ----------
best_score   = -1
best_frames  = []

for attempt in range(1, N_ATTEMPTS + 1):
    obs = obs_to_numpy(unwrap_reset(env.reset()))
    frames, done = [], False

    while not done:
        with torch.no_grad():
            action = int(torch.argmax(q(to_torch_obs(obs[None], "cpu")), dim=1).item())

        obs_raw, _, terminated, truncated, _ = env.step(action)
        # capture after step so the final (death) frame is included
        frames.append(env.unwrapped.current_frame.copy())
        obs = obs_to_numpy(obs_raw)
        done = terminated or truncated

    score = env.unwrapped.get_score()
    print(f"Attempt {attempt:2d}/{N_ATTEMPTS}  score={score}")

    if score > best_score:
        best_score  = score
        best_frames = frames

env.close()

# ---------- write video ----------
if not best_frames:
    print("No frames captured.")
else:
    h, w = best_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (w, h))

    for frame in best_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"\nSaved best run (score={best_score}) to {OUTPUT_PATH}  ({len(best_frames)} frames)")
