# Atari CNN DQN
import json
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

import gymnasium as gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ACTIONS = 3

class DQN(nn.Module):
  def __init__(self, in_channels, num_actions):
    super(DQN, self).__init__()
    self.net = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=10, stride=10),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=2, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1344, 512),
        nn.ReLU(),
        nn.Linear(512, num_actions)
    )
  def forward(self, x):
    return self.net(x)

class ReplayBuffer:
  def __init__(self, capacity):
    self.buffer = deque(maxlen=capacity)
  def push(self, s, a, r, ns, d):
    self.buffer.append((s, a, r, ns, d))
  def sample(self, batch_size):
    batch = random.sample(self.buffer, batch_size)
    s, a, r, ns, d = zip(*batch)
    return (
        torch.tensor(np.array(s), dtype=torch.float).to(DEVICE),
        torch.tensor(a, dtype=torch.long).to(DEVICE),
        torch.tensor(r, dtype=torch.float).to(DEVICE),
        torch.tensor(np.array(ns), dtype=torch.float).to(DEVICE),
        torch.tensor(d, dtype=torch.float).to(DEVICE),
    )
  def __len__(self):
    return len(self.buffer)

def train(
  batch_size: int = 40,
  learning_rate: float = 0.001,
  optimize_gamma: float = 0.99,
  len_memory: int = 10000,
  num_episodes: int = 100,
  num_targe_update: int = 10,
):
  env = gym.make("ChromeDinoNoBrowser-v0")
  env = make_dino(env, timer=False, frame_stack=True)
  env.unwrapped.set_acceleration(True)

  in_channels = 4

  policy_net = DQN(in_channels, NUM_ACTIONS).to(DEVICE)
  target_net = DQN(in_channels, NUM_ACTIONS).to(DEVICE)
  target_net.load_state_dict(policy_net.state_dict())

  criterion = nn.MSELoss()
  optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

  def optimize_model():
    if len(memory) < batch_size:
      return
    s, a, r, ns, d = memory.sample(batch_size)
    s = s.permute(0, 3, 1, 2)
    ns = ns.permute(0, 3, 1, 2)
    q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
      next_q_values = target_net(ns)
      target_q_values = r + (1 - d) * optimize_gamma * next_q_values.max(1)[0]
    loss = criterion(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  memory = ReplayBuffer(len_memory)
  epsilon = 1.0
  EPSILON_DECAY = 0.995
  EPSILON_MIN = 0.1

  metrics = {
      "rewards": [],
      "scores": [],
      "episode_times": [],
      "epsilons": [],
  }

  for episode in range(num_episodes):
      ep_start = time.time()
      done = False
      state, _ = env.reset()
      state = np.array(state)
      total_reward = 0
      while not done:
        if random.random() < epsilon:
          action = random.randint(0, NUM_ACTIONS - 1)
        else:
          state_tensor = torch.FloatTensor([state]).permute(0, 3, 1, 2).to(DEVICE)
          action = policy_net(state_tensor).argmax().item()
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_state)
        done = terminated or truncated
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if len(memory) > batch_size:
          optimize_model()
      ep_time = time.time() - ep_start
      epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
      if episode % num_targe_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

      score = env.unwrapped.get_score()
      metrics["rewards"].append(total_reward)
      metrics["scores"].append(score)
      metrics["episode_times"].append(ep_time)
      metrics["epsilons"].append(epsilon)

      print(
          f"Episode {episode:3d}/{num_episodes} | "
          f"Reward: {total_reward:7.1f} | Score: {score:5d} | "
          f"Epsilon: {epsilon:.3f} | Time: {ep_time:.1f}s"
      )

  with open("cnn_dqn_metrics.json", "w") as f:
      json.dump(metrics, f, indent=2)
  torch.save(policy_net.state_dict(), "cnn_dqn_model.pth")

  print(f"\nTraining complete. Metrics saved to cnn_dqn_metrics.json")
  print(f"Model saved to cnn_dqn_model.pth")

  env.close()
  return metrics


if __name__ == "__main__":
    train()
