# Atari CNN DQN
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

from gym_chrome_dino.envs.chrome_dino_env import ACTION_MEANING
from gym_chrome_dino.utils.atari_wrappers import make_atari, wrap_deepmind

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
  def __init__(self, in_channels, num_actions):
    # reference to https://arxiv.org/pdf/2008.06799
    super(DQN, self).__init__()
    self.net = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=10, stride=10),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=2, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(576, 512),
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
        torch.tensor(s, dtype=torch.float).to(DEVICE),
        torch.tensor(a, dtype=torch.long).to(DEVICE),
        torch.tensor(r, dtype=torch.float).to(DEVICE),
        torch.tensor(ns, dtype=torch.float).to(DEVICE),
        torch.tensor(d, dtype=torch.float).to(DEVICE),
    )
  def __len__(self):
    return len(self.buffer)

def train(
  batch_size: int = 40,
  learning_rate: float = 0.001,
  optimize_gamma: float = 0.99
  len_memory: int = 10000,
  num_episodes: int = 100,
  num_targe_update: int = 10,
):
  # create atari env
  env = make_atari("ChromeDinoNoFrameskip-v0")
  env = wrap_deepmind(env, episode_life=False, frame_stack=True, scale=True)
  
  # initialize
  NUM_ACTIONS = len(ACTION_MEANING)
  in_channels = state.shape[-1] # This should be 4 for frame_stack=True
  
  # training
  policy_net = DQN(in_channels, NUM_ACTIONS).to(DEVICE)
  target_net = DQN(in_channels, NUM_ACTIONS).to(DEVICE)
  
  criterion = nn.MSELoss()
  optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
  
  def optimize_model():
    if len(memory) < batch_size:
      return
    s, a, r, ns, d = memory.sample(batch_size)
    s = s.permute(0, 3, 1, 2)
    ns = ns.permute(0, 3, 1, 2)
    # q values
    q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
      next_q_values = target_net(ns)
      target_q_values = r + (1 - d) * optimize_gamma * next_q_values.max(1)[0]
    # compute loss
    loss = criterion(q_values, target_q_values)
    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  memory = ReplayBuffer(len_memory)
  epsilon = 1.0
  EPSILON_DECAY = 0.995
  EPSILON_MIN = 0.1
  for episode in range(num_episodes):
      done = False
      state, _ = env.reset()
      state = np.array(state)
      total_reward = 0
      while not done:
        if random.random() < epsilon:
          action = policy_net(torch.FloatTensor([state]).permute(0, 3, 1, 2).to(DEVICE)).argmax().item()
        else:
          state_tensor = torch.FloatTensor([state]).permute(0, 3, 1, 2).to(DEVICE)
          action = policy_net(state_tensor).argmax().item()
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_state) # Convert LazyFrames to numpy array for replay buffer
        done = terminated or truncated
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if len(memory) > batch_size:
          optimize_model()
      # episode done
      epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
      if episode % num_targe_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
