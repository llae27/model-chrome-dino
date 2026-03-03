import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import gymnasium as gym
import gym_chrome_dino

from feature_wrapper import FeatureObservationWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ACTIONS = 3


class FeatureDQN(nn.Module):
    def __init__(self, input_dim=8, num_actions=NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
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
            torch.tensor(np.array(s), dtype=torch.float32).to(DEVICE),
            torch.tensor(a, dtype=torch.long).to(DEVICE),
            torch.tensor(r, dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(ns), dtype=torch.float32).to(DEVICE),
            torch.tensor(d, dtype=torch.float32).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


def train(
    batch_size=40,
    learning_rate=0.001,
    gamma=0.99,
    memory_size=10000,
    num_episodes=100,
    target_update_freq=10,
    epsilon_start=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.1,
):
    env = gym.make("ChromeDinoNoBrowser-v0")
    env = FeatureObservationWrapper(env)
    env.unwrapped.set_acceleration(True)

    policy_net = FeatureDQN().to(DEVICE)
    target_net = FeatureDQN().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())

    criterion = nn.MSELoss()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayBuffer(memory_size)

    def optimize_model():
        if len(memory) < batch_size:
            return
        s, a, r, ns, d = memory.sample(batch_size)
        q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = target_net(ns).max(1)[0]
            target_q = r + (1 - d) * gamma * next_q
        loss = criterion(q_values, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    metrics = {
        "rewards": [],
        "scores": [],
        "episode_times": [],
        "epsilons": [],
    }

    epsilon = epsilon_start

    for episode in range(num_episodes):
        ep_start = time.time()
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    action = policy_net(state_t).argmax().item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(memory) > batch_size:
                optimize_model()

        ep_time = time.time() - ep_start
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % target_update_freq == 0:
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

    with open("feature_dqn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(policy_net.state_dict(), "feature_dqn_model.pth")

    print(f"\nTraining complete. Metrics saved to feature_dqn_metrics.json")
    print(f"Model saved to feature_dqn_model.pth")

    env.close()
    return metrics


if __name__ == "__main__":
    train()
