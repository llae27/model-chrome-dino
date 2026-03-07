# Atari CNN DQN
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pathlib import Path
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
            torch.tensor(np.array(s), dtype=torch.float).to(DEVICE),
            torch.tensor(np.array(a), dtype=torch.long).to(DEVICE),
            torch.tensor(np.array(r), dtype=torch.float).to(DEVICE),
            torch.tensor(np.array(ns), dtype=torch.float).to(DEVICE),
            torch.tensor(np.array(d), dtype=torch.float).to(DEVICE),
        )
    def __len__(self):
        return len(self.buffer)

def train(
    batch_size: int = 40,
    learning_rate: float = 0.001,
    optimize_gamma: float = 0.99,
    num_steps: int = 500_000,
    num_target_update: int = 5000,
    len_memory: int = 10000,
    resume_path: str = None,
):
    # create atari env
    env = make_atari("ChromeDinoNoBrowser-v0")
    env = wrap_deepmind(env, episode_life=False, frame_stack=True, scale=True)
    
    # initialize
    NUM_ACTIONS = len(ACTION_MEANING)
    EPSILON_DECAY = 0.99997
    EPSILON_MIN = 0.1
    
    # models
    state, _ = env.reset()
    in_channels = np.array(state).shape[-1] # 4 if frame_stack=True
    policy_net = DQN(in_channels, NUM_ACTIONS).to(DEVICE)
    target_net = DQN(in_channels, NUM_ACTIONS).to(DEVICE)
    criterion = nn.SmoothL1Loss() # DQN loss
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    # resume if needed
    if resume_path is not None:
        checkpoint = torch.load(resume_path)
        policy_net.load_state_dict(checkpoint["policy"])
        target_net.load_state_dict(checkpoint["target"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epsilon = checkpoint["epsilon"]
        step = checkpoint["step"]
    else:
        epsilon = 1.0
        step = 0
    
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
        # add stability
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
        optimizer.step()
    
    memory = ReplayBuffer(len_memory)
    episode = 0
    # training
    latest_rewards = deque(maxlen=100)
    while step < num_steps:
        done = False
        state, _ = env.reset()
        state_array = np.array(state)
        total_reward = 0
        while not done:
            if random.random() < epsilon:
                # Explore
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state_array).unsqueeze(0).permute(0, 3, 1, 2).to(DEVICE)
                    action = policy_net(state_tensor).argmax().item()
            # memorize
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state_array = np.array(next_state)
            done = terminated or truncated
            memory.push(state_array, action, reward, next_state_array, done)
            # update
            state_array = next_state_array
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            step += 1
            total_reward += reward
            if step > num_target_update:
                optimize_model()
            if step % num_target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
            # save
            if step % 10000 == 0:
                torch.save({
                    "policy": policy_net.state_dict(),
                    "target": target_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epsilon": epsilon,
                    "step": step,
                }, f"data/dqn_checkpoint_step_{step}.pt")
                # torch.save(policy_net.state_dict(), f"data/dqn_chrome_dino_step_{step}.pth")
        # episode done
        latest_rewards.append(total_reward)
        # print(f"Episode: {episode}")
        if episode > 0 and episode % 100 == 0:
            with torch.no_grad():
                s, _, _, _, _ = memory.sample(batch_size)
                s = s.permute(0, 3, 1, 2)
                q_value = policy_net(s).max(1)[0].mean().item()
            print(f"Episode: {episode}, Steps: {step}, avg. Reward: {sum(latest_rewards) / 100:.2f}, avg. Q-value: {q_value:.2f}, Epsilon: {epsilon:.2f}")
        episode += 1
    return env, policy_net

def resume(save_path: Path):
    print(f"Resuming training from {save_path}...")
    env, policy_net = train(resume_path=save_path)
    return env, policy_net

def evaluate_model(env, policy_net, episodes=5):
    policy_net.eval()  # set to evaluation mode
    history_rewards = []
    history_scores = []
    for ep in range(episodes):
        state, _ = env.reset()
        state_array = np.array(state)
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.from_numpy(state_array).unsqueeze(0).permute(0, 3, 1, 2).to(DEVICE)
            with torch.no_grad():
                action = policy_net(state_tensor).argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # step
            state_array = np.array(next_state)
            episode_reward += reward
        # episode done
        history_rewards.append(episode_reward)
        score = env.unwrapped.get_score()
        history_scores.append(score)
        print(f"Eval Episode {ep}, Reward: {episode_reward}, Score: {score}")
    # eval done
    policy_net.train()
    print(f"\nAverage Reward: {np.mean(history_rewards)}")
    print(f"Average Score: {np.mean(history_scores)}")

def main():
    # env, policy_net = train()
    env, policy_net = resume("data/dqn_checkpoint_step_100000.pt")
    # Save final model
    torch.save(policy_net.state_dict(), "data/dqn_chrome_dino.pth")
    print("Model saved!")
    # evaluate
    evaluate_model(env, policy_net)

if __name__ == "__main__":
    main()
