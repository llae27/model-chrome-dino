import gymnasium as gym
import gym_chrome_dino


def main():
    env = gym.make("ChromeDinoNoBrowser-v0")
    # env = gym.make("ChromeDino-v0")
    print(env)


if __name__ == "__main__":
    main()
