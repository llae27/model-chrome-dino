import gymnasium as gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino
from gym_chrome_dino.envs.chrome_dino_env import ACTION_MEANING


def main():
    # initialize env
    env = gym.make("ChromeDinoNoBrowser-v0")
    env = make_dino(env, timer=True, frame_stack=True)
    env.unwrapped.set_acceleration(True)
    # env.unwrapped.game.set_parameter("config.ACCELERATION", 1)  # default value 0.001
    params = env.unwrapped.game.get_parameters()
    print(f"Current acceleration: {params['config.ACCELERATION']}")
    env.reset()
    # run experiment
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        # get pixel data
        frame = env.render()
        # get game data
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"{ACTION_MEANING[action]} {reward} / {info}", frame.shape)
    # report
    score = env.unwrapped.get_score()
    print(f"Final score: {score}")


if __name__ == "__main__":
    main()
