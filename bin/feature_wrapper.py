import numpy as np
import gymnasium as gym
from gymnasium import spaces


FEATURE_JS = """
var runner = Runner.instance_;
var tRex = runner.tRex;
var obstacles = runner.horizon.obstacles;
var speed = runner.currentSpeed;

var ahead = [];
for (var i = 0; i < obstacles.length; i++) {
    if (obstacles[i].xPos > tRex.xPos) {
        ahead.push(obstacles[i]);
    }
}

var dist1 = 600, y1 = 0, w1 = 0, h1 = 0, dist2 = 600;

if (ahead.length >= 1) {
    dist1 = ahead[0].xPos - tRex.xPos;
    y1 = ahead[0].yPos;
    w1 = ahead[0].typeConfig.width;
    h1 = ahead[0].typeConfig.height;
}
if (ahead.length >= 2) {
    dist2 = ahead[1].xPos - tRex.xPos;
}

return [dist1, y1, w1, h1, dist2, tRex.yPos, speed, tRex.jumping ? 1 : 0];
"""

NORM_DIVISORS = np.array([600.0, 150.0, 100.0, 100.0, 600.0, 150.0, 20.0, 1.0], dtype=np.float32)


class FeatureObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

    def observation(self, pixel_obs):
        try:
            driver = self.unwrapped.game.driver
            raw = driver.execute_script(FEATURE_JS)
            features = np.array(raw, dtype=np.float32) / NORM_DIVISORS
            return np.clip(features, 0.0, 1.0)
        except Exception:
            return np.zeros(8, dtype=np.float32)


if __name__ == "__main__":
    import gym_chrome_dino

    env = gym.make("ChromeDinoNoBrowser-v0")
    env = FeatureObservationWrapper(env)

    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    print(f"Initial features:  {obs}")

    for step in range(50):
        action = np.random.randint(0, 3)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step:3d} | action={action} reward={reward:+.1f} | features={obs}")
        if terminated:
            print("Game over!")
            break

    score = env.unwrapped.get_score()
    print(f"Final score: {score}")
    env.close()
