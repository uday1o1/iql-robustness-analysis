import gymnasium as gym
import numpy as np


class GravityWrapper(gym.Wrapper):
    """
    Wraps a MuJoCo environment and scales gravity by a given factor.
    Default MuJoCo gravity is -9.81 m/s^2 (along the z-axis).
    
    Args:
        env: The base Gym/D4RL environment.
        gravity_scale: Multiplier applied to the default gravity.
                       0.5x = low gravity, 1.0x = normal, 2.0x = high gravity.
    """
    def __init__(self, env, gravity_scale: float = 1.0):
        super().__init__(env)
        self.gravity_scale = gravity_scale
        # MuJoCo default gravity is [0, 0, -9.81]
        self._default_gravity = env.unwrapped.model.opt.gravity.copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Apply scaled gravity each reset
        self.env.unwrapped.model.opt.gravity[:] = (
            self._default_gravity * self.gravity_scale
        )
        return obs, info

    def step(self, action):
        return self.env.step(action)