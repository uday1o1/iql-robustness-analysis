"""Gravity shift wrapper for MuJoCo environments.

Scales the gravity vector at evaluation time to simulate distribution shift
in environment dynamics. The policy is NOT retrained — this tests robustness.

Usage:
    env = gym.make('hopper-medium-v2')
    env = GravityShift(env, gravity_scale=2.0)  # 2x gravity
"""

import gymnasium as gym
import numpy as np


class GravityShift(gym.Wrapper):
    """Scale MuJoCo gravity to simulate dynamics distribution shift.

    Args:
        env: A MuJoCo gym environment.
        gravity_scale: Multiplier for gravity. 1.0 = no change,
                       0.5 = half gravity, 2.0 = double gravity.
    """

    def __init__(self, env: gym.Env, gravity_scale: float = 1.0):
        super().__init__(env)
        self.gravity_scale = gravity_scale
        self._default_gravity = None

    def reset(self, **kwargs) -> np.ndarray:
        obs = self.env.reset(**kwargs)
        # Store default gravity on first reset
        if self._default_gravity is None:
            self._default_gravity = self.env.unwrapped.model.opt.gravity.copy()
        # Apply gravity scaling
        self.env.unwrapped.model.opt.gravity[:] = (
            self._default_gravity * self.gravity_scale
        )
        return obs
