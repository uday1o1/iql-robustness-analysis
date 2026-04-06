"""Friction shift wrapper for MuJoCo environments.

Scales friction coefficients at evaluation time to simulate
distribution shift in contact dynamics.

Usage:
    env = gym.make('hopper-medium-v2')
    env = FrictionShift(env, friction_scale=2.0)  # 2x friction
"""

import gymnasium as gym
import numpy as np


class FrictionShift(gym.Wrapper):
    """Scale MuJoCo friction to simulate contact dynamics shift.

    Args:
        env: A MuJoCo gym environment.
        friction_scale: Multiplier for friction coefficients.
                        1.0 = no change, 0.5 = slippery, 2.0 = sticky.
    """

    def __init__(self, env: gym.Env, friction_scale: float = 1.0):
        super().__init__(env)
        self.friction_scale = friction_scale
        self._default_friction = None

    def reset(self, **kwargs) -> np.ndarray:
        obs = self.env.reset(**kwargs)
        if self._default_friction is None:
            self._default_friction = self.env.unwrapped.model.geom_friction.copy()
        self.env.unwrapped.model.geom_friction[:] = (
            self._default_friction * self.friction_scale
        )
        return obs
