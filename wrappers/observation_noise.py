"""Observation noise wrapper for MuJoCo environments.

Adds Gaussian noise to observations at evaluation time to simulate
sensor noise / observation distribution shift.

Usage:
    env = gym.make('hopper-medium-v2')
    env = ObservationNoise(env, noise_std=0.1)
"""

import gymnasium as gym
import numpy as np


class ObservationNoise(gym.ObservationWrapper):
    """Add Gaussian noise to observations to simulate sensor noise.

    Args:
        env: A gym environment.
        noise_std: Standard deviation of the Gaussian noise added
                   to each observation dimension.
    """

    def __init__(self, env: gym.Env, noise_std: float = 0.0):
        super().__init__(env)
        self.noise_std = noise_std

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if self.noise_std > 0:
            noise = np.random.normal(
                0, self.noise_std, size=observation.shape
            ).astype(observation.dtype)
            return observation + noise
        return observation
