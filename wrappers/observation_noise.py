"""Observation noise wrapper for MuJoCo environments.

Adds Gaussian noise to observations at evaluation time to simulate
sensor noise / observation distribution shift. The policy is NOT
retrained — this tests robustness to noisy sensor readings.

Usage:
    env = gymnasium.make('Hopper-v4')
    env = ObservationNoise(env, noise_std=0.1)
"""

import gymnasium as gym
import numpy as np


class ObservationNoise(gym.ObservationWrapper):
    """Add Gaussian noise to observations to simulate sensor noise.

    Args:
        env: A gymnasium environment.
        noise_std: Standard deviation of the Gaussian noise added to each
                   observation dimension. 0.0 = no noise (baseline).
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
