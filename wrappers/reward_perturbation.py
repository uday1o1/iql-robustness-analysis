"""Reward perturbation wrapper for MuJoCo environments.

Adds noise or scaling to rewards at evaluation time to simulate
reward distribution shift.

Usage:
    env = gym.make('hopper-medium-v2')
    env = RewardPerturbation(env, noise_std=0.1, scale=1.0)
"""

import gymnasium as gym
import numpy as np

from wrappers.common import TimeStep


class RewardPerturbation(gym.Wrapper):
    """Perturb rewards to simulate reward distribution shift.

    Applies: r' = scale * r + N(0, noise_std²)

    Args:
        env: A gym environment.
        noise_std: Standard deviation of Gaussian noise added to rewards.
        scale: Multiplicative scaling factor for rewards.
    """

    def __init__(self, env: gym.Env, noise_std: float = 0.0,
                 scale: float = 1.0):
        super().__init__(env)
        self.noise_std = noise_std
        self.scale = scale

    def step(self, action: np.ndarray):
        step_result = self.env.step(action)
        if len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
        else:
            observation, reward, done, info = step_result
            terminated = done
            truncated = False
        # Apply perturbation
        perturbed_reward = self.scale * reward
        if self.noise_std > 0:
            perturbed_reward += np.random.normal(0, self.noise_std)
        return observation, perturbed_reward, terminated, truncated, info
