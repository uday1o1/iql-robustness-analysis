import time

import gymnasium as gym
import numpy as np

from wrappers.common import TimeStep


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        step_result = self.env.step(action)
        # Handle both gymnasium (5 values) and gym (4 values) APIs
        if len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            observation, reward, done, info = step_result

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if done:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self, 'get_normalized_score'):
                info['episode']['return'] = self.get_normalized_score(
                    info['episode']['return']) * 100.0

        return observation, reward, done, info

    def reset(self, **kwargs):
        self._reset_stats()
        result = self.env.reset(**kwargs)
        # gymnasium reset() returns (obs, info), gym returns just obs
        if isinstance(result, tuple):
            return result[0]
        return result