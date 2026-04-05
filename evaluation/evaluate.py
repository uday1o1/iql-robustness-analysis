from typing import Dict

import flax.linen as nn
import gymnasium as gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        observation = env.reset()
        # gymnasium reset() returns (obs, info) — handle both APIs
        if isinstance(observation, tuple):
            observation = observation[0]
        done = False

        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            step_result = env.step(action)
            # gymnasium step() returns (obs, reward, terminated, truncated, info)
            # gym step() returns (obs, reward, done, info)
            if len(step_result) == 5:
                observation, _, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                observation, _, done, info = step_result

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
