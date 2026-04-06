"""IQL — Implicit Q-Learning core implementation.

Based on Kostrikov et al., "Offline Reinforcement Learning with
Implicit Q-Learning", ICLR 2022.
"""

from iql.learner import Learner
from iql.common import Batch, Model

__all__ = ['Learner', 'Batch', 'Model']
