# -*- coding: utf-8 -*-
"""Train IQL offline on D4RL datasets.

Usage:
    python scripts/train_offline.py \
        --env_name=hopper-medium-v2 \
        --config=configs/mujoco_config.py \
        --num_critics=2
"""

import os
import sys
from typing import Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers
from iql.dataset_utils import D4RLDataset, split_into_trajectories, d4rl_to_gymnasium_name
from evaluation import evaluate
from iql import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('num_critics', 2,
                     'Number of Q-networks (2=DoubleCritic, 3=TripleCritic).')
flags.DEFINE_integer('save_interval', 0,
                     'Checkpoint save interval (0=save only final).')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
config_flags.DEFINE_config_file(
    'config',
    'configs/mujoco_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def normalize(dataset):
    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew
        return episode_return

    trajs.sort(key=compute_returns)
    ret_range = compute_returns(trajs[-1]) - compute_returns(trajs[0])
    if abs(ret_range) < 1e-8:
        # All trajectories have the same return — skip normalization
        print(f"WARNING: reward range is ~0 ({ret_range:.6f}), skipping normalization")
    else:
        dataset.rewards /= ret_range
        dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    gym_env_name = d4rl_to_gymnasium_name(env_name)
    env = gym.make(gym_env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    # gymnasium uses reset(seed=) instead of env.seed()
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Load D4RL dataset by name (doesn't need mujoco_py)
    dataset = D4RLDataset(env_name)

    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(dataset)

    return env, dataset


def main(_):
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb',
                                                str(FLAGS.seed)),
                                   write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    num_critics=FLAGS.num_critics,
                    **kwargs)

    eval_returns = []
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    # Skip histogram if values contain NaN/Inf
                    import numpy as _np
                    v_np = _np.asarray(v)
                    if _np.isfinite(v_np).any() and v_np.size > 0:
                        summary_writer.add_histogram(f'training/{k}', v_np[_np.isfinite(v_np)], i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])

            # Save checkpoint
            if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
                agent.save(FLAGS.save_dir, i)

    # Always save final checkpoint
    agent.save(FLAGS.save_dir, FLAGS.max_steps)


if __name__ == '__main__':
    app.run(main)
