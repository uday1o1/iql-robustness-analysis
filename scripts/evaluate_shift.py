# -*- coding: utf-8 -*-
"""Evaluate a trained IQL agent under controlled distribution shift.

Supports 4 shift types: gravity, obs_noise, friction, reward_perturb.

Usage:
    python scripts/evaluate_shift.py \
        --env_name=hopper-medium-v2 \
        --config=configs/mujoco_config.py \
        --shift_type=gravity \
        --shift_levels="0.5,1.0,1.5,2.0" \
        --num_critics=2

    # Run all shift types at once:
    python scripts/evaluate_shift.py --shift_type=all --num_critics=2
"""

import csv
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from absl import app, flags
from ml_collections import config_flags

import wrappers
from evaluation import evaluate
from iql import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'hopper-medium-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Directory with trained checkpoints.')
flags.DEFINE_string('output_dir', './results/', 'Directory for shift results.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_integer('num_critics', 2,
                     'Number of Q-networks (2=DoubleCritic, 3=TripleCritic).')
flags.DEFINE_integer('max_steps', int(1e6), 'Training steps (for model init).')
flags.DEFINE_string('shift_type', 'gravity',
                    'Type of shift: gravity, obs_noise, friction, reward_perturb, all.')
flags.DEFINE_string('shift_levels', '',
                    'Comma-separated shift levels (uses defaults if empty).')
config_flags.DEFINE_config_file(
    'config',
    'configs/mujoco_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

# Default shift configurations for each type
DEFAULT_SHIFT_LEVELS = {
    'gravity': [0.5, 1.0, 1.5, 2.0],
    'obs_noise': [0.0, 0.01, 0.1, 0.3],
    'friction': [0.5, 1.0, 1.5, 2.0],
    'reward_perturb': [0.0, 0.1, 0.5, 1.0],  # noise_std values
}

# Baseline level for each shift type (no shift)
BASELINE_LEVELS = {
    'gravity': 1.0,
    'obs_noise': 0.0,
    'friction': 1.0,
    'reward_perturb': 0.0,
}


def make_shifted_env(env_name, shift_type, shift_level, seed):
    """Create an environment with distribution shift applied."""
    from iql.dataset_utils import d4rl_to_gymnasium_name
    gym_env_name = d4rl_to_gymnasium_name(env_name)
    env = gym.make(gym_env_name)

    if shift_type == 'gravity':
        env = wrappers.GravityShift(env, gravity_scale=shift_level)
    elif shift_type == 'obs_noise':
        env = wrappers.ObservationNoise(env, noise_std=shift_level)
    elif shift_type == 'friction':
        env = wrappers.FrictionShift(env, friction_scale=shift_level)
    elif shift_type == 'reward_perturb':
        env = wrappers.RewardPerturbation(env, noise_std=shift_level)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def compute_robustness_drop(baseline_score, shifted_score):
    """Delta(d) = (J(pi, E_0) - J(pi, E_d)) / J(pi, E_0)"""
    if baseline_score == 0:
        return float('inf') if shifted_score != 0 else 0.0
    return (baseline_score - shifted_score) / abs(baseline_score)


def run_shift_evaluation(agent, env_name, shift_type, shift_levels,
                         eval_episodes, seed):
    """Run evaluation across all shift levels for a given shift type."""
    results = []
    for level in shift_levels:
        env = make_shifted_env(env_name, shift_type, level, seed)
        stats = evaluate(agent, env, eval_episodes)
        result = {
            'shift_type': shift_type,
            'shift_level': level,
            'raw_return': stats['return'],
            'episode_length': stats['length'],
        }
        results.append(result)
        print(f"  {shift_type}={level:.2f}: return={stats['return']:.2f}, "
              f"length={stats['length']:.0f}")
    return results


def main(_):
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # Determine which shift types to run
    if FLAGS.shift_type == 'all':
        shift_types = list(DEFAULT_SHIFT_LEVELS.keys())
    else:
        shift_types = [FLAGS.shift_type]

    # Create base environment for agent initialization
    from iql.dataset_utils import d4rl_to_gymnasium_name
    gym_env_name = d4rl_to_gymnasium_name(FLAGS.env_name)
    env = gym.make(gym_env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    env.reset(seed=FLAGS.seed)

    # Initialize agent
    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    num_critics=FLAGS.num_critics,
                    **kwargs)

    # Load checkpoint
    ckpt_dir = os.path.join(FLAGS.save_dir, f'checkpoint_{FLAGS.max_steps}')
    if os.path.exists(ckpt_dir):
        print(f"Loading checkpoint from {ckpt_dir}")
        agent.actor = agent.actor.load(os.path.join(ckpt_dir, 'actor.pkl'))
        agent.critic = agent.critic.load(os.path.join(ckpt_dir, 'critic.pkl'))
        agent.value = agent.value.load(os.path.join(ckpt_dir, 'value.pkl'))
        agent.target_critic = agent.target_critic.load(
            os.path.join(ckpt_dir, 'target_critic.pkl'))
    else:
        print(f"WARNING: No checkpoint found at {ckpt_dir}")
        print("Running evaluation with randomly initialized agent.")

    # Run evaluations
    all_results = []
    critic_label = f"{FLAGS.num_critics}Q"

    for shift_type in shift_types:
        # Parse shift levels
        if FLAGS.shift_levels:
            levels = [float(x) for x in FLAGS.shift_levels.split(',')]
        else:
            levels = DEFAULT_SHIFT_LEVELS[shift_type]

        print(f"\n{'='*60}")
        print(f"Evaluating {critic_label} under {shift_type} shift")
        print(f"Environment: {FLAGS.env_name}")
        print(f"Levels: {levels}")
        print(f"{'='*60}")

        results = run_shift_evaluation(
            agent, FLAGS.env_name, shift_type, levels,
            FLAGS.eval_episodes, FLAGS.seed)
        all_results.extend(results)

    # Compute robustness drop
    baseline_scores = {}
    for r in all_results:
        bl = BASELINE_LEVELS.get(r['shift_type'], 0)
        if r['shift_level'] == bl:
            baseline_scores[r['shift_type']] = r['raw_return']

    for r in all_results:
        baseline = baseline_scores.get(r['shift_type'], r['raw_return'])
        r['robustness_drop'] = compute_robustness_drop(baseline, r['raw_return'])

    # Save results — include expectile (tau) in filename when non-default
    tau = FLAGS.config.expectile
    tau_suffix = f'_tau{tau}' if tau != 0.7 else ''
    output_file = os.path.join(
        FLAGS.output_dir,
        f'shift_{FLAGS.env_name}_{critic_label}_seed{FLAGS.seed}{tau_suffix}.csv')

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'shift_type', 'shift_level', 'raw_return',
            'episode_length', 'robustness_drop'])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {FLAGS.env_name} ({critic_label})")
    print(f"{'='*60}")
    print(f"{'Shift Type':<18} {'Level':<10} {'Return':<12} {'Drop':<10}")
    print(f"{'-'*50}")
    for r in all_results:
        print(f"{r['shift_type']:<18} {r['shift_level']:<10.2f} "
              f"{r['raw_return']:<12.2f} {r['robustness_drop']:<10.4f}")


if __name__ == '__main__':
    app.run(main)
