"""Evaluate a trained IQL agent under controlled distribution shift.

This script loads a trained agent and evaluates it under gravity shift
and observation noise perturbations, computing normalized scores and
robustness metrics.

Usage:
    python evaluate_shift.py \
        --env_name=hopper-medium-v2 \
        --config=configs/mujoco_config.py \
        --shift_type=gravity \
        --shift_levels="0.5,1.0,1.5,2.0" \
        --num_critics=2 \
        --eval_episodes=10 \
        --seed=42
"""

import csv
import os
from typing import Dict, List, Tuple

import gym
import numpy as np
from absl import app, flags
from ml_collections import config_flags

import wrappers
from dataset_utils import D4RLDataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner

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
                    'Type of shift: gravity, obs_noise, or both.')
flags.DEFINE_string('shift_levels', '0.5,1.0,1.5,2.0',
                    'Comma-separated shift levels to evaluate.')
config_flags.DEFINE_config_file(
    'config',
    'configs/mujoco_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


# Default shift configurations
GRAVITY_LEVELS = [0.5, 1.0, 1.5, 2.0]
OBS_NOISE_LEVELS = [0.0, 0.01, 0.1, 0.3]


def make_shifted_env(env_name: str, shift_type: str, shift_level: float,
                     seed: int) -> gym.Env:
    """Create an environment with distribution shift applied."""
    env = gym.make(env_name)

    if shift_type == 'gravity':
        env = wrappers.GravityShift(env, gravity_scale=shift_level)
    elif shift_type == 'obs_noise':
        env = wrappers.ObservationNoise(env, noise_std=shift_level)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def compute_robustness_drop(baseline_score: float,
                            shifted_score: float) -> float:
    """Compute robustness drop metric Δ(δ).

    Δ(δ) = (J(π, E_0) - J(π, E_δ)) / J(π, E_0)

    A value close to 0 indicates robustness.
    Positive values indicate performance degradation.
    """
    if baseline_score == 0:
        return float('inf') if shifted_score != 0 else 0.0
    return (baseline_score - shifted_score) / abs(baseline_score)


def run_shift_evaluation(agent, env_name: str, shift_type: str,
                         shift_levels: List[float], eval_episodes: int,
                         seed: int) -> List[Dict]:
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

    # Parse shift levels
    shift_levels = [float(x) for x in FLAGS.shift_levels.split(',')]

    # Create base environment for agent initialization
    env = gym.make(FLAGS.env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    env.seed(FLAGS.seed)

    # Initialize agent (same architecture as training)
    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    num_critics=FLAGS.num_critics,
                    **kwargs)

    # Load checkpoint if available
    ckpt_dir = os.path.join(FLAGS.save_dir,
                            f'checkpoint_{FLAGS.max_steps}')
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
        print("Train first with: python train_offline.py ...")

    # Determine which shift types to run
    if FLAGS.shift_type == 'both':
        shift_configs = [
            ('gravity', GRAVITY_LEVELS),
            ('obs_noise', OBS_NOISE_LEVELS),
        ]
    else:
        shift_configs = [(FLAGS.shift_type, shift_levels)]

    # Run evaluations
    all_results = []
    critic_label = f"{FLAGS.num_critics}Q"

    for shift_type, levels in shift_configs:
        print(f"\n{'='*60}")
        print(f"Evaluating under {shift_type} shift ({critic_label})")
        print(f"Environment: {FLAGS.env_name}")
        print(f"{'='*60}")

        results = run_shift_evaluation(
            agent, FLAGS.env_name, shift_type, levels,
            FLAGS.eval_episodes, FLAGS.seed)
        all_results.extend(results)

    # Find baseline score (no shift: gravity=1.0 or obs_noise=0.0)
    baseline_scores = {}
    for r in all_results:
        if (r['shift_type'] == 'gravity' and r['shift_level'] == 1.0) or \
           (r['shift_type'] == 'obs_noise' and r['shift_level'] == 0.0):
            baseline_scores[r['shift_type']] = r['raw_return']

    # Compute robustness drop
    for r in all_results:
        baseline = baseline_scores.get(r['shift_type'], r['raw_return'])
        r['robustness_drop'] = compute_robustness_drop(
            baseline, r['raw_return'])

    # Save results to CSV
    output_file = os.path.join(
        FLAGS.output_dir,
        f'shift_results_{FLAGS.env_name}_{critic_label}_seed{FLAGS.seed}.csv')

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'shift_type', 'shift_level', 'raw_return',
            'episode_length', 'robustness_drop'])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {output_file}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"ROBUSTNESS SUMMARY — {FLAGS.env_name} ({critic_label})")
    print(f"{'='*60}")
    print(f"{'Shift Type':<15} {'Level':<10} {'Return':<12} {'Δ(δ)':<10}")
    print(f"{'-'*47}")
    for r in all_results:
        print(f"{r['shift_type']:<15} {r['shift_level']:<10.2f} "
              f"{r['raw_return']:<12.2f} {r['robustness_drop']:<10.4f}")


if __name__ == '__main__':
    app.run(main)
