# -*- coding: utf-8 -*-
"""Validate the full IQL pipeline before submitting HPC jobs.

Tests every component: imports, wrappers, JAX JIT, TFP distributions,
Learner initialization (2Q and 3Q), evaluation loop, and action sampling.

Usage:
    python scripts/validate_pipeline.py

Run this after setup to catch ALL issues before submitting batch jobs.
"""

import sys
import os
import warnings

# Setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

PASS = 0
FAIL = 0


def step(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f'  PASS  {name}')
        PASS += 1
    except Exception as e:
        print(f'  FAIL  {name}: {e}')
        FAIL += 1


def test_core_imports():
    import numpy as np
    import jax
    from tensorflow_probability.substrates import jax as tfp
    import gymnasium as gym
    import mujoco
    devices = jax.devices()
    gpu_count = len([d for d in devices if d.platform == 'gpu'])
    backend = f'GPU x{gpu_count}' if gpu_count > 0 else 'CPU only'
    print(f'         numpy={np.__version__}, jax={jax.__version__}, mujoco={mujoco.__version__}, backend={backend}')


def test_cuda_jaxlib():
    """Check if CUDA-enabled jaxlib is installed for GPU acceleration.

    Uses pip metadata (not jaxlib.__version__) because the Python version
    string doesn't include the +cuda suffix even for CUDA builds.
    """
    import jaxlib
    import jax
    import subprocess

    # Get version from pip metadata (includes +cuda suffix)
    try:
        result = subprocess.run(
            ['pip', 'show', 'jaxlib'],
            capture_output=True, text=True, timeout=10
        )
        pip_version = ''
        for line in result.stdout.splitlines():
            if line.lower().startswith('version:'):
                pip_version = line.split(':', 1)[1].strip()
                break
    except Exception:
        pip_version = jaxlib.__version__

    has_cuda = 'cuda' in pip_version.lower()
    devices = jax.devices()
    gpu_devs = [d for d in devices if d.platform == 'gpu']

    print(f'         jaxlib (pip): {pip_version}')
    print(f'         jaxlib (python): {jaxlib.__version__}')
    print(f'         CUDA build: {has_cuda}')
    print(f'         GPU devices: {len(gpu_devs)}')

    if gpu_devs:
        for d in gpu_devs:
            print(f'         -> {d}')
    elif has_cuda:
        print('         NOTE: CUDA jaxlib installed but no GPU on this node (login node).')
        print('         GPU will be used automatically on batch GPU nodes.')
    else:
        print('         WARNING: CPU-only jaxlib. Training will be ~15x slower.')
        print('         This is OK — training still works, just slower (~10h vs ~2h).')


def test_iql_imports():
    from iql.learner import Learner
    from iql.policy import sample_actions
    from iql.dataset_utils import d4rl_to_gymnasium_name
    from iql.common import Model
    from iql.value_net import DoubleCritic, TripleCritic


def test_wrappers():
    import gymnasium as gym
    import wrappers
    for name, cls, kw in [
        ('gravity', wrappers.GravityShift, {'gravity_scale': 2.0}),
        ('obs_noise', wrappers.ObservationNoise, {'noise_std': 0.1}),
        ('friction', wrappers.FrictionShift, {'friction_scale': 1.5}),
        ('reward', wrappers.RewardPerturbation, {'noise_std': 0.1}),
    ]:
        e = gym.make('Hopper-v4')
        e = cls(e, **kw)
        e = wrappers.EpisodeMonitor(e)
        e = wrappers.SinglePrecision(e)
        obs, info = e.reset(seed=42)
        obs, r, t, tr, info = e.step(e.action_space.sample())
        assert obs.shape == (11,), f'{name}: wrong obs shape {obs.shape}'
        assert obs.dtype.name == 'float32', f'{name}: wrong dtype {obs.dtype}'
        e.close()


def test_jax_tfp():
    import jax
    import jax.numpy as jnp
    from tensorflow_probability.substrates import jax as tfp

    @jax.jit
    def f(x):
        return x * 2
    assert float(f(jnp.array(3.0))) == 6.0

    s = tfp.distributions.Normal(0., 1.).sample(seed=jax.random.PRNGKey(0))
    assert abs(float(s)) < 10  # sanity check


def test_learner_2q():
    import gymnasium as gym
    import wrappers
    from iql.learner import Learner

    env = gym.make('Hopper-v4')
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    obs, _ = env.reset(seed=42)

    agent = Learner(
        seed=42, observations=obs, actions=env.action_space.sample(),
        num_critics=2, actor_lr=3e-4, value_lr=3e-4, critic_lr=3e-4,
        hidden_dims=(256, 256), discount=0.99, tau=0.005,
        expectile=0.7, temperature=3.0, dropout_rate=None,
        opt_decay_schedule='cosine', max_steps=300000,
    )
    env.close()
    return agent, obs


def test_learner_3q():
    import gymnasium as gym
    import wrappers
    from iql.learner import Learner

    env = gym.make('Hopper-v4')
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    obs, _ = env.reset(seed=42)

    Learner(
        seed=42, observations=obs, actions=env.action_space.sample(),
        num_critics=3, actor_lr=3e-4, value_lr=3e-4, critic_lr=3e-4,
        hidden_dims=(256, 256), discount=0.99, tau=0.005,
        expectile=0.7, temperature=3.0, dropout_rate=None,
        opt_decay_schedule='cosine', max_steps=300000,
    )
    env.close()


def test_evaluation():
    import gymnasium as gym
    import wrappers
    from evaluation import evaluate

    agent, _ = test_learner_2q()
    env = gym.make('Hopper-v4')
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    stats = evaluate(agent, env, num_episodes=1)
    assert 'return' in stats
    assert 'length' in stats
    print(f'         return={stats["return"]:.2f}, length={stats["length"]:.0f}')
    env.close()


def test_sample_actions():
    import jax
    from iql.policy import sample_actions

    agent, obs = test_learner_2q()
    rng, actions = sample_actions(
        jax.random.PRNGKey(0), agent.actor.apply_fn, agent.actor.params, obs
    )
    assert actions.shape == (3,), f'Wrong action shape: {actions.shape}'


def test_datasets_cached():
    """Check that D4RL datasets are downloaded and cached.

    D4RL v2 filename convention:
      env name:     hopper-medium-v2
      cache file:   hopper_medium-v2.hdf5  (underscore between env and dataset)
    """
    # Import the helper to get the correct filename
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from iql.dataset_utils import _d4rl_url_filename

    cache_dir = os.path.join(os.path.expanduser('~'), '.d4rl', 'datasets')
    missing = []
    for env_name in ['hopper-medium-v2', 'halfcheetah-medium-v2', 'walker2d-medium-v2']:
        filename = _d4rl_url_filename(env_name)
        path = os.path.join(cache_dir, filename)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f'         {env_name} -> {filename}: {size_mb:.1f} MB')
        else:
            missing.append(f'{env_name} ({filename})')
    if missing:
        raise FileNotFoundError(
            f'Missing datasets: {missing}. Run iql-setup on the login node first.'
        )


if __name__ == '__main__':
    print('=' * 60)
    print('IQL Pipeline Validation')
    print('=' * 60)
    print(f'Python: {sys.version.split()[0]}')
    print()

    step('Core imports (numpy, jax, tfp, gymnasium, mujoco)', test_core_imports)
    step('CUDA jaxlib installed (GPU acceleration)', test_cuda_jaxlib)
    step('IQL package imports', test_iql_imports)
    step('All 4 shift wrappers', test_wrappers)
    step('JAX JIT + TFP distributions', test_jax_tfp)
    step('Learner init (2Q DoubleCritic)', test_learner_2q)
    step('Learner init (3Q TripleCritic)', test_learner_3q)
    step('Evaluation loop (1 episode)', test_evaluation)
    step('Sample actions from policy', test_sample_actions)
    step('D4RL datasets cached locally', test_datasets_cached)

    print()
    print('=' * 60)
    print(f'Results: {PASS} passed, {FAIL} failed')
    if FAIL > 0:
        print('Fix failures before submitting batch jobs.')
        sys.exit(1)
    else:
        print('Pipeline validated. Ready to submit:')
        print('  sbatch scripts/run_all_hpc.sh')
    print('=' * 60)
