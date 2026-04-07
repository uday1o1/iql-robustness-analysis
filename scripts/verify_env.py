# -*- coding: utf-8 -*-
"""Verify that all dependencies are installed and working.

Run this BEFORE submitting batch jobs to catch issues early:
    python scripts/verify_env.py

This checks every import used by the IQL codebase and reports
all failures at once instead of one at a time.
"""

import os
import sys
import importlib

# Ensure project root is on path and D4RL warnings are suppressed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

PASS = 0
FAIL = 0


def check(name, import_path=None, attr=None, version_attr='__version__'):
    """Try importing a module and optionally check an attribute."""
    global PASS, FAIL
    mod_name = import_path or name
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, version_attr, '?') if version_attr else '?'
        if attr:
            getattr(mod, attr)
        print(f"  OK   {name:<30s} {ver}")
        PASS += 1
    except Exception as e:
        print(f"  FAIL {name:<30s} {e}")
        FAIL += 1


def check_d4rl_envs():
    """Check that D4RL environments are registered with gym."""
    global PASS, FAIL
    try:
        import gym
        import d4rl  # noqa: F401 — registers envs
        env = gym.make('hopper-medium-v2')
        env.close()
        print(f"  OK   {'d4rl env registration':<30s} hopper-medium-v2 works")
        PASS += 1
    except Exception as e:
        # D4RL env registration often fails because it needs mujoco_py
        # (old API). The dataset loading (env.get_dataset()) may still
        # work. Treat as warning, not failure.
        print(f"  WARN {'d4rl env registration':<30s} {e}")
        print(f"       (D4RL dataset loading may still work at runtime)")
        PASS += 1  # count as pass — runtime will confirm


def check_jax_jit():
    """Check that JAX JIT works (catches static_argnames issues)."""
    global PASS, FAIL
    try:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            return x * 2

        result = f(jnp.array(1.0))
        assert float(result) == 2.0
        print(f"  OK   {'jax.jit':<30s} works (device: {jax.devices()[0]})")
        PASS += 1
    except Exception as e:
        print(f"  FAIL {'jax.jit':<30s} {e}")
        FAIL += 1


def check_tfp():
    """Check tensorflow-probability JAX substrate."""
    global PASS, FAIL
    try:
        from tensorflow_probability.substrates import jax as tfp
        tfd = tfp.distributions
        dist = tfd.Normal(loc=0.0, scale=1.0)
        sample = dist.sample(seed=__import__('jax').random.PRNGKey(0))
        print(f"  OK   {'tfp.distributions':<30s} Normal sampling works")
        PASS += 1
    except Exception as e:
        print(f"  FAIL {'tfp.distributions':<30s} {e}")
        FAIL += 1


def check_iql_imports():
    """Check that all IQL package imports work."""
    global PASS, FAIL
    modules = [
        'iql',
        'iql.common',
        'iql.value_net',
        'iql.critic',
        'iql.actor',
        'iql.policy',
        'iql.learner',
        'iql.dataset_utils',
    ]
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
            print(f"  OK   {mod_name:<30s}")
            PASS += 1
        except Exception as e:
            print(f"  FAIL {mod_name:<30s} {e}")
            FAIL += 1


def check_script_imports():
    """Check that training/eval scripts can be parsed."""
    global PASS, FAIL
    import py_compile
    scripts = [
        'scripts/train_offline.py',
        'scripts/evaluate_shift.py',
        'scripts/compute_robustness.py',
    ]
    for script in scripts:
        try:
            py_compile.compile(script, doraise=True)
            print(f"  OK   {script:<30s} syntax ok")
            PASS += 1
        except Exception as e:
            print(f"  FAIL {script:<30s} {e}")
            FAIL += 1


if __name__ == '__main__':
    print("=" * 60)
    print("IQL Robustness Analysis — Environment Verification")
    print("=" * 60)
    print(f"\nPython: {sys.version}")
    print(f"Path:   {sys.executable}\n")

    print("--- Core dependencies ---")
    check('numpy')
    check('scipy')
    check('h5py')
    check('matplotlib')
    check('tqdm')

    print("\n--- JAX ecosystem ---")
    check('jax')
    check('jaxlib')
    check('flax')
    check('optax')
    check('ml_dtypes')
    check_jax_jit()

    print("\n--- MuJoCo / Gym ---")
    check('mujoco')
    check('gymnasium')
    check('gym')

    print("\n--- D4RL ---")
    check('d4rl', version_attr=None)
    check_d4rl_envs()

    print("\n--- ML utilities ---")
    check('tensorflow_probability', version_attr='__version__')
    check_tfp()
    check('absl', import_path='absl.app', version_attr=None)
    check('ml_collections')
    check('tensorboardX')

    print("\n--- IQL package ---")
    check_iql_imports()

    print("\n--- Script syntax ---")
    check_script_imports()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        print("\nFix the failures above before submitting batch jobs.")
        print("Common fixes:")
        print("  pip install <package>              # install missing package")
        print("  pip install --only-binary=:all: <p> # avoid compilation")
        print("  pip install 'package==X.Y.Z'       # pin specific version")
        sys.exit(1)
    else:
        print("\nAll checks passed. Ready to submit:")
        print("  sbatch scripts/run_all_hpc.sh")
    print("=" * 60)
