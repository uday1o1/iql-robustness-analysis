import collections
import os
from typing import Optional

import numpy as np
from tqdm import tqdm

# Suppress D4RL import warnings for envs we don't use
os.environ.setdefault('D4RL_SUPPRESS_IMPORT_ERROR', '1')

try:
    import d4rl
    import gym
    _HAS_D4RL = True
except ImportError:
    _HAS_D4RL = False

try:
    import gymnasium
    _HAS_GYMNASIUM = True
except ImportError:
    _HAS_GYMNASIUM = False


# D4RL env names -> gymnasium MuJoCo env names
_D4RL_TO_GYMNASIUM = {
    'hopper': 'Hopper-v4',
    'halfcheetah': 'HalfCheetah-v4',
    'walker2d': 'Walker2d-v4',
    'ant': 'Ant-v4',
}


def d4rl_to_gymnasium_name(d4rl_name):
    """Convert D4RL env name to gymnasium MuJoCo env name.

    e.g., 'hopper-medium-v2' -> 'Hopper-v4'
    """
    base = d4rl_name.split('-')[0]
    return _D4RL_TO_GYMNASIUM.get(base, d4rl_name)

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])


def _d4rl_url_filename(env_name):
    """Convert D4RL env name to the URL filename used by D4RL servers.

    D4RL naming convention for v2 datasets:
      env name:     hopper-medium-v2       (all dashes)
      URL filename: hopper_medium-v2.hdf5  (underscore between env and dataset)

    The filename in the URL is: {env}_{dataset}-{version}.hdf5
    where the first dash (between env and dataset) becomes an underscore.
    """
    # Split: 'hopper-medium-v2' -> ['hopper', 'medium', 'v2']
    parts = env_name.split('-')
    if len(parts) >= 3 and parts[-1].startswith('v'):
        # e.g., hopper-medium-replay-v2 -> hopper_medium_replay-v2
        env_part = parts[0]
        version = parts[-1]
        dataset_parts = parts[1:-1]
        return f"{env_part}_{'_'.join(dataset_parts)}-{version}.hdf5"
    # Fallback: just replace first dash with underscore
    return env_name.replace('-', '_', 1) + '.hdf5'


def _load_from_hdf5(cache_path):
    """Load a D4RL dataset from a local HDF5 file."""
    import h5py

    with h5py.File(cache_path, 'r') as f:
        dataset = {
            'observations': f['observations'][:],
            'actions': f['actions'][:],
            'rewards': f['rewards'][:],
            'terminals': f['terminals'][:],
            'next_observations': f['observations'][1:],
        }
        # next_observations needs special handling for terminal states
        # Append the last observation to match array lengths
        dataset['next_observations'] = np.concatenate([
            dataset['next_observations'],
            dataset['observations'][-1:],
        ], axis=0)

    return dataset


def _load_d4rl_dataset(env_or_name):
    """Load a D4RL dataset, handling both gym.Env and string env names.

    Priority order:
      1. If the HDF5 file is already cached locally, load it directly
         (no internet needed — works on GPU nodes).
      2. If D4RL is installed, try its API (may trigger a download).
      3. Download the HDF5 file from the D4RL server as a last resort.
    """
    # Resolve the dataset name for cache lookup
    name = env_or_name if isinstance(env_or_name, str) else getattr(
        getattr(env_or_name, 'spec', None), 'id', str(env_or_name))

    # D4RL stores files using URL filename convention (underscores)
    url_filename = _d4rl_url_filename(name)
    cache_dir = os.path.join(os.path.expanduser('~'), '.d4rl', 'datasets')
    cache_path = os.path.join(cache_dir, url_filename)

    # 1. Prefer local cache — no internet, no D4RL import needed
    if os.path.exists(cache_path):
        print(f"Loading cached dataset: {cache_path}")
        return _load_from_hdf5(cache_path)

    # 2. Try D4RL API (may download internally)
    if _HAS_D4RL:
        try:
            if hasattr(env_or_name, 'get_dataset'):
                return d4rl.qlearning_dataset(env_or_name)
            else:
                env = gym.make(env_or_name)
                ds = d4rl.qlearning_dataset(env)
                env.close()
                return ds
        except Exception:
            pass

    # 3. Fallback: download the HDF5 file directly from D4RL URLs
    import urllib.request

    url = f"http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/{url_filename}"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading D4RL dataset: {url} ...")
    urllib.request.urlretrieve(url, cache_path)
    print(f"Saved to {cache_path}")

    return _load_from_hdf5(cache_path)


class D4RLDataset(Dataset):
    def __init__(self,
                 env_or_name,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = _load_d4rl_dataset(env_or_name)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


class ReplayBuffer(Dataset):
    def __init__(self, observation_space, action_dim: int, capacity: int):
        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0
        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert data only into empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)

        assert self.capacity >= num_samples, \
            'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
