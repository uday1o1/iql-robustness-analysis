from typing import Tuple

import jax.numpy as jnp

from iql.common import Batch, InfoDict, Model, Params


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def _min_q(critic_output):
    """Take element-wise min across all Q-networks (supports 2 or 3)."""
    if len(critic_output) == 2:
        return jnp.minimum(critic_output[0], critic_output[1])
    elif len(critic_output) == 3:
        return jnp.minimum(jnp.minimum(critic_output[0], critic_output[1]),
                           critic_output[2])
    else:
        raise ValueError(f"Expected 2 or 3 Q-networks, got {len(critic_output)}")


def update_v(critic: Model, value: Model, batch: Batch,
             expectile: float) -> Tuple[Model, InfoDict]:
    actions = batch.actions
    critic_output = critic(batch.observations, actions)
    q = _min_q(critic_output)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, batch.observations)
        value_loss = loss(q - v, expectile).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def update_q(critic: Model, target_value: Model, batch: Batch,
             discount: float) -> Tuple[Model, InfoDict]:
    next_v = target_value(batch.next_observations)

    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        critic_output = critic.apply({'params': critic_params},
                                     batch.observations, batch.actions)
        # Sum MSE loss across all Q-networks
        critic_loss = sum(
            (q - target_q)**2 for q in critic_output
        ).mean()
        info = {'critic_loss': critic_loss}
        for i, q in enumerate(critic_output):
            info[f'q{i+1}'] = q.mean()
        return critic_loss, info

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
