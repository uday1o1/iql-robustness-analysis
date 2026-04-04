# Check-In 3 Brainstorm — IQL Robustness Analysis

## Status Summary (What's Done)

| Component | Owner | Status | Details |
|---|---|---|---|
| Baseline IQL on hopper-medium-v2 | Joao | ✅ Done | Score: 52.79 at 300k steps |
| Q-Ensemble (TripleCritic) on hopper-medium-v2 | Uday | ✅ Done | Score: 50.88 at 300k steps |
| Literature survey (IQL, CQL, TD3+BC) | Pramod | ✅ Done | In Check-in 2 report |
| Robustness metric definition (Δ(δ)) | Pramod | ✅ Done | Formal definition in report |
| Gravity/noise wrapper design | Shloak | 🔶 Designed | Not yet committed to repo |
| Baseline on halfcheetah/walker2d | Joao | ❌ Pending | Needed for Check-in 3 |
| Shift evaluation runs | All | ❌ Pending | Core deliverable for Check-in 3 |
| Robustness results table | Pramod | ❌ Pending | Depends on shift runs |

---

## Check-In 3 Deliverables (from the doc)

1. **Joao** — Run baseline IQL on `halfcheetah-medium-v2` and `walker2d-medium-v2`, then run through gravity shift
2. **Shloak** — Commit gravity + observation noise wrappers, run baseline IQL through shifts on `hopper-medium-v2`
3. **Pramod** — Compute Δ(δ) robustness drop metric, build results table, update evaluation section
4. **Uday** — Run TripleCritic through gravity/noise on `hopper-medium-v2`, build DoubleCritic vs TripleCritic comparison

---

## What Needs to Be Built (Code Gaps)

### 1. Distribution Shift Wrappers (Shloak's task, but we can help)

The repo currently has **no shift wrappers** committed. We need:

```python
# wrappers/gravity_shift.py
class GravityShift(gym.Wrapper):
    """Scale MuJoCo gravity at eval time."""
    def __init__(self, env, gravity_scale=1.0):
        super().__init__(env)
        # MuJoCo gravity is model.opt.gravity (default [0, 0, -9.81])
        self.gravity_scale = gravity_scale
    
    def reset(self):
        obs = self.env.reset()
        self.env.unwrapped.model.opt.gravity[:] = [0, 0, -9.81 * self.gravity_scale]
        return obs
```

```python
# wrappers/observation_noise.py
class ObservationNoise(gym.ObservationWrapper):
    """Add Gaussian noise to observations at eval time."""
    def __init__(self, env, noise_std=0.0):
        super().__init__(env)
        self.noise_std = noise_std
    
    def observation(self, obs):
        return obs + np.random.normal(0, self.noise_std, size=obs.shape).astype(np.float32)
```

**Shift levels planned:**
- Gravity: `0.5x, 1.0x, 1.5x, 2.0x`
- Observation noise: `σ = 0.01, 0.1, 0.3`

### 2. Evaluation Under Shift Script

The current `evaluation.py` only evaluates in the original environment. We need a script that:
- Loads a trained agent checkpoint
- Wraps the environment with shift wrappers
- Runs evaluation episodes
- Computes normalized score and Δ(δ)

```python
# evaluate_shift.py
def evaluate_under_shift(agent, env_name, shift_type, shift_level, num_episodes=10, seed=42):
    env = gym.make(env_name)
    if shift_type == 'gravity':
        env = GravityShift(env, gravity_scale=shift_level)
    elif shift_type == 'obs_noise':
        env = ObservationNoise(env, noise_std=shift_level)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    stats = evaluate(agent, env, num_episodes)
    return stats
```

### 3. Model Checkpointing

The current `train_offline.py` saves eval returns to a text file but **does not save model checkpoints**. The `Model` class in `common.py` has `save()` and `load()` methods but they're never called. We need to add:

```python
# In train_offline.py, after eval:
if i % FLAGS.eval_interval == 0:
    agent.actor.save(os.path.join(FLAGS.save_dir, f'actor_{i}.pkl'))
    agent.critic.save(os.path.join(FLAGS.save_dir, f'critic_{i}.pkl'))
    agent.value.save(os.path.join(FLAGS.save_dir, f'value_{i}.pkl'))
```

### 4. TripleCritic Integration into Main Codebase

Uday's TripleCritic is only in a Colab notebook. It needs to be integrated:

```python
# In value_net.py — add TripleCritic
class TripleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations, actions):
        critic1 = Critic(self.hidden_dims, activations=self.activations)(observations, actions)
        critic2 = Critic(self.hidden_dims, activations=self.activations)(observations, actions)
        critic3 = Critic(self.hidden_dims, activations=self.activations)(observations, actions)
        return critic1, critic2, critic3
```

This also requires updating:
- `critic.py` — `update_v` and `update_q` to handle 3 Q-values
- `actor.py` — `update` to take min of 3 Q-values
- `learner.py` — flag to switch between DoubleCritic and TripleCritic
- `configs/mujoco_config.py` — add `num_critics` parameter

---

## Brainstorm: What Else Can We Do for Check-In 3?

### A. Additional Shift Types (Low-Hanging Fruit)

Beyond gravity and observation noise, we could add:

1. **Friction shift** — Scale MuJoCo friction coefficients
   ```python
   self.env.unwrapped.model.geom_friction[:] *= friction_scale
   ```
   Levels: `0.5x, 1.0x, 2.0x`

2. **Action noise** — Add noise to actions before stepping
   ```python
   class ActionNoise(gym.ActionWrapper):
       def action(self, action):
           return action + np.random.normal(0, self.noise_std, size=action.shape)
   ```
   Levels: `σ = 0.01, 0.05, 0.1`

3. **Mass perturbation** — Scale body masses
   ```python
   self.env.unwrapped.model.body_mass[:] *= mass_scale
   ```

These are mentioned in the proposal but not yet planned for implementation.

### B. Richer Evaluation Metrics

Beyond Δ(δ), we could compute:

1. **Area Under Degradation Curve (AUDC)** — Integrate Δ(δ) over all shift levels
   - Single number summarizing robustness across the entire shift spectrum
   - Easy to compare DoubleCritic vs TripleCritic

2. **Worst-case performance** — `min_δ J(π, E_δ)` across all shift levels
   - Relevant for safety-critical applications

3. **Robustness threshold** — Maximum δ where performance stays above X% of baseline
   - "At what gravity scale does the policy break?"

4. **Per-episode variance under shift** — TD3+BC paper (2106.06860) highlights that offline RL policies have high episodic variance. We could measure whether Q-ensemble reduces this variance under shift.

### C. Ablation Studies

1. **Number of critics ablation**: 2 vs 3 vs 5 Q-networks
   - Does more pessimism help under shift?
   - Diminishing returns analysis

2. **Expectile τ sensitivity under shift**
   - The IQL paper shows τ=0.7 for MuJoCo, τ=0.9 for AntMaze
   - Does higher τ (more aggressive policy improvement) hurt robustness?
   - Test τ ∈ {0.5, 0.7, 0.8, 0.9} under gravity shift

3. **Temperature β sensitivity under shift**
   - Current config uses β=3.0 for MuJoCo
   - Higher β → more aggressive advantage weighting → potentially less robust
   - Test β ∈ {1.0, 3.0, 10.0}

### D. Visualization Ideas

1. **Heatmap**: Environments × Shift levels, color = normalized score
   - One heatmap for DoubleCritic, one for TripleCritic
   - Side-by-side comparison is visually compelling

2. **Degradation curves**: X-axis = shift level, Y-axis = normalized score
   - One line per method (baseline IQL, Q-ensemble IQL)
   - Shows at what point each method "breaks"

3. **Q-value distribution plots**: Under shift, do Q-values become more overestimated?
   - Plot predicted Q vs actual return under each shift level
   - This connects to CQL's gap-expanding property (Theorem 3.4 in CQL paper)

4. **Radar/spider chart**: Each axis = one shift type, radius = robustness score
   - Compact way to show multi-dimensional robustness profile

### E. Connecting to the Literature (Strengthens the Paper)

1. **CQL comparison** — The CQL paper (2006.04779) explicitly claims pessimistic value estimation improves robustness. Our Q-ensemble is a lighter form of pessimism (min over 3 Q-values vs CQL's explicit regularizer). We could argue:
   - Q-ensemble provides "implicit pessimism" similar to CQL but without the computational overhead
   - Under shift, does this implicit pessimism help as much as CQL's explicit pessimism?

2. **TD3+BC instability** — The TD3+BC paper (2106.06860) identifies high episodic variance in offline-trained policies (Figures 2-3). We could measure:
   - Does Q-ensemble reduce episodic variance?
   - Is variance worse under distribution shift?

3. **IQL's expectile as robustness knob** — The IQL paper (2110.06169) shows τ controls the spectrum from SARSA (τ=0.5) to Q-learning (τ→1). Under shift:
   - Lower τ = more conservative = potentially more robust
   - This is a novel finding if confirmed experimentally

---

## Recommended Priority for Pramod (Check-In 3)

### Must-Do (Core Deliverables)
1. ✅ Build the `evaluate_shift.py` script
2. ✅ Compute Δ(δ) from whatever shift results Shloak/Joao produce
3. ✅ Build the results table (environments × shift levels × methods)
4. ✅ Write the Experimental Results section with real numbers

### Should-Do (Strengthens the Paper)
5. Add AUDC metric computation
6. Create degradation curve plots (shift level vs normalized score)
7. Create heatmap visualization

### Nice-to-Have (If Time Permits)
8. Expectile τ ablation under shift
9. Q-value overestimation analysis under shift
10. Friction shift wrapper + experiments

---

## Concrete Next Steps (Immediate Actions)

1. **Implement shift wrappers** → `wrappers/gravity_shift.py`, `wrappers/observation_noise.py`
2. **Add model checkpointing** to `train_offline.py`
3. **Create `evaluate_shift.py`** — standalone script to evaluate saved models under shift
4. **Integrate TripleCritic** into the main codebase (not just notebook)
5. **Create `compute_robustness.py`** — takes eval CSVs, computes Δ(δ), outputs tables
6. **Run experiments** — at minimum: hopper-medium-v2 under gravity {0.5, 1.0, 1.5, 2.0} for both DoubleCritic and TripleCritic

---

## Technical Notes

### JAX/Flax Compatibility Issues
- The codebase uses `jax.tree_multimap` in `learner.py:18` which is deprecated in newer JAX versions (use `jax.tree.map` instead)
- Uday noted d4rl has Python 3.12 incompatibility — stick with Python 3.10/3.11
- The `requirements.txt` pins `jax <= 0.2.21` which is very old — may need updating for Colab

### MuJoCo Access
- Gravity modification: `env.unwrapped.model.opt.gravity`
- Friction modification: `env.unwrapped.model.geom_friction`
- Mass modification: `env.unwrapped.model.body_mass`
- These work with both `mujoco-py` and the newer `mujoco` package

### D4RL Normalized Scores
- Formula: `100 * (score - random_score) / (expert_score - random_score)`
- Reference scores for hopper-medium-v2: random ≈ -20.3, expert ≈ 3234.3
- The `EpisodeMonitor` wrapper already computes this via `get_normalized_score()`
