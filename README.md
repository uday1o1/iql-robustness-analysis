# iql-robustness-analysis

**Robustness of Implicit Q-Learning Under Controlled Distribution Shift**

CMPE 260 — Group 6 | San José State University

---

## Team

| Student | SJSU ID | Role |
|---|---|---|
| Joao Lucas Veras | 015555345 | Baseline IQL reproduction |
| Shloak Aggarwal | 018189938 | Distribution shift design |
| Pramod Yadav | 019142370 | Evaluation metrics & literature survey |
| Uday Arora | 019098404 | Q-ensemble extension |

---

## Project Overview

This project investigates the robustness of Implicit Q-Learning (IQL) under controlled distribution shift. We train IQL policies on standard D4RL offline RL benchmarks, then evaluate them under modified environments to measure performance degradation. We also implement a Q-ensemble extension and compare it against the baseline.

**Research Question:** How robust is Implicit Q-Learning under controlled distribution shift, and can we improve its robustness?

**Base Paper:** Kostrikov et al., *Offline Reinforcement Learning with Implicit Q-Learning*, ICLR 2022.

---

## Datasets

We use the D4RL benchmark datasets for MuJoCo continuous control:

- `hopper-medium-v2` — one-legged hopping robot, mediocre policy data
- `halfcheetah-medium-v2` — two-legged running robot, mediocre policy data
- `walker2d-medium-v2` — two-legged walking robot, mediocre policy data

---

## Current Progress (Check-In 3)

### Baseline IQL — Joao
- ✅ Forked and set up the original IQL implementation
- ✅ Baseline training on `hopper-medium-v2` — score: **52.79**
- 🔶 Running baseline on `halfcheetah-medium-v2` and `walker2d-medium-v2`

### Distribution Shift Wrappers — Shloak
- ✅ Gravity shift wrapper — `wrappers/gravity_shift.py`
  - Scales MuJoCo gravity at `0.5x, 1.0x, 1.5x, 2.0x`
- ✅ Observation noise wrapper — `wrappers/observation_noise.py`
  - Gaussian noise at `σ = 0.01, 0.1, 0.3`
- 🔶 Running shift evaluations on `hopper-medium-v2`

### Evaluation Metrics — Pramod
- ✅ Literature survey covering IQL, CQL, TD3+BC
- ✅ Formal robustness definition:
  - Normalized performance: `J(π, E_δ) = (Return(π, E_δ) - Return_random) / (Return_expert - Return_random)`
  - Robustness drop: `Δ(δ) = (J(π, E_0) - J(π, E_δ)) / J(π, E_0)`
- ✅ Shift evaluation script — `evaluate_shift.py`
- ✅ Robustness computation script — `compute_robustness.py`
  - Computes AUDC, worst-case performance, robustness threshold
  - Generates DoubleCritic vs TripleCritic comparison tables

### Q-Ensemble Extension — Uday (`uday/q-ensemble` branch)
- ✅ Implemented `TripleCritic` — 3 Q-networks (now integrated into main codebase)
- ✅ Takes `min(q1, q2, q3)` for conservative value estimation
- ✅ Trained and compared baseline IQL vs Q-ensemble on `hopper-medium-v2`
- Results saved in `results_baseline_iql.csv` and `results_ensemble_iql.csv`
- See `uday_q_ensemble_iql.ipynb` for full implementation and results

---

## Repository Structure

```
iql-robustness-analysis/
├── actor.py                      # Actor (policy) update
├── critic.py                     # Critic (Q-network) update — supports 2 or 3 Q-nets
├── common.py                     # Shared model utilities
├── learner.py                    # Main IQL training loop — configurable num_critics
├── value_net.py                  # DoubleCritic + TripleCritic network definitions
├── policy.py                     # Policy sampling
├── dataset_utils.py              # D4RL dataset loading
├── evaluation.py                 # Policy evaluation
├── train_offline.py              # Offline training script (--num_critics flag)
├── evaluate_shift.py             # Evaluate under distribution shift
├── compute_robustness.py         # Compute Δ(δ), AUDC, comparison tables
├── configs/                      # Hyperparameter configs
│   ├── mujoco_config.py
│   └── antmaze_config.py
├── wrappers/                     # Environment wrappers
│   ├── episode_monitor.py
│   ├── single_precision.py
│   ├── gravity_shift.py          # Gravity scaling for dynamics shift
│   └── observation_noise.py      # Gaussian noise for observation shift
├── plans/                        # Research papers + check-in docs
│   ├── CHECKIN3_BRAINSTORM.md    # Brainstorm & action items for Check-in 3
│   ├── 2110.06169v1.pdf          # IQL paper (Kostrikov et al., 2022)
│   ├── 2006.04779v3.pdf          # CQL paper (Kumar et al., 2020)
│   └── 2106.06860v2.pdf          # TD3+BC paper (Fujimoto & Gu, 2021)
│
│── [uday/q-ensemble branch]
│   ├── uday_q_ensemble_iql.ipynb     # Q-ensemble implementation + results
│   ├── results_baseline_iql.csv      # Baseline IQL scores per step
│   ├── results_ensemble_iql.csv      # Q-ensemble scores per step
│   └── results_comparison.png        # Learning curve comparison plot
```

---

## How to Run

### Install dependencies
```bash
pip install --upgrade pip
pip install jax jaxlib flax optax
pip install mujoco "gymnasium[mujoco]"
pip install h5py tqdm matplotlib
```

### Download dataset
```bash
wget -O hopper-medium-v2.hdf5 "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5"
```

### Run baseline IQL (DoubleCritic)
```bash
python train_offline.py --env_name=hopper-medium-v2 --config=configs/mujoco_config.py --num_critics=2
```

### Run Q-ensemble IQL (TripleCritic)
```bash
python train_offline.py --env_name=hopper-medium-v2 --config=configs/mujoco_config.py --num_critics=3
```

### Evaluate under distribution shift
```bash
# Gravity shift
python evaluate_shift.py --env_name=hopper-medium-v2 --config=configs/mujoco_config.py \
    --shift_type=gravity --shift_levels="0.5,1.0,1.5,2.0" --num_critics=2

# Observation noise
python evaluate_shift.py --env_name=hopper-medium-v2 --config=configs/mujoco_config.py \
    --shift_type=obs_noise --shift_levels="0.0,0.01,0.1,0.3" --num_critics=2
```

### Compute robustness metrics
```bash
python compute_robustness.py --results_dir=./results/ --env_name=hopper-medium-v2
```

### Run Q-ensemble in Colab
See `uday_q_ensemble_iql.ipynb` — runs fully in Google Colab with no local setup required.

---

## Experiments (Check-In 3)

### Completed
- ✅ Baseline IQL on `hopper-medium-v2` — score: 52.79
- ✅ Q-ensemble IQL on `hopper-medium-v2` — score: 50.88
- ✅ Shift wrappers implemented (gravity, observation noise)
- ✅ Evaluation pipeline built (evaluate_shift.py + compute_robustness.py)

### In Progress
- Evaluate baseline and Q-ensemble under all shift types and levels
- Run across all 3 environments and 3 seeds
- Compute formal robustness metric `Δ(δ)` across shift levels
- Ablation study comparing DoubleCritic vs TripleCritic under each shift condition

---

## References

1. Kostrikov, I., Nair, A., & Levine, S. (2022). Offline Reinforcement Learning with Implicit Q-Learning. ICLR.
2. Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning. NeurIPS.
3. Fujimoto, S., Meger, D., & Precup, D. (2021). A Minimalist Approach to Offline Reinforcement Learning (TD3+BC). NeurIPS.
