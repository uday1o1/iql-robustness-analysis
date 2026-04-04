# Robustness of Implicit Q-Learning Under Controlled Distribution Shift

**CMPE 260 — Reinforcement Learning | Group 6 | San José State University**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/framework-JAX%2FFlax-orange.svg)](https://github.com/google/jax)
[![D4RL](https://img.shields.io/badge/benchmark-D4RL-green.svg)](https://github.com/Farama-Foundation/d4rl)

---

## Team

| Student | SJSU ID | Role |
|---|---|---|
| Joao Lucas Veras | 015555345 | Baseline IQL reproduction |
| Shloak Aggarwal | 018189938 | Distribution shift design |
| Pramod Yadav | 019142370 | Evaluation metrics & literature survey |
| Uday Arora | 019098404 | Q-ensemble extension |

---

## Abstract

Offline Reinforcement Learning (Offline RL) learns decision-making policies from previously collected datasets without environment interaction. A key challenge is **distribution shift** — the policy learned from static data must operate in environments that differ from the training distribution. Such shifts arise from changes in dynamics, observations, or reward structures, and often lead to performance degradation.

We study the robustness of **Implicit Q-Learning (IQL)**, a state-of-the-art offline RL algorithm that avoids overestimation of unseen actions through expectile regression and advantage-weighted policy updates. We:

1. **Reproduce** IQL on standard D4RL benchmark tasks to establish baseline performance
2. **Evaluate** trained policies under controlled environment modifications (gravity shift, observation noise)
3. **Extend** IQL with a Q-function ensemble (TripleCritic) to improve robustness
4. **Measure** performance degradation using formal robustness metrics

**Research Question:** *How robust is Implicit Q-Learning under controlled distribution shift, and can we improve its robustness?*

---

## Background & Literature

### Implicit Q-Learning (IQL) — Base Paper

> Kostrikov, I., Nair, A., & Levine, S. (2022). *Offline Reinforcement Learning with Implicit Q-Learning*. ICLR.

IQL avoids explicit maximization over actions in the Bellman backup — a major source of overestimation in offline settings. Instead, it:

- Estimates a state value function using **expectile regression** over dataset actions
- Updates Q-functions using **in-sample TD learning** (never queries OOD actions)
- Extracts the policy via **advantage-weighted behavioral cloning**

Key equations:
- **Value loss:** `L_V(ψ) = E_{(s,a)~D} [L_τ² (Q̂(s,a) - V_ψ(s))]` where `L_τ²` is the asymmetric squared loss
- **Q-function loss:** `L_Q(θ) = E_{(s,a,s')~D} [(r(s,a) + γV_ψ(s') - Q_θ(s,a))²]`
- **Policy loss:** `L_π(φ) = E_{(s,a)~D} [exp(β(Q̂(s,a) - V_ψ(s))) log π_φ(a|s)]`

The expectile parameter `τ` controls the spectrum from SARSA (`τ=0.5`) to Q-learning (`τ→1`).

### Conservative Q-Learning (CQL)

> Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). *Conservative Q-Learning for Offline Reinforcement Learning*. NeurIPS.

CQL learns a **pessimistic Q-function** that lower-bounds the true value by adding a regularizer that penalizes high Q-values for OOD actions. Key property: CQL backups are **gap-expanding** — they increase the difference between in-distribution and OOD action values.

### TD3+BC

> Fujimoto, S. & Gu, S. (2021). *A Minimalist Approach to Offline Reinforcement Learning*. NeurIPS.

TD3+BC adds a behavior cloning term to TD3's policy update: `π = argmax_π [λQ(s,π(s)) - (π(s)-a)²]`. Notable finding: offline-trained policies exhibit **high episodic variance** compared to online-trained policies.

### Our Contribution

These methods are primarily evaluated under the assumption that training and testing environments are **identical**. None explicitly evaluate robustness under environment-level perturbations. We fill this gap by:
- Retaining standard offline training
- Evaluating policies under **controlled distribution shift** at deployment time
- Comparing baseline IQL vs Q-ensemble IQL under shift

---

## Datasets

We use the [D4RL](https://github.com/Farama-Foundation/d4rl) benchmark datasets for MuJoCo continuous control:

| Environment | Obs Dim | Act Dim | Dataset Size | Description |
|---|---|---|---|---|
| `hopper-medium-v2` | 11 | 3 | 1M transitions | One-legged hopping robot, mediocre policy data |
| `halfcheetah-medium-v2` | 17 | 6 | 1M transitions | Two-legged running robot, mediocre policy data |
| `walker2d-medium-v2` | 17 | 6 | 1M transitions | Two-legged walking robot, mediocre policy data |

Each dataset contains transitions from a partially trained ("medium") policy — not random, not expert.

---

## Methodology

### Distribution Shift Design

We apply perturbations **at evaluation time only** — the policy is never retrained:

| Shift Type | Parameter | Levels | Mechanism |
|---|---|---|---|
| **Gravity** | `model.opt.gravity` | 0.5x, 1.0x, 1.5x, 2.0x | Scales MuJoCo gravity vector |
| **Observation Noise** | Gaussian σ | 0.0, 0.01, 0.1, 0.3 | Adds N(0,σ²) to observations |

### Q-Ensemble Extension

We extend IQL's `DoubleCritic` (2 Q-networks, `min(q1,q2)`) to a `TripleCritic` (3 Q-networks, `min(q1,q2,q3)`). The hypothesis: more conservative value estimation reduces overestimation under distribution shift, providing **implicit pessimism** similar to CQL but without the computational overhead.

### Robustness Metrics

- **Normalized score:** `J(π, E_δ) = 100 × (Return - Return_random) / (Return_expert - Return_random)`
- **Robustness drop:** `Δ(δ) = (J(π, E_0) - J(π, E_δ)) / J(π, E_0)` — 0 = robust, positive = degraded
- **AUDC:** Area Under Degradation Curve — integrates |Δ(δ)| over all shift levels
- **Worst-case:** `min_δ J(π, E_δ)` across all shift levels

---

## Progress Timeline

### Check-In 1 — Project Setup
- ✅ Identified D4RL datasets (hopper, halfcheetah, walker2d)
- ✅ Loaded datasets, verified MuJoCo compatibility
- ✅ Inspected state/action distributions
- ✅ Confirmed environment parameters for modification
- ✅ Paper outline drafted

### Check-In 2 — Baseline & Extension
- ✅ **Joao:** Baseline IQL running on `hopper-medium-v2` — score: **52.79**
- ✅ **Shloak:** Gravity + observation noise wrapper design
- ✅ **Pramod:** Literature survey (IQL, CQL, TD3+BC), formal robustness definition
- ✅ **Uday:** TripleCritic implemented and trained — score: **50.88** on `hopper-medium-v2`
- ✅ Results: Baseline and Q-ensemble converge to comparable final performance (difference of 1.91 points), confirming correct implementation

### Check-In 3 — Shift Evaluation & Analysis
- ✅ Codebase restructured into clean package layout
- ✅ Gravity shift wrapper implemented (`wrappers/gravity_shift.py`)
- ✅ Observation noise wrapper implemented (`wrappers/observation_noise.py`)
- ✅ TripleCritic integrated into main codebase (via `--num_critics=3` flag)
- ✅ Shift evaluation pipeline built (`scripts/evaluate_shift.py`)
- ✅ Robustness metrics computation (`scripts/compute_robustness.py`)
- ✅ Full Colab notebook for end-to-end experiments (`notebooks/iql_shift_evaluation.ipynb`)
- ✅ Model checkpointing added to training script
- 🔶 Running shift evaluations across all environments and shift levels
- 🔶 Computing Δ(δ) and AUDC across all conditions
- 🔶 Building comparison tables (DoubleCritic vs TripleCritic under shift)

---

## Results

### Baseline Performance (No Shift)

| Environment | Baseline IQL (2Q) | Q-Ensemble IQL (3Q) | Δ |
|---|---|---|---|
| hopper-medium-v2 | **52.79** | 50.88 | -1.91 |
| halfcheetah-medium-v2 | Pending | Pending | — |
| walker2d-medium-v2 | Pending | Pending | — |

### Shift Evaluation Results

*Pending — run `notebooks/iql_shift_evaluation.ipynb` on Colab.*

---

## Repository Structure

```
iql-robustness-analysis/
│
├── iql/                              # Core IQL implementation
│   ├── __init__.py                   #   Exports: Learner, Batch, Model
│   ├── actor.py                      #   Actor update (advantage-weighted BC)
│   ├── critic.py                     #   Critic update — supports 2 or 3 Q-nets
│   ├── common.py                     #   MLP, Model dataclass, Batch namedtuple
│   ├── learner.py                    #   Training loop + checkpoint save
│   ├── policy.py                     #   NormalTanhPolicy + sampling
│   ├── value_net.py                  #   DoubleCritic + TripleCritic
│   └── dataset_utils.py             #   D4RL dataset loading
│
├── evaluation/                       # Evaluation utilities
│   ├── __init__.py
│   └── evaluate.py                   #   Policy evaluation loop
│
├── wrappers/                         # Environment wrappers
│   ├── __init__.py
│   ├── common.py                     #   Shared types
│   ├── episode_monitor.py            #   Episode return/length tracking
│   ├── single_precision.py           #   Float32 observation casting
│   ├── gravity_shift.py              #   Gravity scaling for dynamics shift
│   └── observation_noise.py          #   Gaussian noise for observation shift
│
├── configs/                          # Hyperparameter configs
│   ├── mujoco_config.py              #   τ=0.7, β=3.0 (HalfCheetah/Hopper/Walker2d)
│   ├── antmaze_config.py             #   τ=0.9, β=10.0 (AntMaze tasks)
│   ├── antmaze_finetune_config.py    #   AntMaze finetuning
│   └── kitchen_config.py             #   τ=0.7, β=0.5, dropout=0.1
│
├── scripts/                          # Training & evaluation scripts
│   ├── train_offline.py              #   Offline training (--num_critics flag)
│   ├── train_finetune.py             #   Online finetuning after offline
│   ├── evaluate_shift.py             #   Evaluate under distribution shift
│   └── compute_robustness.py         #   Compute Δ(δ), AUDC, comparison tables
│
├── notebooks/                        # Jupyter notebooks
│   ├── iql_shift_evaluation.ipynb    #   Full pipeline (train → shift → plots)
│   └── uday_q_ensemble_iql.ipynb     #   Original Q-ensemble experiments
│
├── results/                          # Experiment results
│   ├── results_baseline_iql.csv      #   Baseline IQL scores (hopper, 300k steps)
│   ├── results_ensemble_iql.csv      #   Q-ensemble scores (hopper, 300k steps)
│   └── results_comparison.png        #   Learning curve comparison plot
│
├── requirements.txt                  # Dependencies
├── LICENSE
└── .gitignore
```

---

## How to Run

### Option 1: Google Colab (Recommended)

Open `notebooks/iql_shift_evaluation.ipynb` in Google Colab with GPU runtime. It runs the full pipeline end-to-end (~40 min on T4):
1. Install dependencies
2. Train baseline IQL (2Q) and Q-ensemble IQL (3Q)
3. Evaluate under gravity shift and observation noise
4. Compute robustness metrics
5. Generate comparison plots and tables

### Option 2: Local / CLI

#### Install dependencies
```bash
pip install jax jaxlib flax optax
pip install mujoco "gymnasium[mujoco]" gym
pip install h5py tqdm matplotlib numpy scipy
pip install absl-py ml_collections tensorboardX tensorflow-probability
pip install git+https://github.com/Farama-Foundation/d4rl@master
```

#### Train baseline IQL (DoubleCritic)
```bash
python scripts/train_offline.py \
    --env_name=hopper-medium-v2 \
    --config=configs/mujoco_config.py \
    --num_critics=2 \
    --max_steps=300000
```

#### Train Q-ensemble IQL (TripleCritic)
```bash
python scripts/train_offline.py \
    --env_name=hopper-medium-v2 \
    --config=configs/mujoco_config.py \
    --num_critics=3 \
    --max_steps=300000
```

#### Evaluate under distribution shift
```bash
# Gravity shift
python scripts/evaluate_shift.py \
    --env_name=hopper-medium-v2 \
    --config=configs/mujoco_config.py \
    --shift_type=gravity \
    --shift_levels="0.5,1.0,1.5,2.0" \
    --num_critics=2

# Observation noise
python scripts/evaluate_shift.py \
    --env_name=hopper-medium-v2 \
    --config=configs/mujoco_config.py \
    --shift_type=obs_noise \
    --shift_levels="0.0,0.01,0.1,0.3" \
    --num_critics=2

# Both shift types at once
python scripts/evaluate_shift.py \
    --shift_type=both --num_critics=2
```

#### Compute robustness metrics
```bash
python scripts/compute_robustness.py \
    --results_dir=./results/ \
    --env_name=hopper-medium-v2
```

---

## Hyperparameters

| Parameter | MuJoCo | AntMaze | Kitchen |
|---|---|---|---|
| Actor LR | 3e-4 | 3e-4 | 3e-4 |
| Critic LR | 3e-4 | 3e-4 | 3e-4 |
| Value LR | 3e-4 | 3e-4 | 3e-4 |
| Hidden dims | (256, 256) | (256, 256) | (256, 256) |
| Discount γ | 0.99 | 0.99 | 0.99 |
| Expectile τ | 0.7 | 0.9 | 0.7 |
| Temperature β | 3.0 | 10.0 | 0.5 |
| Soft target τ | 0.005 | 0.005 | 0.005 |
| Dropout | None | None | 0.1 |
| Optimizer | Adam | Adam | Adam |
| Actor schedule | Cosine decay | Cosine decay | Cosine decay |

---

## References

1. Kostrikov, I., Nair, A., & Levine, S. (2022). *Offline Reinforcement Learning with Implicit Q-Learning*. ICLR. [arXiv:2110.06169](https://arxiv.org/abs/2110.06169)
2. Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). *Conservative Q-Learning for Offline Reinforcement Learning*. NeurIPS. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779)
3. Fujimoto, S. & Gu, S. (2021). *A Minimalist Approach to Offline Reinforcement Learning (TD3+BC)*. NeurIPS. [arXiv:2106.06860](https://arxiv.org/abs/2106.06860)
4. Fu, J., Kumar, A., Nachum, O., Tucker, G., & Levine, S. (2020). *D4RL: Datasets for Deep Data-Driven Reinforcement Learning*. [arXiv:2004.07219](https://arxiv.org/abs/2004.07219)
