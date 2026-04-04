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

Offline Reinforcement Learning (Offline RL) learns decision-making policies from previously collected datasets without environment interaction. A key challenge is **distribution shift** — the policy learned from static data must operate in environments that differ from the training distribution.

We study the robustness of **Implicit Q-Learning (IQL)**, a state-of-the-art offline RL algorithm that avoids overestimation of unseen actions through expectile regression and advantage-weighted policy updates. We:

1. **Reproduce** IQL on standard D4RL benchmark tasks to establish baseline performance
2. **Evaluate** trained policies under controlled environment modifications (gravity shift, observation noise)
3. **Extend** IQL with a Q-function ensemble (TripleCritic) to improve robustness
4. **Measure** performance degradation using formal robustness metrics (Δ(δ), AUDC)

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

CQL learns a **pessimistic Q-function** that lower-bounds the true value by adding a regularizer that penalizes high Q-values for OOD actions. Key property: CQL backups are **gap-expanding** — they increase the difference between in-distribution and OOD action values (Theorem 3.4).

### TD3+BC

> Fujimoto, S. & Gu, S. (2021). *A Minimalist Approach to Offline Reinforcement Learning*. NeurIPS.

TD3+BC adds a behavior cloning term to TD3's policy update: `π = argmax_π [λQ(s,π(s)) - (π(s)-a)²]`. Notable finding: offline-trained policies exhibit **high episodic variance** compared to online-trained policies (Figures 2-3 in paper).

### Our Contribution

These methods are primarily evaluated under the assumption that training and testing environments are **identical**. None explicitly evaluate robustness under environment-level perturbations. We fill this gap by retaining standard offline training and evaluating policies under **controlled distribution shift** at deployment time.

---

## Datasets

We use the [D4RL](https://github.com/Farama-Foundation/d4rl) benchmark datasets for MuJoCo continuous control:

| Environment | Obs Dim | Act Dim | Dataset Size | Description |
|---|---|---|---|---|
| `hopper-medium-v2` | 11 | 3 | 1M transitions | One-legged hopping robot, mediocre policy data |
| `halfcheetah-medium-v2` | 17 | 6 | 1M transitions | Two-legged running robot, mediocre policy data |
| `walker2d-medium-v2` | 17 | 6 | 1M transitions | Two-legged walking robot, mediocre policy data |

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
- **AUDC:** Area Under Degradation Curve — integrates |Δ(δ)| over all shift levels (lower = more robust)
- **Worst-case:** `min_δ J(π, E_δ)` across all shift levels

---

## Progress & Status

### ✅ Completed

| Item | Details |
|---|---|
| Literature review | IQL, CQL, TD3+BC — formal comparison of distribution shift strategies |
| IQL implementation | Full JAX/Flax implementation in `iql/` package |
| Baseline training (hopper) | Score: **52.79** on hopper-medium-v2 (300k steps) |
| Q-ensemble (TripleCritic) | Integrated into codebase via `--num_critics=3` flag |
| Q-ensemble training (hopper) | Score: **50.88** on hopper-medium-v2 (300k steps) |
| Gravity shift wrapper | `wrappers/gravity_shift.py` — scales MuJoCo gravity |
| Observation noise wrapper | `wrappers/observation_noise.py` — Gaussian noise on obs |
| Shift evaluation pipeline | `scripts/evaluate_shift.py` → CSV output |
| Robustness metrics | `scripts/compute_robustness.py` — Δ(δ), AUDC, comparison tables |
| Colab notebooks | 4 independent notebooks for each pipeline step |
| Codebase restructure | Clean package layout (iql/, evaluation/, wrappers/, scripts/, notebooks/) |
| Model checkpointing | `Learner.save()` for persisting trained models |
| Comprehensive README | Background, methodology, results, usage |

### 🔶 Ready to Run (Code exists, needs execution)

| Item | Command / Notebook | Est. Time |
|---|---|---|
| Baseline training on halfcheetah + walker2d | `01_train_baseline.ipynb` or CLI | ~40 min |
| Q-ensemble training on halfcheetah + walker2d | `02_train_ensemble.ipynb` or CLI | ~40 min |
| Shift evaluation (all envs × all shifts × 2Q/3Q) | `03_evaluate_shift.ipynb` or CLI | ~10 min |
| Robustness analysis + plots | `04_analyze_results.ipynb` | ~1 min |
| Multiple seeds (3 seeds for error bars) | Re-run with `SEED=0,1,2` | ~4 hrs total |

### ❌ Not Implemented (mentioned in proposal)

| Item | Difficulty | Notes |
|---|---|---|
| **Friction shift** wrapper | Easy (~15 lines) | `model.geom_friction *= scale` |
| **Reward perturbation** wrapper | Easy (~15 lines) | Scale or add noise to rewards |
| **Expectile τ ablation** | Trivial (config change) | Test τ ∈ {0.5, 0.7, 0.8, 0.9} under shift |
| **Temperature β ablation** | Trivial (config change) | Test β ∈ {1.0, 3.0, 10.0} under shift |
| **5-critic ablation** | Easy (~10 lines) | Add `QuintupleCritic` to value_net.py |

---

## Results

### Baseline Performance (No Shift)

| Environment | Baseline IQL (2Q) | Q-Ensemble IQL (3Q) | Δ |
|---|---|---|---|
| hopper-medium-v2 | **52.79** | 50.88 | -1.91 |
| halfcheetah-medium-v2 | Pending | Pending | — |
| walker2d-medium-v2 | Pending | Pending | — |

### Shift Evaluation Results

*Pending — run notebooks or HPC scripts to generate.*

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
├── notebooks/                        # Jupyter notebooks (step-by-step)
│   ├── 01_train_baseline.ipynb       #   Step 1: Train 2Q on all envs
│   ├── 02_train_ensemble.ipynb       #   Step 2: Train 3Q on all envs
│   ├── 03_evaluate_shift.ipynb       #   Step 3: Eval under shift → CSVs
│   ├── 04_analyze_results.ipynb      #   Step 4: Plots + tables (no GPU)
│   ├── iql_shift_evaluation.ipynb    #   All-in-one pipeline
│   └── uday_q_ensemble_iql.ipynb     #   Original Q-ensemble experiments
│
├── results/                          # Experiment results
│   ├── results_baseline_iql.csv      #   Baseline IQL scores (hopper, 300k)
│   ├── results_ensemble_iql.csv      #   Q-ensemble scores (hopper, 300k)
│   └── results_comparison.png        #   Learning curve comparison plot
│
├── requirements.txt                  # Dependencies
├── LICENSE
└── .gitignore
```

---

## How to Run

### Option 1: SJSU HPC (Recommended)

HPC is the best option — persistent storage, no disconnections, GPU access, and you can run all experiments in parallel via SLURM batch jobs.

#### Setup on HPC
```bash
# SSH into HPC
ssh <your_sjsu_id>@coe-hpc1.sjsu.edu

# Clone repo
git clone -b sp1ffygeek_check_3 https://github.com/shloakk/iql-robustness-analysis.git
cd iql-robustness-analysis

# Create conda environment
module load anaconda3
conda create -n iql python=3.11 -y
conda activate iql

# Install dependencies
pip install jax jaxlib flax optax
pip install mujoco "gymnasium[mujoco]" gym
pip install h5py tqdm matplotlib numpy scipy
pip install absl-py ml_collections tensorboardX tensorflow-probability
pip install git+https://github.com/Farama-Foundation/d4rl@master
```

#### Submit batch jobs (run all experiments in parallel)
```bash
# Create a SLURM script for each experiment
cat > run_train.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=iql-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/%j_%x.out

module load anaconda3
conda activate iql

ENV_NAME=$1
NUM_CRITICS=$2

python scripts/train_offline.py \
    --env_name=${ENV_NAME} \
    --config=configs/mujoco_config.py \
    --num_critics=${NUM_CRITICS} \
    --max_steps=300000 \
    --save_dir=tmp/${ENV_NAME}_${NUM_CRITICS}Q
EOF

mkdir -p logs

# Submit all 6 training jobs (3 envs × 2 critic configs)
sbatch run_train.sh hopper-medium-v2 2
sbatch run_train.sh hopper-medium-v2 3
sbatch run_train.sh halfcheetah-medium-v2 2
sbatch run_train.sh halfcheetah-medium-v2 3
sbatch run_train.sh walker2d-medium-v2 2
sbatch run_train.sh walker2d-medium-v2 3
```

#### Submit shift evaluation jobs (after training completes)
```bash
cat > run_eval.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=iql-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=logs/%j_%x.out

module load anaconda3
conda activate iql

ENV_NAME=$1
NUM_CRITICS=$2

python scripts/evaluate_shift.py \
    --env_name=${ENV_NAME} \
    --config=configs/mujoco_config.py \
    --num_critics=${NUM_CRITICS} \
    --shift_type=both \
    --save_dir=tmp/${ENV_NAME}_${NUM_CRITICS}Q \
    --output_dir=results/
EOF

# Submit all 6 eval jobs
for env in hopper-medium-v2 halfcheetah-medium-v2 walker2d-medium-v2; do
    sbatch run_eval.sh $env 2
    sbatch run_eval.sh $env 3
done
```

#### Analyze results (runs instantly, no GPU needed)
```bash
python scripts/compute_robustness.py --results_dir=results/ --env_name=hopper-medium-v2
python scripts/compute_robustness.py --results_dir=results/ --env_name=halfcheetah-medium-v2
python scripts/compute_robustness.py --results_dir=results/ --env_name=walker2d-medium-v2
```

### Option 2: Google Colab

Run the notebooks in order:

| Step | Notebook | Runtime | GPU? |
|---|---|---|---|
| 1 | `notebooks/01_train_baseline.ipynb` | ~60 min | Yes (T4) |
| 2 | `notebooks/02_train_ensemble.ipynb` | ~60 min | Yes (T4) |
| 3 | `notebooks/03_evaluate_shift.ipynb` | ~10 min | Yes |
| 4 | `notebooks/04_analyze_results.ipynb` | ~1 min | No |

### Option 3: Local CLI

```bash
# Train baseline
python scripts/train_offline.py --env_name=hopper-medium-v2 --config=configs/mujoco_config.py --num_critics=2

# Train Q-ensemble
python scripts/train_offline.py --env_name=hopper-medium-v2 --config=configs/mujoco_config.py --num_critics=3

# Evaluate under shift
python scripts/evaluate_shift.py --env_name=hopper-medium-v2 --shift_type=both --num_critics=2

# Compute metrics
python scripts/compute_robustness.py --results_dir=results/ --env_name=hopper-medium-v2
```

---

## HPC vs Colab Comparison

| Feature | SJSU HPC | Google Colab |
|---|---|---|
| **Runtime limit** | Hours (configurable) | 90 min (free), 24h (Pro) |
| **Disconnection risk** | None | Frequent on free tier |
| **Parallel jobs** | ✅ 6+ jobs simultaneously | ❌ 1 notebook at a time |
| **Persistent storage** | ✅ Home directory persists | ❌ Lost on disconnect |
| **GPU** | ✅ (SLURM `--gres=gpu:1`) | ✅ (T4/V100) |
| **Total time (all exps)** | ~2 hrs (parallel) | ~6 hrs (sequential) |
| **Best for** | Full experiment sweep | Quick prototyping |

**Recommendation:** Use HPC for the full experiment sweep (6 training + 6 eval jobs in parallel = ~2 hrs total). Use Colab only for quick debugging or if HPC is unavailable.

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
