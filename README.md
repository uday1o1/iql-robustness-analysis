# Robustness of Implicit Q-Learning Under Controlled Distribution Shift

**CMPE 260 — Reinforcement Learning | Group 6 | San José State University**

> 📖 **New to this project?** Read the [Experiment Guide](docs/EXPERIMENT_GUIDE.md) for a plain-language explanation of the entire experiment — what IQL is, what MuJoCo robots are, what the distribution shifts do, and how to interpret the results.

---

## Team

| Student | Role |
|---|---|
| Joao Lucas Veras | Baseline IQL reproduction |
| Shloak Aggarwal | Distribution shift design |
| Pramod Yadav | Evaluation metrics & literature survey |
| Uday Arora | Q-ensemble extension |

---

## What We Set Out To Do

From our proposal, we planned to:

1. Review offline RL literature (IQL, CQL, TD3+BC)
2. Reproduce IQL on D4RL benchmarks and establish baseline scores
3. Design controlled distribution shifts in MuJoCo (gravity, friction, observation noise, reward perturbations)
4. Evaluate robustness across multiple seeds
5. Implement a robustness-oriented extension (Q-ensemble with 3 critics)
6. Conduct ablation studies comparing DoubleCritic vs TripleCritic under shift

The core research question: *How robust is Implicit Q-Learning under controlled distribution shift, and can we improve its robustness?*

---

## What We Have Demonstrated

### Literature & Theory
- Surveyed three offline RL approaches: IQL (expectile regression), CQL (pessimistic Q-values), TD3+BC (behavior cloning regularization)
- Defined formal robustness metrics: normalized performance `J(π, E_δ)`, robustness drop `Δ(δ)`, and Area Under Degradation Curve (AUDC)
- Identified the gap: none of these methods have been evaluated under environment-level perturbations at test time

### Baseline Reproduction
- Trained IQL (2Q) on all 3 D4RL medium datasets: hopper (1712), halfcheetah (5510), walker2d (3550)
- Implementation uses JAX/Flax with 2-layer MLPs (256 hidden units), expectile τ=0.7, temperature β=3.0

### Q-Ensemble Extension
- Implemented `TripleCritic` — 3 Q-networks taking `min(q1, q2, q3)` for more conservative value estimation
- Trained on all 3 environments: hopper (1426), halfcheetah (5536), walker2d (3549)
- 3Q improves robustness on Hopper (lower AUDC across all 4 shifts) but the effect is environment-dependent

### Distribution Shift Evaluation
- Evaluated 5 configurations (2Q, 3Q, 2Q-τ0.5, 2Q-τ0.8, 2Q-τ0.9) across 3 datasets and 4 shift types
- **Total: 240 shift-level evaluations** (3 envs × 5 configs × 4 shifts × 4 levels), each averaged over 10 episodes
- Gravity and friction are the most damaging shifts (AUDC > 0.5); reward perturbation has negligible impact

### Expectile τ Ablation
- Ablated τ ∈ {0.5, 0.7, 0.8, 0.9} on all 3 environments with 2Q
- Clear trend: lower τ → lower baseline but better robustness (more pessimistic value estimates)

### Evaluation Pipeline
- `scripts/evaluate_shift.py` — evaluates a trained agent under any combination of shift types, outputs CSV
- `scripts/compute_robustness.py` — reads CSVs and computes Δ(δ), AUDC, worst-case performance, writes summary CSVs
- `scripts/run_all_hpc.sh` — single script that runs all training, evaluation, ablation, and analysis on SLURM

---

## Experiment Status

| Task | Status | Notes |
|---|---|---|
| Baseline training (2Q) on all 3 envs | ✅ Complete | 300k steps, seed=42 |
| Q-ensemble training (3Q) on all 3 envs | ✅ Complete | 300k steps, seed=42 |
| Shift evaluation (2Q + 3Q × 4 shifts × 4 levels) | ✅ Complete | 6 CSVs in `results/` |
| Expectile τ ablation (τ = 0.5, 0.8, 0.9) | ✅ Complete | 9 CSVs in `results/` |
| Robustness metrics (AUDC, worst-case) | ✅ Complete | 3 summary CSVs |
| Multiple seeds for error bars | Not yet run | Change `SEEDS` in `run_all_hpc.sh` |
| Final results table and plots | Pending | `notebooks/04_analyze_results.ipynb` |

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

### Training
Standard IQL with expectile regression for value learning, TD updates for Q-learning, and advantage-weighted behavioral cloning for policy extraction. Two-layer MLP networks (256 hidden units), Adam optimizer, cosine learning rate schedule for the actor.

### Distribution Shift
Perturbations are applied **at evaluation time only** — the policy is never retrained. This isolates the sensitivity of offline-trained policies to changes in environment dynamics.

| Shift Type | MuJoCo Parameter | Levels | What It Tests |
|---|---|---|---|
| Gravity | `model.opt.gravity` | 0.5x, 1.0x, 1.5x, 2.0x | Robustness to physics/dynamics changes |
| Observation Noise | Gaussian σ | 0.0, 0.01, 0.1, 0.3 | Robustness to sensor noise (perception) |
| Friction | `model.geom_friction` | 0.5x, 1.0x, 1.5x, 2.0x | Robustness to contact dynamics changes |
| Reward Perturbation | Gaussian σ | 0.0, 0.1, 0.5, 1.0 | **Control experiment** — verifies the policy is truly offline (no test-time adaptation) |

> **Note on Reward Perturbation:** Since IQL is an offline algorithm, the policy is frozen after training and never updates from rewards during evaluation. Reward perturbation therefore has near-zero impact on agent behavior (AUDC < 0.003 across all configs). This serves as a sanity check: if reward perturbation showed large degradation, it would indicate the agent is incorrectly adapting at test time.

### Q-Ensemble Extension
We extend IQL's `DoubleCritic` (2 Q-networks, `min(q1,q2)`) to a `TripleCritic` (3 Q-networks, `min(q1,q2,q3)`). The hypothesis: taking the minimum over more Q-networks produces more conservative value estimates, which should degrade less under distribution shift.

### Metrics
- **Robustness drop:** `Δ(δ) = (J(π, E_0) - J(π, E_δ)) / J(π, E_0)` — 0 means robust, positive means degraded
- **AUDC:** Area Under Degradation Curve — integrates |Δ(δ)| over shift levels. Lower is better.
- **Worst-case:** minimum score across all shift levels

### Evaluation Strategy

The experiment pipeline runs in 4 phases, each building on the previous:

1. **Phase 1 — Training:** Train IQL agents for each (environment, num_critics) pair at the default expectile τ=0.7. This produces 6 trained models (3 envs × 2Q/3Q). Each model is saved as a checkpoint.

2. **Phase 2 — Shift Evaluation:** Load each checkpoint and evaluate it under all 4 shift types at 4 severity levels (16 conditions per model). The baseline-level rows (gravity=1.0, noise=0.0, etc.) in these CSVs serve as the **no-shift baseline performance** — no separate baseline evaluation is needed. This produces 6 CSVs with the 2Q vs 3Q comparison.

3. **Phase 3 — τ Ablation:** For each non-default τ ∈ {0.5, 0.8, 0.9}, retrain from scratch and re-evaluate under all shifts. This is done for both 2Q and 3Q, producing 18 additional CSVs (3 envs × 3 τ values × 2 critic configs). Combined with the τ=0.7 results from Phase 2, this gives a complete 4-point τ sweep for each (environment, critic) pair.

4. **Phase 4 — Analysis:** Compute AUDC and worst-case metrics from all CSVs, write per-environment summary files (`summary_{env}.csv`) with one row per (config, shift_type) combination.

The key design choice is that **shift evaluation inherently includes baseline measurement** — the no-shift condition is just one of the 4 levels tested. This avoids redundant evaluation runs and ensures baseline and shifted results use the exact same trained model.

---

## Results

All experiments run on SJSU CoE HPC (GPU partition), 300k training steps, seed=42. The experiment covers **3 D4RL datasets × 8 configurations × 4 shift types × 4 levels = 384 shift-level evaluations**, each averaged over 10 episodes.

The shift evaluation CSVs serve double duty: the baseline-level rows (gravity=1.0, obs_noise=0.0, friction=1.0, reward_perturb=0.0) provide the **no-shift baseline performance** for each configuration, while the shifted rows measure robustness degradation. This means the shift evaluation results contain both the baseline IQL and Q-ensemble performance numbers — no separate baseline files are needed.

### Baseline Performance (No Shift, τ=0.7)

| Environment | 2Q Return | 3Q Return |
|---|---|---|
| hopper-medium-v2 | **1712** | 1426 |
| halfcheetah-medium-v2 | 5510 | **5536** |
| walker2d-medium-v2 | **3550** | 3549 |

### 2Q vs 3Q Robustness (AUDC — lower is better)

**Hopper** — 3Q more robust across all shifts:

| Shift Type | 2Q AUDC | 3Q AUDC | Winner |
|---|---|---|---|
| Gravity | 0.596 | **0.574** | 3Q |
| Obs Noise | 0.146 | **0.142** | 3Q |
| Friction | 0.738 | **0.687** | 3Q |
| Reward Perturb | 0.002 | **0.001** | 3Q |

**HalfCheetah** — Mixed results:

| Shift Type | 2Q AUDC | 3Q AUDC | Winner |
|---|---|---|---|
| Gravity | **0.245** | 0.255 | 2Q |
| Obs Noise | 0.144 | **0.134** | 3Q |
| Friction | 0.016 | **0.011** | 3Q |
| Reward Perturb | **0.000** | 0.001 | 2Q |

**Walker2d** — 2Q more robust on most shifts:

| Shift Type | 2Q AUDC | 3Q AUDC | Winner |
|---|---|---|---|
| Gravity | **0.693** | 0.790 | 2Q |
| Obs Noise | **0.118** | 0.128 | 2Q |
| Friction | 0.161 | **0.154** | 3Q |
| Reward Perturb | **0.001** | 0.002 | 2Q |

### Expectile τ Ablation (2Q + 3Q)

We ablated the expectile hyperparameter τ ∈ {0.5, 0.7, 0.8, 0.9} for both 2Q and 3Q across all 3 environments and all 4 shift types. The default τ=0.7 results come from the Phase 2 shift evaluation; the non-default τ values were trained and evaluated separately in Phase 3. This gives a complete 4×2 grid (4 τ values × 2 critic configs) per environment.

**Hopper — AUDC by τ (lower = more robust):**

| τ | Config | Baseline | Gravity | Obs Noise | Friction | Reward Perturb |
|---|---|---|---|---|---|---|
| 0.5 | 2Q | 1483 | 0.546 | **0.104** | 0.700 | 0.002 |
| 0.5 | 3Q | 1478 | 0.599 | 0.148 | 0.693 | 0.001 |
| 0.7 | 2Q | 1712 | 0.596 | 0.146 | 0.738 | 0.002 |
| 0.7 | 3Q | 1426 | **0.574** | 0.142 | **0.687** | 0.001 |
| 0.8 | 2Q | 2045 | 0.757 | 0.169 | 0.782 | 0.001 |
| 0.8 | 3Q | 1782 | 0.668 | 0.146 | 0.749 | 0.001 |
| 0.9 | 2Q | 1603 | 0.668 | 0.151 | 0.714 | 0.001 |
| 0.9 | 3Q | 1688 | 0.739 | 0.156 | 0.734 | **0.001** |

> **Hopper summary:** Lower τ improves robustness: 2Q at τ=0.5 has the lowest gravity AUDC (0.546) and noise AUDC (0.104), while 3Q at τ=0.7 has the lowest friction AUDC (0.687). Higher τ (0.8, 0.9) consistently degrades robustness across all shifts. The trade-off: τ=0.5 gives up ~13% baseline return (1483 vs 1712) but gains ~8% gravity robustness and ~29% noise robustness compared to the default τ=0.7.

**HalfCheetah — AUDC by τ:**

| τ | Config | Baseline | Gravity | Obs Noise | Friction | Reward Perturb |
|---|---|---|---|---|---|---|
| 0.5 | 2Q | 5385 | 0.305 | 0.130 | 0.017 | 0.000 |
| 0.5 | 3Q | 5404 | 0.290 | **0.128** | 0.012 | 0.000 |
| 0.7 | 2Q | 5510 | **0.245** | 0.144 | 0.016 | **0.000** |
| 0.7 | 3Q | 5536 | 0.255 | 0.134 | **0.011** | 0.001 |
| 0.8 | 2Q | 5551 | 0.263 | 0.131 | **0.013** | 0.001 |
| 0.8 | 3Q | 5522 | 0.280 | 0.144 | 0.014 | 0.000 |
| 0.9 | 2Q | 5512 | 0.296 | 0.143 | 0.037 | 0.001 |
| 0.9 | 3Q | 5586 | 0.276 | 0.131 | 0.019 | 0.001 |

> **HalfCheetah summary:** This environment is the most robust overall — friction and reward perturbation AUDC are near zero across all configs. The main vulnerability is gravity shift (AUDC 0.25–0.31). The τ effect is weaker here: baselines are stable across τ values (~5400–5590), and robustness differences are small. 3Q at τ=0.7 achieves the best friction AUDC (0.011), while 2Q at τ=0.7 has the best gravity AUDC (0.245).

**Walker2d — AUDC by τ:**

| τ | Config | Baseline | Gravity | Obs Noise | Friction | Reward Perturb |
|---|---|---|---|---|---|---|
| 0.5 | 2Q | 3361 | 0.776 | 0.114 | 0.138 | **0.000** |
| 0.5 | 3Q | 3499 | **0.708** | 0.126 | **0.074** | 0.001 |
| 0.7 | 2Q | 3550 | **0.693** | **0.118** | 0.161 | 0.001 |
| 0.7 | 3Q | 3549 | 0.790 | 0.128 | 0.154 | 0.002 |
| 0.8 | 2Q | 3376 | 0.776 | **0.103** | 0.147 | 0.001 |
| 0.8 | 3Q | 3459 | 0.727 | 0.100 | 0.179 | 0.002 |
| 0.9 | 2Q | 3561 | 0.729 | 0.138 | 0.200 | 0.001 |
| 0.9 | 3Q | 3281 | 0.740 | 0.120 | 0.204 | 0.002 |

> **Walker2d summary:** Gravity is devastating across all configs (AUDC 0.69–0.79). The standout result is **3Q at τ=0.5** which achieves friction AUDC of 0.074 — the lowest across all environments and configs — while maintaining a competitive baseline of 3499. However, 2Q at τ=0.7 remains the best for gravity robustness (0.693). Observation noise robustness is best at τ=0.8 for both 2Q (0.103) and 3Q (0.100).

### Key Findings

1. **Q-ensemble (3Q) improves robustness on Hopper** but the effect is environment-dependent — Walker2d shows the opposite trend at default τ
2. **Reward perturbation has negligible impact** (AUDC < 0.003 everywhere) — offline policies are insensitive to reward noise at test time since they don't update
3. **Gravity and friction are the most damaging shifts** — AUDC > 0.5 on Hopper and Walker2d
4. **Lower expectile τ generally improves robustness** at the cost of baseline performance — consistent with IQL theory (more pessimistic value estimates)
5. **3Q + low τ can be highly effective** — Walker2d 3Q at τ=0.5 achieves the lowest friction AUDC (0.074) across all configurations
6. **The interaction between Q-ensemble and τ is non-trivial** — 3Q doesn't uniformly help; the optimal (critics, τ) pair is environment-dependent

---

## Repository Structure

```
iql-robustness-analysis/
├── iql/                          # Core IQL implementation
│   ├── actor.py                  #   Actor update (advantage-weighted BC)
│   ├── critic.py                 #   Critic update (2 or 3 Q-networks)
│   ├── common.py                 #   MLP, Model, Batch definitions
│   ├── learner.py                #   Training loop + checkpointing
│   ├── policy.py                 #   NormalTanhPolicy + sampling
│   ├── value_net.py              #   DoubleCritic, TripleCritic, ValueCritic
│   └── dataset_utils.py          #   D4RL dataset loading
│
├── evaluation/                   # Policy evaluation
│   └── evaluate.py
│
├── wrappers/                     # Environment wrappers
│   ├── episode_monitor.py        #   Episode return/length tracking
│   ├── single_precision.py       #   Float32 casting
│   ├── gravity_shift.py          #   Gravity scaling
│   ├── observation_noise.py      #   Gaussian observation noise
│   ├── friction_shift.py         #   Friction scaling
│   └── reward_perturbation.py    #   Reward noise/scaling
│
├── configs/                      # Hyperparameter configs
│   ├── mujoco_config.py          #   τ=0.7, β=3.0
│   ├── antmaze_config.py         #   τ=0.9, β=10.0
│   └── kitchen_config.py         #   τ=0.7, β=0.5, dropout=0.1
│
├── scripts/                      # Training & evaluation
│   ├── train_offline.py          #   Offline training (--num_critics flag)
│   ├── train_finetune.py         #   Online finetuning
│   ├── evaluate_shift.py         #   Evaluate under shift (all 4 types)
│   ├── compute_robustness.py     #   Compute metrics from CSVs
│   └── run_all_hpc.sh            #   Submit all experiments to SLURM
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_train_baseline.ipynb   #   Train 2Q on all envs
│   ├── 02_train_ensemble.ipynb   #   Train 3Q on all envs
│   ├── 03_evaluate_shift.ipynb   #   Evaluate under shift
│   └── 04_analyze_results.ipynb  #   Generate plots and tables
│
├── docs/                         # Documentation
│   └── EXPERIMENT_GUIDE.md       #   Plain-language experiment explanation
│
├── results/                      # Experiment outputs (24 shift CSVs + 3 summaries)
│   ├── shift_{env}_{2Q|3Q}_seed42.csv        # Phase 2: shift eval (6 files)
│   ├── shift_{env}_{2Q|3Q}_seed42_tau{τ}.csv # Phase 3: ablation (18 files)
│   ├── summary_{env}.csv                     # Phase 4: AUDC summary (3 files)
│   └── archive/                              # Legacy training curves
│
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Running Experiments

### On SJSU CoE HPC (recommended)

The SJSU College of Engineering HPC uses SLURM for job scheduling. Connect via
VPN if off-campus, then SSH in. All setup runs on the login node (which has
internet access). The script uses `--only-binary=:all:` to install pre-built
wheels, avoiding any C compilation on the login node.

```bash
# 1. SSH in (use coe-hpc1 if coe-hpc times out over VPN)
ssh <sjsu_id>@coe-hpc.sjsu.edu

# 2. Clone the repo
git clone -b sp1ffygeek_check_3 https://github.com/shloakk/iql-robustness-analysis.git
cd iql-robustness-analysis

# 3. One-time setup (on login node — downloads pre-built wheels)
bash scripts/run_all_hpc.sh setup

# 4. Submit experiments to GPU nodes
mkdir -p logs
sbatch scripts/run_all_hpc.sh          # full pipeline
# or run individual steps:
# sbatch scripts/run_all_hpc.sh train    # training only
# sbatch scripts/run_all_hpc.sh eval     # shift evaluation only
# sbatch scripts/run_all_hpc.sh analyze  # compute metrics only

# 5. Monitor
squeue -u $USER                        # check job status
tail -f logs/slurm_<job_id>.out        # watch output
```

The setup step creates a Python venv using the system Python 3.11 and installs
all dependencies as pre-built binary wheels (no compilation needed). This only
needs to be done once — the `/home` directory is shared across all HPC nodes,
so batch jobs on GPU nodes activate the same venv.

The batch job runs sequentially within a single SLURM allocation:
- **Phase 1:** Training — 6 runs (3 envs x 2 critic configs), ~20 min each
- **Phase 2:** Shift evaluation — 6 runs (all 4 shift types per model)
- **Phase 3:** Expectile tau ablation — 9 runs (3 envs x 3 tau values)
- **Phase 4:** Analysis — computes robustness metrics from CSVs

HPC partitions: `gpu` (P100/A100/H100, 48h max), `compute` (CPU only, 24h max),
`condo` (preemptible). The script defaults to the `gpu` partition with a 24h
time limit, which is sufficient for the full pipeline.

A convenience alias file is provided for common HPC commands:

```bash
source scripts/hpc_aliases.sh    # load once per session
# or add to ~/.bashrc for permanent use:
# echo 'source ~/iql-robustness-analysis/scripts/hpc_aliases.sh' >> ~/.bashrc
```

| Alias | Command |
|---|---|
| `jobs` | Check your job status |
| `myjobs` | Detailed job listing |
| `killall` | Cancel all your jobs |
| `iql-setup` | One-time environment setup |
| `iql-run` | Submit full pipeline |
| `iql-train` | Submit training only |
| `iql-eval` | Submit evaluation only |
| `iql-analyze` | Submit analysis only |
| `lastlog` | Tail the latest output log |
| `lasterr` | Tail the latest error log |
| `clearlogs` | Delete all log files |
| `cleanvenv` | Delete venv (run `iql-setup` to recreate) |
| `cleanall` | Delete venv, tmp, and logs |
| `gpunode` | Get an interactive GPU session |
| `results` | List result CSVs |

### On Google Colab

Run the notebooks in order: `01_train_baseline.ipynb` → `02_train_ensemble.ipynb` → `03_evaluate_shift.ipynb` → `04_analyze_results.ipynb`

### Locally

```bash
python scripts/train_offline.py --env_name=hopper-medium-v2 --config=configs/mujoco_config.py --num_critics=2
python scripts/evaluate_shift.py --env_name=hopper-medium-v2 --shift_type=all --num_critics=2
python scripts/compute_robustness.py --results_dir=results/ --env_name=hopper-medium-v2
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Actor / Critic / Value LR | 3e-4 |
| Hidden dims | (256, 256) |
| Discount γ | 0.99 |
| Expectile τ | 0.7 |
| Temperature β | 3.0 |
| Soft target update rate | 0.005 |
| Batch size | 256 |
| Training steps | 300,000 |
| Optimizer | Adam |
| Actor LR schedule | Cosine decay |

---

## References

1. Kostrikov, I., Nair, A., & Levine, S. (2022). *Offline Reinforcement Learning with Implicit Q-Learning*. ICLR. [arXiv:2110.06169](https://arxiv.org/abs/2110.06169)
2. Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). *Conservative Q-Learning for Offline Reinforcement Learning*. NeurIPS. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779)
3. Fujimoto, S. & Gu, S. (2021). *A Minimalist Approach to Offline Reinforcement Learning (TD3+BC)*. NeurIPS. [arXiv:2106.06860](https://arxiv.org/abs/2106.06860)
4. Fu, J., Kumar, A., Nachum, O., Tucker, G., & Levine, S. (2020). *D4RL: Datasets for Deep Data-Driven Reinforcement Learning*. [arXiv:2004.07219](https://arxiv.org/abs/2004.07219)
