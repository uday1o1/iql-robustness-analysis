# Experiment Guide — IQL Robustness Analysis

A plain-language explanation of the entire experiment, from motivation to results.

---

## What Is This Project About?

We trained simulated robots to move using **offline reinforcement learning** — the robot learns from a fixed dataset of past experience, never interacting with the real world during training. Then we tested what happens when the environment changes at test time.

**The core question:** *If a robot learned to hop on Earth, what happens when gravity changes? When the floor gets slippery? Does it fall apart or adapt gracefully?*

---

## What Is IQL?

**IQL (Implicit Q-Learning)** is an offline RL algorithm. It learns two things from the dataset:

1. **Q-function**: "How good is taking action A in state S?" — estimated by Q-networks
2. **Policy**: "What action should I take?" — extracted from the Q-function using advantage-weighted behavioral cloning

The key innovation of IQL is **expectile regression** for value learning. Instead of learning the average value, it learns a specific quantile controlled by the hyperparameter **τ (tau)**:

- **τ = 0.5**: Learn the median value (cautious, pessimistic)
- **τ = 0.7**: Default — slightly optimistic
- **τ = 0.9**: Learn near the top of the distribution (very optimistic)

Higher τ → the robot expects better outcomes → higher baseline performance but potentially more fragile when things change.

---

## What Is MuJoCo?

**MuJoCo** stands for **Mu**lti-**Jo**int dynamics with **Co**ntact. It's a physics simulation engine — think of it as a virtual world with realistic gravity, friction, collisions, and joints.

The "robots" aren't physical hardware — they're **simulated bodies** made of rigid links connected by motorized joints, running inside this physics engine. It's like a very accurate video game where the physics actually works.

### The Three Robots

```
🦘 Hopper                🐆 HalfCheetah           🚶 Walker2d

   O  ← torso               O----O                    O  ← torso
   |                        /      \                  / \
   |  ← thigh             /   body  \               /   \
   |                      |          |              |     |
   O  ← knee             /\        /\              O     O
   |                     /  \      /  \            / \   / \
   |  ← leg            legs  \   legs  \         legs  legs
   |                          feet
   O  ← foot

   1 leg, 3 joints       2 legs, 6 joints         2 legs, 6 joints
   Goal: hop forward     Goal: run forward        Goal: walk forward
   11 sensor readings    17 sensor readings        17 sensor readings
```

### Why These Robots?

They're the standard benchmark in RL research (from the [D4RL](https://github.com/Farama-Foundation/d4rl) dataset suite). Everyone uses them, so results are comparable across papers.

### Why "Medium" Datasets?

D4RL provides data at different quality levels:

| Quality | What It Is | Difficulty |
|---|---|---|
| **expert** | Data from a well-trained policy | Easy to learn from |
| **medium** | Data from a partially-trained policy | Realistic — not perfect |
| **random** | Data from random actions | Very hard to learn from |

We use **medium** because it's the most realistic scenario — the training data isn't perfect, which makes robustness more interesting to study.

---

## The Four Distribution Shifts ("Pranks")

After training, we mess with the environment to see if the robot breaks. Perturbations are applied **at evaluation time only** — the policy is never retrained.

### 🌍 Gravity Shift

We scale the gravity vector (default is Earth gravity, -9.81 m/s²):

| Level | Gravity | Analogy |
|---|---|---|
| 0.5x | -4.9 m/s² | Moon-like |
| 1.0x | -9.81 m/s² | Earth (baseline) |
| 1.5x | -14.7 m/s² | Heavy planet |
| 2.0x | -19.6 m/s² | Jupiter-like |

**What it tests:** Can the robot handle fundamental physics changes? This is the most damaging shift — AUDC > 0.5 on Hopper and Walker2d.

### 👀 Observation Noise

We add Gaussian noise to every sensor reading the robot receives:

| Level | Noise σ | Analogy |
|---|---|---|
| 0.0 | No noise | Perfect sensors (baseline) |
| 0.01 | Tiny noise | Slight sensor drift |
| 0.1 | Moderate noise | Dirty/degraded sensors |
| 0.3 | Heavy noise | Severely damaged sensors |

**What it tests:** Can the robot handle noisy perception? This simulates real-world sensor degradation.

### 🧊 Friction Shift

We scale all friction coefficients in the simulation:

| Level | Friction | Analogy |
|---|---|---|
| 0.5x | Half friction | Icy floor |
| 1.0x | Normal friction | Normal floor (baseline) |
| 1.5x | 1.5x friction | Rough surface |
| 2.0x | Double friction | Sandpaper floor |

**What it tests:** Can the robot handle contact dynamics changes? This is particularly damaging for walking/hopping robots.

### 💰 Reward Perturbation (Control Experiment)

We add Gaussian noise to the reward signal:

| Level | Noise σ | Effect |
|---|---|---|
| 0.0 | No noise | Baseline |
| 0.1 | Tiny noise | Slight reward jitter |
| 0.5 | Moderate noise | Noisy reward signal |
| 1.0 | Heavy noise | Very noisy rewards |

**What it tests:** Nothing, intentionally! Since IQL is an **offline** algorithm, the policy is frozen after training — it never updates from rewards during evaluation. This serves as a **sanity check**: if reward perturbation showed large degradation, something would be wrong. In practice, AUDC < 0.003 everywhere, confirming the policy is truly offline.

---

## The Two Knobs We Turn

### Knob 1: Number of Q-Networks (2Q vs 3Q)

IQL uses Q-networks to judge "how good is this action?" We test two architectures:

- **2Q (DoubleCritic)**: 2 Q-networks, take `min(q1, q2)` → standard IQL
- **3Q (TripleCritic)**: 3 Q-networks, take `min(q1, q2, q3)` → more conservative

**Hypothesis:** More Q-networks = more conservative value estimates = less overestimation = degrades less when the environment changes.

**Reality:** It's environment-dependent. 3Q helps on Hopper but hurts on Walker2d at the default τ.

### Knob 2: Expectile τ (Pessimism Level)

τ controls how much the robot focuses on worst-case vs best-case outcomes:

```
τ = 0.5          τ = 0.7 (default)     τ = 0.9
"Expect average"  "Slightly optimistic"  "Expect great things"
Lower baseline    Balanced               Higher baseline
More robust       Moderate               More fragile
```

**The trade-off:** Lower τ sacrifices baseline performance for better robustness. For example, on Hopper:
- τ=0.5: baseline 1483, gravity AUDC 0.546
- τ=0.7: baseline 1712 (+15%), gravity AUDC 0.596 (+9% worse)

---

## The Complete Experiment Matrix

```
3 robots × 8 configurations × 4 shifts × 4 levels = 384 evaluations
```

Each evaluation runs 10 episodes and averages the return.

| Config | Q-Networks | τ | Description |
|---|---|---|---|
| 2Q | 2 | 0.5 | Standard IQL, cautious |
| 2Q | 2 | 0.7 | Standard IQL, default |
| 2Q | 2 | 0.8 | Standard IQL, optimistic |
| 2Q | 2 | 0.9 | Standard IQL, very optimistic |
| 3Q | 3 | 0.5 | Q-ensemble, cautious |
| 3Q | 3 | 0.7 | Q-ensemble, default |
| 3Q | 3 | 0.8 | Q-ensemble, optimistic |
| 3Q | 3 | 0.9 | Q-ensemble, very optimistic |

---

## How We Measure Robustness

### AUDC (Area Under Degradation Curve)

For each shift type, we compute the **robustness drop** at each severity level:

```
Δ(δ) = (J(π, E₀) - J(π, E_δ)) / J(π, E₀)
```

Where:
- `J(π, E₀)` = performance under no shift (baseline)
- `J(π, E_δ)` = performance under shift level δ
- `Δ(δ)` = 0 means no degradation, 1 means complete failure

**AUDC** integrates |Δ(δ)| over all shift levels — a single number capturing total degradation:

```
AUDC = 0.0  →  Perfectly robust (no degradation at any level)
AUDC = 0.5  →  Moderate degradation
AUDC = 1.0  →  Complete failure under shift
```

**Lower AUDC is better.**

---

## The Pipeline (How It Runs)

The experiment runs in 4 phases on the SJSU CoE HPC cluster:

### Phase 1 — Training
Train IQL agents for each (environment, num_critics) pair at default τ=0.7.
- **Input:** D4RL datasets (pre-downloaded HDF5 files)
- **Output:** 6 model checkpoints (3 envs × 2Q/3Q)
- **Time:** ~20 min per model on GPU

### Phase 2 — Shift Evaluation
Load each checkpoint and evaluate under all 4 shift types at 4 severity levels.
- **Input:** 6 checkpoints from Phase 1
- **Output:** 6 CSVs (`shift_{env}_{2Q|3Q}_seed42.csv`)
- **Key insight:** The baseline-level rows (gravity=1.0, noise=0.0) in these CSVs **are** the baseline performance — no separate baseline run needed

### Phase 3 — τ Ablation
For each non-default τ ∈ {0.5, 0.8, 0.9}, retrain from scratch and re-evaluate under all shifts. Done for both 2Q and 3Q.
- **Input:** D4RL datasets
- **Output:** 18 CSVs (`shift_{env}_{2Q|3Q}_seed42_tau{τ}.csv`)
- **Time:** ~20 min per model × 18 models = ~6 hours

### Phase 4 — Analysis
Compute AUDC and worst-case metrics from all CSVs.
- **Input:** 24 shift CSVs from Phases 2+3
- **Output:** 3 summary CSVs (`summary_{env}.csv`)

### Running It

```bash
# On SJSU HPC:
sbatch scripts/run_all_hpc.sh          # full pipeline (all 4 phases)
sbatch scripts/run_all_hpc.sh ablation # Phase 3 only
sbatch scripts/run_all_hpc.sh analyze  # Phase 4 only
```

---

## Key Findings

1. **Q-ensemble (3Q) improves robustness on Hopper** but the effect is environment-dependent — Walker2d shows the opposite trend at default τ

2. **Reward perturbation has negligible impact** (AUDC < 0.003 everywhere) — confirms the policy is truly offline and doesn't adapt at test time

3. **Gravity and friction are the most damaging shifts** — AUDC > 0.5 on Hopper and Walker2d, while HalfCheetah is relatively robust

4. **Lower expectile τ generally improves robustness** at the cost of baseline performance — consistent with IQL theory (more pessimistic value estimates are more conservative)

5. **3Q + low τ can be highly effective** — Walker2d 3Q at τ=0.5 achieves friction AUDC of 0.074, the lowest across all environments and configurations

6. **The interaction between Q-ensemble and τ is non-trivial** — 3Q doesn't uniformly help; the optimal (critics, τ) pair is environment-dependent. There is no single "best" configuration.

---

## Glossary

| Term | Definition |
|---|---|
| **IQL** | Implicit Q-Learning — an offline RL algorithm |
| **Offline RL** | Learning from a fixed dataset without environment interaction |
| **MuJoCo** | Multi-Joint dynamics with Contact — physics simulation engine |
| **D4RL** | Datasets for Deep Data-Driven Reinforcement Learning — standard benchmark |
| **Q-network** | Neural network that estimates action quality Q(s,a) |
| **DoubleCritic (2Q)** | Standard IQL with 2 Q-networks, takes min(q1,q2) |
| **TripleCritic (3Q)** | Our extension with 3 Q-networks, takes min(q1,q2,q3) |
| **Expectile τ** | Hyperparameter controlling pessimism (0.5=cautious, 0.9=optimistic) |
| **AUDC** | Area Under Degradation Curve — robustness metric (lower=better) |
| **Distribution shift** | Change in environment dynamics between training and evaluation |
| **HPC** | High-Performance Computing cluster (SJSU CoE) |
| **SLURM** | Job scheduler used on HPC clusters |
