#!/bin/bash
# =============================================================================
# IQL Robustness Analysis — Full HPC Experiment Runner
# =============================================================================
#
# Submits all experiments to SLURM. Handles environment setup automatically.
#
# Usage:
#   chmod +x scripts/run_all_hpc.sh
#   ./scripts/run_all_hpc.sh
#
# No prerequisites — the script sets up conda and installs deps in each job.
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs results tmp

# =============================================================================
# CONFIGURATION — edit these as needed
# =============================================================================
ENVIRONMENTS="hopper-medium-v2 halfcheetah-medium-v2 walker2d-medium-v2"
CRITIC_CONFIGS="2 3"
MAX_STEPS=300000
SEEDS="42"
CONDA_ENV="iql"
PARTITION="gpu"

# Shared setup commands injected into every SLURM job
SETUP_CMD="
    # Load modules
    module load anaconda3 2>/dev/null || true

    # Create conda env if it doesn't exist
    if ! conda info --envs | grep -q ${CONDA_ENV}; then
        echo 'Creating conda environment ${CONDA_ENV}...'
        conda create -n ${CONDA_ENV} python=3.11 -y
    fi
    conda activate ${CONDA_ENV}

    # Install deps (pip is idempotent — skips already-installed packages)
    pip install -q jax jaxlib flax optax
    pip install -q mujoco 'gymnasium[mujoco]' gym
    pip install -q h5py tqdm matplotlib numpy scipy
    pip install -q absl-py ml_collections tensorboardX tensorflow-probability
    pip install -q git+https://github.com/Farama-Foundation/d4rl@master 2>/dev/null || true

    cd ${PROJECT_DIR}
"

echo "=============================================="
echo "IQL Robustness — HPC Experiment Suite"
echo "=============================================="
echo "Project: $PROJECT_DIR"
echo "Envs:    $ENVIRONMENTS"
echo "Critics: $CRITIC_CONFIGS"
echo "Seeds:   $SEEDS"
echo "Steps:   $MAX_STEPS"
echo ""

# =============================================================================
# PHASE 1: Training (3 envs × 2 critic configs = 6 jobs)
# =============================================================================
echo "--- Phase 1: Training ---"

TRAIN_JOBS=""
for env in $ENVIRONMENTS; do
    for nq in $CRITIC_CONFIGS; do
        for seed in $SEEDS; do
            JOB_NAME="train_${env}_${nq}Q_s${seed}"
            SAVE_DIR="tmp/${env}_${nq}Q_s${seed}"

            JOB_ID=$(sbatch --parsable \
                --job-name="$JOB_NAME" \
                --partition=$PARTITION \
                --gres=gpu:1 \
                --time=02:00:00 \
                --mem=16G \
                --cpus-per-task=4 \
                --output="logs/%j_${JOB_NAME}.out" \
                --error="logs/%j_${JOB_NAME}.err" \
                --wrap="
                    ${SETUP_CMD}
                    python scripts/train_offline.py \
                        --env_name=${env} \
                        --config=configs/mujoco_config.py \
                        --num_critics=${nq} \
                        --max_steps=${MAX_STEPS} \
                        --seed=${seed} \
                        --save_dir=${SAVE_DIR}
                ")

            echo "  $JOB_NAME -> Job $JOB_ID"
            TRAIN_JOBS="${TRAIN_JOBS}:${JOB_ID}"
        done
    done
done

TRAIN_JOBS="${TRAIN_JOBS#:}"

# =============================================================================
# PHASE 2: Shift Evaluation (depends on training)
# =============================================================================
echo ""
echo "--- Phase 2: Shift Evaluation (waits for training) ---"

EVAL_JOBS=""
for env in $ENVIRONMENTS; do
    for nq in $CRITIC_CONFIGS; do
        for seed in $SEEDS; do
            JOB_NAME="eval_${env}_${nq}Q_s${seed}"
            SAVE_DIR="tmp/${env}_${nq}Q_s${seed}"

            JOB_ID=$(sbatch --parsable \
                --job-name="$JOB_NAME" \
                --partition=$PARTITION \
                --gres=gpu:1 \
                --time=00:30:00 \
                --mem=16G \
                --cpus-per-task=4 \
                --output="logs/%j_${JOB_NAME}.out" \
                --error="logs/%j_${JOB_NAME}.err" \
                --dependency=afterok:${TRAIN_JOBS} \
                --wrap="
                    ${SETUP_CMD}
                    python scripts/evaluate_shift.py \
                        --env_name=${env} \
                        --config=configs/mujoco_config.py \
                        --num_critics=${nq} \
                        --shift_type=all \
                        --max_steps=${MAX_STEPS} \
                        --seed=${seed} \
                        --save_dir=${SAVE_DIR} \
                        --output_dir=results/
                ")

            echo "  $JOB_NAME -> Job $JOB_ID (after training)"
            EVAL_JOBS="${EVAL_JOBS}:${JOB_ID}"
        done
    done
done

EVAL_JOBS="${EVAL_JOBS#:}"

# =============================================================================
# PHASE 3: Expectile τ Ablation (independent, 2Q only)
# =============================================================================
echo ""
echo "--- Phase 3: Expectile τ Ablation ---"

TAU_VALUES="0.5 0.8 0.9"
ABLATION_JOBS=""

for env in $ENVIRONMENTS; do
    for tau in $TAU_VALUES; do
        for seed in $SEEDS; do
            JOB_NAME="abl_tau${tau}_${env}_s${seed}"
            SAVE_DIR="tmp/abl_tau${tau}_${env}_s${seed}"

            JOB_ID=$(sbatch --parsable \
                --job-name="$JOB_NAME" \
                --partition=$PARTITION \
                --gres=gpu:1 \
                --time=02:00:00 \
                --mem=16G \
                --cpus-per-task=4 \
                --output="logs/%j_${JOB_NAME}.out" \
                --error="logs/%j_${JOB_NAME}.err" \
                --wrap="
                    ${SETUP_CMD}

                    python scripts/train_offline.py \
                        --env_name=${env} \
                        --config=configs/mujoco_config.py \
                        --config.expectile=${tau} \
                        --num_critics=2 \
                        --max_steps=${MAX_STEPS} \
                        --seed=${seed} \
                        --save_dir=${SAVE_DIR}

                    python scripts/evaluate_shift.py \
                        --env_name=${env} \
                        --config=configs/mujoco_config.py \
                        --config.expectile=${tau} \
                        --num_critics=2 \
                        --shift_type=all \
                        --max_steps=${MAX_STEPS} \
                        --seed=${seed} \
                        --save_dir=${SAVE_DIR} \
                        --output_dir=results/ablation_tau/
                ")

            echo "  $JOB_NAME -> Job $JOB_ID"
            ABLATION_JOBS="${ABLATION_JOBS}:${JOB_ID}"
        done
    done
done

ABLATION_JOBS="${ABLATION_JOBS#:}"

# =============================================================================
# PHASE 4: Analysis (after all evals finish)
# =============================================================================
echo ""
echo "--- Phase 4: Analysis (waits for all evals) ---"

ALL_DEPS="${EVAL_JOBS}:${ABLATION_JOBS}"

sbatch \
    --job-name="analyze" \
    --partition=cpu \
    --time=00:10:00 \
    --mem=4G \
    --output="logs/%j_analyze.out" \
    --dependency=afterok:${ALL_DEPS} \
    --wrap="
        ${SETUP_CMD}
        for env in $ENVIRONMENTS; do
            echo \"--- \$env ---\"
            python scripts/compute_robustness.py \
                --results_dir=results/ \
                --env_name=\$env
        done
        echo ''
        echo 'Done. Results in results/'
    "

echo ""
echo "=============================================="
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: logs/"
echo "Results: results/"
echo "=============================================="
