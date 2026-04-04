#!/bin/bash
# =============================================================================
# IQL Robustness Analysis — Full HPC Experiment Runner
# =============================================================================
#
# This script submits ALL experiments to SLURM on SJSU HPC.
# Total: 6 training jobs + 6 eval jobs + 18 ablation jobs = 30 jobs
# Estimated wall time: ~2-3 hours (all run in parallel)
#
# Usage:
#   chmod +x scripts/run_all_hpc.sh
#   ./scripts/run_all_hpc.sh
#
# Prerequisites:
#   1. conda activate iql
#   2. pip install all dependencies (see README.md)
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs results tmp

echo "=============================================="
echo "IQL Robustness Analysis — HPC Experiment Suite"
echo "=============================================="
echo "Project dir: $PROJECT_DIR"
echo ""

# =============================================================================
# CONFIGURATION
# =============================================================================
ENVIRONMENTS="hopper-medium-v2 halfcheetah-medium-v2 walker2d-medium-v2"
CRITIC_CONFIGS="2 3"
MAX_STEPS=300000
SEEDS="42"  # Add more seeds: "42 0 1" for error bars

# =============================================================================
# PHASE 1: Training (6 jobs — 3 envs × 2 critic configs)
# =============================================================================
echo "--- PHASE 1: Submitting training jobs ---"

TRAIN_JOBS=""
for env in $ENVIRONMENTS; do
    for nq in $CRITIC_CONFIGS; do
        for seed in $SEEDS; do
            JOB_NAME="train_${env}_${nq}Q_s${seed}"
            SAVE_DIR="tmp/${env}_${nq}Q_s${seed}"

            JOB_ID=$(sbatch --parsable \
                --job-name="$JOB_NAME" \
                --partition=gpu \
                --gres=gpu:1 \
                --time=02:00:00 \
                --mem=16G \
                --cpus-per-task=4 \
                --output="logs/%j_${JOB_NAME}.out" \
                --error="logs/%j_${JOB_NAME}.err" \
                --wrap="
                    module load anaconda3 2>/dev/null || true
                    conda activate iql 2>/dev/null || true
                    cd $PROJECT_DIR
                    python scripts/train_offline.py \
                        --env_name=${env} \
                        --config=configs/mujoco_config.py \
                        --num_critics=${nq} \
                        --max_steps=${MAX_STEPS} \
                        --seed=${seed} \
                        --save_dir=${SAVE_DIR}
                ")

            echo "  Submitted: $JOB_NAME (Job $JOB_ID)"
            TRAIN_JOBS="${TRAIN_JOBS}:${JOB_ID}"
        done
    done
done

# Remove leading colon
TRAIN_JOBS="${TRAIN_JOBS#:}"

# =============================================================================
# PHASE 2: Shift Evaluation (6 jobs — depends on training)
# =============================================================================
echo ""
echo "--- PHASE 2: Submitting shift evaluation jobs (after training) ---"

EVAL_JOBS=""
for env in $ENVIRONMENTS; do
    for nq in $CRITIC_CONFIGS; do
        for seed in $SEEDS; do
            JOB_NAME="eval_${env}_${nq}Q_s${seed}"
            SAVE_DIR="tmp/${env}_${nq}Q_s${seed}"

            JOB_ID=$(sbatch --parsable \
                --job-name="$JOB_NAME" \
                --partition=gpu \
                --gres=gpu:1 \
                --time=00:30:00 \
                --mem=16G \
                --cpus-per-task=4 \
                --output="logs/%j_${JOB_NAME}.out" \
                --error="logs/%j_${JOB_NAME}.err" \
                --dependency=afterok:${TRAIN_JOBS} \
                --wrap="
                    module load anaconda3 2>/dev/null || true
                    conda activate iql 2>/dev/null || true
                    cd $PROJECT_DIR
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

            echo "  Submitted: $JOB_NAME (Job $JOB_ID, depends on training)"
            EVAL_JOBS="${EVAL_JOBS}:${JOB_ID}"
        done
    done
done

EVAL_JOBS="${EVAL_JOBS#:}"

# =============================================================================
# PHASE 3: Expectile τ Ablation (9 jobs — 3 envs × 3 τ values, 2Q only)
# =============================================================================
echo ""
echo "--- PHASE 3: Submitting expectile τ ablation jobs ---"

TAU_VALUES="0.5 0.8 0.9"
ABLATION_JOBS=""

for env in $ENVIRONMENTS; do
    for tau in $TAU_VALUES; do
        for seed in $SEEDS; do
            JOB_NAME="ablation_tau${tau}_${env}_s${seed}"
            SAVE_DIR="tmp/ablation_tau${tau}_${env}_s${seed}"

            JOB_ID=$(sbatch --parsable \
                --job-name="$JOB_NAME" \
                --partition=gpu \
                --gres=gpu:1 \
                --time=02:00:00 \
                --mem=16G \
                --cpus-per-task=4 \
                --output="logs/%j_${JOB_NAME}.out" \
                --error="logs/%j_${JOB_NAME}.err" \
                --wrap="
                    module load anaconda3 2>/dev/null || true
                    conda activate iql 2>/dev/null || true
                    cd $PROJECT_DIR
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

            echo "  Submitted: $JOB_NAME (Job $JOB_ID)"
            ABLATION_JOBS="${ABLATION_JOBS}:${JOB_ID}"
        done
    done
done

# =============================================================================
# PHASE 4: Analysis (runs after all eval jobs complete)
# =============================================================================
echo ""
echo "--- PHASE 4: Submitting analysis job (after all evals) ---"

ALL_DEPS="${EVAL_JOBS}:${ABLATION_JOBS#:}"

sbatch \
    --job-name="analyze_results" \
    --partition=cpu \
    --time=00:10:00 \
    --mem=4G \
    --output="logs/%j_analyze.out" \
    --dependency=afterok:${ALL_DEPS} \
    --wrap="
        module load anaconda3 2>/dev/null || true
        conda activate iql 2>/dev/null || true
        cd $PROJECT_DIR
        for env in $ENVIRONMENTS; do
            python scripts/compute_robustness.py \
                --results_dir=results/ \
                --env_name=\$env
        done
        echo 'Analysis complete. Check results/ for CSVs.'
    "

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs:   ls logs/"
echo "Results in:   results/"
echo ""
echo "Job summary:"
echo "  Training:   $(echo $TRAIN_JOBS | tr ':' '\n' | wc -l | tr -d ' ') jobs"
echo "  Evaluation: $(echo $EVAL_JOBS | tr ':' '\n' | wc -l | tr -d ' ') jobs"
echo "  Ablation:   $(echo $ABLATION_JOBS | tr ':' '\n' | wc -l | tr -d ' ') jobs"
echo "  Analysis:   1 job"
