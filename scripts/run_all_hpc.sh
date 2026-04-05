#!/bin/bash
#SBATCH --job-name=iql_robustness
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#
# IQL Robustness Analysis — SJSU CoE HPC Experiment Runner
#
# Self-contained script that handles environment setup, training,
# evaluation under distribution shift, and analysis.
#
# Submit from the project root:
#   mkdir -p logs
#   sbatch scripts/run_all_hpc.sh            # run everything
#   sbatch scripts/run_all_hpc.sh setup       # env setup only
#   sbatch scripts/run_all_hpc.sh train       # training only
#   sbatch scripts/run_all_hpc.sh eval        # evaluation only
#   sbatch scripts/run_all_hpc.sh analyze     # analysis only
#   sbatch scripts/run_all_hpc.sh all         # full pipeline (default)
#
# For interactive testing:
#   srun -p gpu --gres=gpu -n 1 -N 1 -c 4 --pty /bin/bash
#   bash scripts/run_all_hpc.sh setup
#
# SJSU CoE HPC reference:
#   https://www.sjsu.edu/cmpe/resources/hpc.php
#   Partitions: compute (CPU, 24h max), gpu (P100/A100/H100, 48h max),
#               condo (preemptible, 2 day max)
#   GPU request: --gres=gpu (not --gres=gpu:1)
#   Modules: python3, cuda/10.0, singularity
#   Anaconda: user-installed in ~/anaconda3

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
ENV_NAME="iql"
PYTHON_VERSION="3.11"
ENVIRONMENTS="hopper-medium-v2 halfcheetah-medium-v2 walker2d-medium-v2"
CRITIC_CONFIGS="2 3"
MAX_STEPS=300000
SEEDS="42"
TAU_VALUES="0.5 0.8 0.9"

# ─────────────────────────────────────────────────────────────────────
# PROJECT DIR
# ─────────────────────────────────────────────────────────────────────
if [ -f "scripts/run_all_hpc.sh" ]; then
    PROJECT_DIR="$(pwd)"
elif [ -f "run_all_hpc.sh" ]; then
    PROJECT_DIR="$(cd .. && pwd)"
else
    PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
fi
cd "$PROJECT_DIR"

mkdir -p logs results tmp

echo "============================================"
echo "IQL Robustness Analysis — HPC Job"
echo "============================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURM_NODELIST:-$(hostname)}"
echo "Project dir: $PROJECT_DIR"
echo "Date:        $(date)"
echo "============================================"

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Environment setup
# ─────────────────────────────────────────────────────────────────────
setup_environment() {
    echo ""
    echo ">>> Setting up environment..."

    # Load available HPC modules
    module load python3 2>/dev/null || true
    module load cuda 2>/dev/null || true

    # Initialize conda. On SJSU HPC, Anaconda is user-installed
    # (typically at ~/anaconda3). If conda is not found, we install
    # Miniconda into ~/miniconda3.
    if ! command -v conda &>/dev/null; then
        # Check common install locations
        for CONDA_DIR in ~/anaconda3 ~/miniconda3; do
            if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
                source "${CONDA_DIR}/etc/profile.d/conda.sh"
                break
            fi
        done
    fi

    if ! command -v conda &>/dev/null; then
        echo "Conda not found. Installing Miniconda..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
            -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p ~/miniconda3
        rm /tmp/miniconda.sh
        source ~/miniconda3/etc/profile.d/conda.sh
        echo "Miniconda installed at ~/miniconda3"
    fi

    # Create conda env if it doesn't exist
    if ! conda env list | grep -q "^${ENV_NAME} "; then
        echo "Creating conda environment: ${ENV_NAME} (python ${PYTHON_VERSION})"
        conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    fi

    # Activate — try conda activate first, fall back to source activate
    conda activate "${ENV_NAME}" 2>/dev/null || source activate "${ENV_NAME}"

    # Install dependencies (pip skips already-installed packages)
    echo "Installing Python dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet jax jaxlib flax optax
    pip install --quiet mujoco "gymnasium[mujoco]" gym
    pip install --quiet h5py tqdm matplotlib numpy scipy
    pip install --quiet absl-py ml_collections tensorboardX tensorflow-probability
    pip install --quiet git+https://github.com/Farama-Foundation/d4rl@master 2>/dev/null || true

    # Verify
    python -c "
import jax
print(f'Python:  {__import__(\"sys\").version.split()[0]}')
print(f'JAX:     {jax.__version__}')
print(f'Devices: {jax.devices()}')
"
    echo "Environment ready."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Training (3 envs x 2 critic configs = 6 runs)
# ─────────────────────────────────────────────────────────────────────
run_training() {
    echo ""
    echo ">>> Phase 1: Training"
    echo "    Environments: $ENVIRONMENTS"
    echo "    Critics:      $CRITIC_CONFIGS"
    echo "    Steps:        $MAX_STEPS"
    echo ""

    for env in $ENVIRONMENTS; do
        for nq in $CRITIC_CONFIGS; do
            for seed in $SEEDS; do
                SAVE_DIR="tmp/${env}_${nq}Q_s${seed}"
                echo "--- Training: ${env} | ${nq}Q | seed=${seed} ---"
                python scripts/train_offline.py \
                    --env_name="${env}" \
                    --config=configs/mujoco_config.py \
                    --num_critics="${nq}" \
                    --max_steps="${MAX_STEPS}" \
                    --seed="${seed}" \
                    --save_dir="${SAVE_DIR}"
                echo "    Saved to ${SAVE_DIR}"
            done
        done
    done

    echo "Training complete."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Shift evaluation (all 4 shift types per model)
# ─────────────────────────────────────────────────────────────────────
run_evaluation() {
    echo ""
    echo ">>> Phase 2: Shift Evaluation"
    echo ""

    for env in $ENVIRONMENTS; do
        for nq in $CRITIC_CONFIGS; do
            for seed in $SEEDS; do
                SAVE_DIR="tmp/${env}_${nq}Q_s${seed}"
                echo "--- Evaluating: ${env} | ${nq}Q | seed=${seed} ---"
                python scripts/evaluate_shift.py \
                    --env_name="${env}" \
                    --config=configs/mujoco_config.py \
                    --num_critics="${nq}" \
                    --shift_type=all \
                    --max_steps="${MAX_STEPS}" \
                    --seed="${seed}" \
                    --save_dir="${SAVE_DIR}" \
                    --output_dir=results/
                echo "    Results written to results/"
            done
        done
    done

    echo "Shift evaluation complete."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 4: Expectile tau ablation (2Q only, 3 tau values)
# ─────────────────────────────────────────────────────────────────────
run_ablation() {
    echo ""
    echo ">>> Phase 3: Expectile tau Ablation"
    echo "    Tau values: $TAU_VALUES"
    echo ""

    mkdir -p results/ablation_tau

    for env in $ENVIRONMENTS; do
        for tau in $TAU_VALUES; do
            for seed in $SEEDS; do
                SAVE_DIR="tmp/abl_tau${tau}_${env}_s${seed}"
                echo "--- Ablation: ${env} | tau=${tau} | seed=${seed} ---"

                python scripts/train_offline.py \
                    --env_name="${env}" \
                    --config=configs/mujoco_config.py \
                    --config.expectile="${tau}" \
                    --num_critics=2 \
                    --max_steps="${MAX_STEPS}" \
                    --seed="${seed}" \
                    --save_dir="${SAVE_DIR}"

                python scripts/evaluate_shift.py \
                    --env_name="${env}" \
                    --config=configs/mujoco_config.py \
                    --config.expectile="${tau}" \
                    --num_critics=2 \
                    --shift_type=all \
                    --max_steps="${MAX_STEPS}" \
                    --seed="${seed}" \
                    --save_dir="${SAVE_DIR}" \
                    --output_dir=results/ablation_tau/
            done
        done
    done

    echo "Ablation complete."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 5: Analysis (compute robustness metrics, no GPU needed)
# ─────────────────────────────────────────────────────────────────────
run_analysis() {
    echo ""
    echo ">>> Phase 4: Analysis"
    echo ""

    for env in $ENVIRONMENTS; do
        echo "--- Metrics: ${env} ---"
        python scripts/compute_robustness.py \
            --results_dir=results/ \
            --env_name="${env}"
    done

    echo ""
    echo "Analysis complete. Results in results/"
}

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
MODE="${1:-all}"

case "$MODE" in
    setup)
        setup_environment
        ;;
    train)
        setup_environment
        run_training
        ;;
    eval)
        setup_environment
        run_evaluation
        ;;
    ablation)
        setup_environment
        run_ablation
        ;;
    analyze)
        setup_environment
        run_analysis
        ;;
    all)
        setup_environment
        run_training
        run_evaluation
        run_ablation
        run_analysis
        ;;
    *)
        echo "Usage: sbatch scripts/run_all_hpc.sh [setup|train|eval|ablation|analyze|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Job finished at $(date)"
echo ""
echo "Results:"
ls -la results/ 2>/dev/null || echo "  (no results directory)"
echo ""
echo "To copy results to your local machine:"
echo "  scp -r $(whoami)@coe-hpc1.sjsu.edu:${PROJECT_DIR}/results/ ./results/"
echo "============================================"
