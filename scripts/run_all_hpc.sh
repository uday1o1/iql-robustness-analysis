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
# FIRST TIME SETUP (run on the login node — has internet):
#   bash scripts/run_all_hpc.sh setup
#
# Then submit experiments:
#   sbatch scripts/run_all_hpc.sh            # full pipeline
#   sbatch scripts/run_all_hpc.sh train      # training only
#   sbatch scripts/run_all_hpc.sh eval       # evaluation only
#   sbatch scripts/run_all_hpc.sh analyze    # analysis only
#
# SJSU HPC notes:
#   - Login node: GLIBC 2.17 (CentOS 7), has internet, no GCC
#   - GPU nodes:  GLIBC 2.17, no internet, have GPU
#   - /home is shared across all nodes
#   - Setup downloads pre-built wheels on login node (no compilation)
#   - Batch jobs use the venv created during setup

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
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

VENV_DIR="${PROJECT_DIR}/venv"

echo "============================================"
echo "IQL Robustness Analysis — HPC Job"
echo "============================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURM_NODELIST:-$(hostname)}"
echo "Project dir: $PROJECT_DIR"
echo "Date:        $(date)"
echo "============================================"

# ─────────────────────────────────────────────────────────────────────
# ENVIRONMENT ACTIVATION (used by batch jobs)
# ─────────────────────────────────────────────────────────────────────
activate_env() {
    module load python3 2>/dev/null || true
    module load cuda 2>/dev/null || true

    if [ -f "${VENV_DIR}/bin/activate" ]; then
        source "${VENV_DIR}/bin/activate"
        echo "Activated venv: $(python --version)"
    else
        echo "ERROR: venv not found at ${VENV_DIR}"
        echo "Run setup first on the login node:"
        echo "  bash scripts/run_all_hpc.sh setup"
        exit 1
    fi

    # Ensure project root is on PYTHONPATH for 'import iql'
    export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
    # Suppress D4RL warnings for envs we don't use (mujoco_py, flow, etc.)
    export D4RL_SUPPRESS_IMPORT_ERROR=1
}

# ─────────────────────────────────────────────────────────────────────
# STEP 0: One-time setup (run on LOGIN NODE — has internet)
# ─────────────────────────────────────────────────────────────────────
setup_environment() {
    echo ""
    echo ">>> One-time environment setup"
    echo ">>> Run this on the LOGIN NODE (has internet access)"
    echo ""

    module load python3 2>/dev/null || true

    echo "Python: $(python3 --version 2>&1)"
    echo "Node:   $(hostname)"
    echo ""

    # Create venv
    if [ ! -d "${VENV_DIR}" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "${VENV_DIR}"
    fi

    source "${VENV_DIR}/bin/activate"
    echo "Activated venv: $(which python)"

    # CRITICAL: upgrade pip first. The system pip (23.2.1) may not
    # properly handle --only-binary and manylinux2014 wheel resolution.
    echo "Upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel

    echo ""
    echo "Installing dependencies..."
    echo ""

    # Download all wheels first (binary only, no source builds).
    # This ensures pip never tries to compile anything.
    WHEEL_DIR="${PROJECT_DIR}/.wheels"
    mkdir -p "$WHEEL_DIR"

    echo "  Downloading binary wheels..."
    pip download --only-binary=:all: --dest "$WHEEL_DIR" \
        numpy==1.26.4 scipy==1.13.1 h5py==3.11.0 \
        jax==0.4.35 jaxlib==0.4.35 ml_dtypes==0.4.1 \
        mujoco==3.1.6 matplotlib==3.9.2 \
        flax==0.8.5 optax==0.2.3 \
        tensorflow-probability==0.23.0

    echo ""
    echo "  Installing from downloaded wheels..."
    pip install --no-index --find-links="$WHEEL_DIR" \
        numpy==1.26.4 scipy==1.13.1 h5py==3.11.0 \
        jax==0.4.35 jaxlib==0.4.35 ml_dtypes==0.4.1 \
        mujoco==3.1.6 matplotlib==3.9.2 \
        flax==0.8.5 optax==0.2.3 \
        tensorflow-probability==0.23.0

    # Pure Python packages — install normally from PyPI
    echo ""
    echo "  Installing pure Python packages..."
    pip install \
        gymnasium==0.29.1 imageio \
        absl-py==2.1.0 ml_collections==0.1.1 \
        tensorboardX==2.6.2.2 tqdm==4.66.5

    # gym (legacy API, needed by D4RL)
    echo "  Installing gym..."
    pip install "gym==0.23.1" 2>/dev/null || pip install gym 2>/dev/null || true

    # D4RL — install without its heavy deps (mujoco-py, etc.)
    echo ""
    echo "Installing D4RL (no-deps to avoid mujoco-py compilation)..."
    pip install --no-deps git+https://github.com/Farama-Foundation/d4rl@master 2>/dev/null || \
        pip install --no-deps d4rl 2>/dev/null || \
        echo "WARNING: D4RL install failed"

    # Install the IQL package itself in editable mode so 'import iql' works
    echo ""
    echo "Installing IQL package (editable)..."
    pip install -e "${PROJECT_DIR}" 2>/dev/null || \
        echo "WARNING: editable install failed — will use PYTHONPATH instead"

    # Set D4RL env var to suppress import warnings for envs we don't use
    export D4RL_SUPPRESS_IMPORT_ERROR=1

    # Pre-download D4RL datasets (GPU nodes have no internet)
    # Downloads HDF5 files directly to ~/.d4rl/datasets/ so the
    # cache-first path in _load_d4rl_dataset() finds them on GPU nodes.
    echo ""
    echo "Pre-downloading D4RL datasets (GPU nodes have no internet)..."
    python -c "
import os, urllib.request
cache_dir = os.path.join(os.path.expanduser('~'), '.d4rl', 'datasets')
os.makedirs(cache_dir, exist_ok=True)
for name in ['hopper-medium-v2', 'halfcheetah-medium-v2', 'walker2d-medium-v2']:
    path = os.path.join(cache_dir, f'{name}.hdf5')
    if os.path.exists(path):
        sz = os.path.getsize(path) / (1024*1024)
        print(f'  {name}: already cached ({sz:.1f} MB)')
    else:
        url = f'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/{name}.hdf5'
        print(f'  {name}: downloading from {url} ...')
        urllib.request.urlretrieve(url, path)
        sz = os.path.getsize(path) / (1024*1024)
        print(f'  {name}: saved ({sz:.1f} MB)')
"

    # Verify datasets were downloaded
    echo ""
    echo "Verifying dataset cache..."
    CACHE_DIR="$HOME/.d4rl/datasets"
    MISSING=0
    for ds in hopper-medium-v2 halfcheetah-medium-v2 walker2d-medium-v2; do
        if [ -f "${CACHE_DIR}/${ds}.hdf5" ]; then
            SIZE=$(du -h "${CACHE_DIR}/${ds}.hdf5" | cut -f1)
            echo "  OK  ${ds}.hdf5 (${SIZE})"
        else
            echo "  MISSING  ${ds}.hdf5"
            MISSING=$((MISSING + 1))
        fi
    done
    if [ "$MISSING" -gt 0 ]; then
        echo "ERROR: ${MISSING} dataset(s) missing. Cannot proceed."
        exit 1
    fi
    echo "All datasets cached."

    # Run full pipeline validation
    echo ""
    echo "Running full pipeline validation..."
    python scripts/validate_pipeline.py

    echo ""
    echo "============================================"
    echo "Setup complete. Next steps:"
    echo "  python scripts/validate_pipeline.py  # re-validate anytime"
    echo "  sbatch scripts/run_all_hpc.sh        # submit full pipeline"
    echo "============================================"
}

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Training (3 envs x 2 critic configs = 6 runs)
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
# STEP 2: Shift evaluation (all 4 shift types per model)
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
# STEP 3: Expectile tau ablation (2Q only, 3 tau values)
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
# STEP 4: Analysis (compute robustness metrics)
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
        activate_env
        run_training
        ;;
    eval)
        activate_env
        run_evaluation
        ;;
    ablation)
        activate_env
        run_ablation
        ;;
    analyze)
        activate_env
        run_analysis
        ;;
    all)
        activate_env
        run_training
        run_evaluation
        run_ablation
        run_analysis
        ;;
    *)
        echo "Usage:"
        echo "  First time:  bash scripts/run_all_hpc.sh setup    (on login node)"
        echo "  Then:        sbatch scripts/run_all_hpc.sh         (full pipeline)"
        echo "               sbatch scripts/run_all_hpc.sh train"
        echo "               sbatch scripts/run_all_hpc.sh eval"
        echo "               sbatch scripts/run_all_hpc.sh ablation"
        echo "               sbatch scripts/run_all_hpc.sh analyze"
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
