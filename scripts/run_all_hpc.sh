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

    # GPU check — report JAX backend immediately
    echo ""
    echo "--- JAX Backend Check ---"
    python -c "
import jax
devices = jax.devices()
gpu_devs = [d for d in devices if d.platform == 'gpu']
if gpu_devs:
    print(f'  GPU ENABLED: {len(gpu_devs)} GPU(s) detected')
    for d in gpu_devs:
        print(f'    {d}')
else:
    print('  CPU ONLY: No GPU detected by JAX')
    print('  (Re-run setup with CUDA to enable GPU acceleration)')
print(f'  JAX version: {jax.__version__}')
" 2>/dev/null || echo "  WARNING: JAX import failed"
    echo "-------------------------"
    echo ""
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
        jax==0.4.35 ml_dtypes==0.4.1 \
        mujoco==3.1.6 matplotlib==3.9.2 \
        flax==0.8.5 optax==0.2.3 \
        tensorflow-probability==0.23.0

    echo ""
    echo "  Installing from downloaded wheels..."
    pip install --no-index --find-links="$WHEEL_DIR" \
        numpy==1.26.4 scipy==1.13.1 h5py==3.11.0 \
        jax==0.4.35 ml_dtypes==0.4.1 \
        mujoco==3.1.6 matplotlib==3.9.2 \
        flax==0.8.5 optax==0.2.3 \
        tensorflow-probability==0.23.0

    # Install jaxlib with CUDA support for GPU acceleration.
    # Strategy: always try CUDA 12 first (works on login node without GPU),
    # then CUDA 11, then fall back to CPU-only.
    # The CUDA jaxlib can be installed on a CPU-only machine — JAX will
    # simply use GPU when one becomes available on the batch node.
    echo ""
    echo "  Installing jaxlib with CUDA support..."
    echo "  (Trying CUDA 12 -> CUDA 11 -> CPU-only)"
    echo ""

    JAXLIB_INSTALLED=0

    # Temporarily disable exit-on-error for CUDA install attempts
    # (pip install will fail if CUDA wheels are incompatible with GLIBC 2.17)
    set +e

    # Try CUDA 12 (most common on modern HPC)
    if [ "$JAXLIB_INSTALLED" -eq 0 ]; then
        echo "  Attempting: jaxlib==0.4.35+cuda12 ..."
        pip install "jax[cuda12_pip]==0.4.35" \
            -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
            2>&1
        if python -c "import jaxlib; v=jaxlib.__version__; print(v); assert 'cuda' in v.lower() or 'cu12' in v.lower()" 2>/dev/null; then
            echo "  SUCCESS: CUDA 12 jaxlib installed"
            JAXLIB_INSTALLED=1
        else
            echo "  CUDA 12 attempt did not produce a CUDA jaxlib."
        fi
    fi

    # Try CUDA 11
    if [ "$JAXLIB_INSTALLED" -eq 0 ]; then
        echo ""
        echo "  Attempting: jaxlib==0.4.35+cuda11 ..."
        pip install "jax[cuda11_pip]==0.4.35" \
            -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
            2>&1
        if python -c "import jaxlib; v=jaxlib.__version__; print(v); assert 'cuda' in v.lower() or 'cu11' in v.lower()" 2>/dev/null; then
            echo "  SUCCESS: CUDA 11 jaxlib installed"
            JAXLIB_INSTALLED=1
        else
            echo "  CUDA 11 attempt did not produce a CUDA jaxlib."
        fi
    fi

    # Fallback: CPU-only (jaxlib was already installed from .wheels earlier)
    if [ "$JAXLIB_INSTALLED" -eq 0 ]; then
        echo ""
        echo "  WARNING: CUDA jaxlib install failed."
        echo "  Using CPU-only jaxlib (already installed). Training will be slower."
        echo "  The SJSU HPC may have GLIBC 2.17 which is too old for CUDA JAX wheels."
    fi

    # Re-enable exit-on-error
    set -e

    # Report what was installed
    echo ""
    echo "  Installed jaxlib version:"
    pip show jaxlib 2>/dev/null | grep -i version
    echo ""

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
    #
    # D4RL v2 filename convention:
    #   env name:     hopper-medium-v2
    #   URL filename: hopper_medium-v2.hdf5  (underscore between env and dataset)
    #   URL: http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5
    #
    # Uses curl instead of Python urllib to avoid SJSU Proofpoint URL proxy.
    echo ""
    echo "Pre-downloading D4RL datasets (GPU nodes have no internet)..."
    CACHE_DIR="$HOME/.d4rl/datasets"
    mkdir -p "$CACHE_DIR"

    for ds_file in hopper_medium-v2.hdf5 halfcheetah_medium-v2.hdf5 walker2d_medium-v2.hdf5; do
        FPATH="${CACHE_DIR}/${ds_file}"
        if [ -f "$FPATH" ]; then
            SIZE=$(du -h "$FPATH" | cut -f1)
            echo "  ${ds_file}: already cached (${SIZE})"
        else
            URL="http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/${ds_file}"
            echo "  ${ds_file}: downloading from ${URL} ..."
            curl -L -o "$FPATH" "$URL"
            SIZE=$(du -h "$FPATH" | cut -f1)
            echo "  ${ds_file}: saved (${SIZE})"
        fi
    done

    # Verify datasets were downloaded
    echo ""
    echo "Verifying dataset cache..."
    MISSING=0
    for ds_file in hopper_medium-v2.hdf5 halfcheetah_medium-v2.hdf5 walker2d_medium-v2.hdf5; do
        FPATH="${CACHE_DIR}/${ds_file}"
        if [ -f "$FPATH" ]; then
            SIZE=$(du -h "$FPATH" | cut -f1)
            echo "  OK  ${ds_file} (${SIZE})"
        else
            echo "  MISSING  ${ds_file}"
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
