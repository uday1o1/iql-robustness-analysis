#!/bin/bash
# HPC convenience aliases for IQL Robustness Analysis
#
# Source this file on the HPC login node:
#   source scripts/hpc_aliases.sh
#
# Or add to your ~/.bashrc:
#   echo 'source ~/iql-robustness-analysis/scripts/hpc_aliases.sh' >> ~/.bashrc

# ─── Job management ──────────────────────────────────────────────────
alias jobs='squeue -u $USER'
alias myjobs='squeue -u $USER --format="%.8i %.20j %.8T %.10M %.6D %R"'
alias history='sacct -u $USER --format=JobID,JobName%20,State,ExitCode,Elapsed,Start --starttime=$(date -d "7 days ago" +%Y-%m-%d 2>/dev/null || date -v-7d +%Y-%m-%d)'
alias killall='scancel -u $USER'

# ─── Quick submit ────────────────────────────────────────────────────
alias iql-setup='bash scripts/run_all_hpc.sh setup'
alias iql-verify='python scripts/verify_env.py'
alias iql-validate='python scripts/validate_pipeline.py'
alias iql-run='sbatch scripts/run_all_hpc.sh'
alias iql-train='sbatch scripts/run_all_hpc.sh train'
alias iql-eval='sbatch scripts/run_all_hpc.sh eval'
alias iql-analyze='sbatch scripts/run_all_hpc.sh analyze'

# ─── Log viewing ─────────────────────────────────────────────────────
alias lastlog='ls -t logs/slurm_*.out 2>/dev/null | head -1 | xargs tail -f'
alias lasterr='ls -t logs/slurm_*.err 2>/dev/null | head -1 | xargs tail -f'
alias alllogs='ls -lt logs/slurm_*.out 2>/dev/null | head -10'
alias clearlogs='rm -f logs/slurm_*.out logs/slurm_*.err logs/*.out logs/*.err && echo "Logs cleared"'

# ─── Cleanup ─────────────────────────────────────────────────────────
alias cleanvenv='rm -rf venv && echo "venv deleted. Run iql-setup to recreate."'
alias cleanall='rm -rf venv tmp logs/slurm_* && echo "Cleaned venv, tmp, and logs"'

# ─── Results ─────────────────────────────────────────────────────────
alias results='ls -la results/*.csv 2>/dev/null'
alias checkpoints='ls -la tmp/*/checkpoint_* 2>/dev/null'

# ─── GPU node interactive session ────────────────────────────────────
alias gpunode='srun -p gpu --gres=gpu -n 1 -N 1 -c 4 --mem=16G --time=01:00:00 --pty /bin/bash'

# ─── Cluster info ────────────────────────────────────────────────────
alias nodes='sinfo -N -l'
alias gpus='sinfo -p gpu -N -l'
alias quota='df -h /home/$USER'

echo "IQL HPC aliases loaded. Commands:"
echo "  jobs / myjobs     — check job status"
echo "  killall           — cancel all your jobs"
echo "  iql-setup         — one-time environment setup"
echo "  iql-verify        — verify all deps are working"
echo "  iql-run           — submit full pipeline"
echo "  iql-train         — submit training only"
echo "  iql-eval          — submit evaluation only"
echo "  lastlog / lasterr — tail latest log/error"
echo "  clearlogs         — delete all log files"
echo "  cleanvenv         — delete venv"
echo "  cleanall          — delete venv + tmp + logs"
echo "  gpunode           — get interactive GPU session"
