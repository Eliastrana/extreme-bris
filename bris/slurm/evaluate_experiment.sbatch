#!/usr/bin/env bash
#SBATCH --job-name=bris-score
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/bris-score-%j.out
#SBATCH --error=logs/bris-score-%j.err
#
# Joint scoring of baseline, control, tail, and deterministic MEPS on the
# frozen common-case evaluation plan.  Submit after all three inference jobs:
#
#   sbatch --dependency=afterok:${BASE_JOB}:${CONTROL_JOB}:${TAIL_JOB} \
#     bris/slurm/evaluate_experiment.sbatch
#
# Override any path with BRIS_BASELINE_DIR, BRIS_CONTROL_DIR, BRIS_TAIL_DIR,
# BRIS_MEPS_FILE, BRIS_OBSERVATIONS_DIR, or BRIS_EVALUATION_SCORE_DIR.

set -euo pipefail

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
source "$REPO_DIR/scripts/env.sh"

BASELINE="${BRIS_BASELINE_DIR:-$HOME/bris-runs/evaluation/baseline}"
CONTROL="${BRIS_CONTROL_DIR:-$HOME/bris-runs/evaluation/control}"
TAIL="${BRIS_TAIL_DIR:-$HOME/bris-runs/evaluation/tail}"
MEPS="${BRIS_MEPS_FILE:-$HOME/bris-runs/meps/precipitation_daily.npz}"
OBSERVATIONS="${BRIS_OBSERVATIONS_DIR:-$HOME/bris-runs/observations}"
OUT="${BRIS_EVALUATION_SCORE_DIR:-$HOME/bris-runs/evaluation-score}"
PYTHON="${BRIS_ENV_DIR:-$HOME/bris-env}/.venv/bin/python"

mkdir -p logs "$OUT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

echo "=== node         : $(hostname)"
echo "=== baseline     : $BASELINE"
echo "=== control      : $CONTROL"
echo "=== tail         : $TAIL"
echo "=== MEPS         : $MEPS"
echo "=== observations : $OBSERVATIONS"
echo "=== output       : $OUT"
echo

"$PYTHON" "$REPO_DIR/scripts/evaluate_experiment.py" \
  --model "baseline=$BASELINE/nordic_*.nc" \
  --model "control=$CONTROL/nordic_*.nc" \
  --model "tail=$TAIL/nordic_*.nc" \
  --meps "$MEPS" \
  --observations "$OBSERVATIONS" \
  --out-dir "$OUT"
