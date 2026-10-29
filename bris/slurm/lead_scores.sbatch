#!/usr/bin/env bash
#SBATCH --job-name=lead-scores
#SBATCH --propagate=NONE
#SBATCH --partition=fpgaq,defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/lead-scores-%j.out
#SBATCH --error=logs/lead-scores-%j.err
#
# Six-hour rain, lead by lead, on the test year (evaluation/lead_plan.json):
#
#   sbatch bris/slurm/lead_scores.sbatch
#
# Override paths with BRIS_EVALUATION_DIR, BRIS_OBSERVATIONS_DIR,
# BRIS_MEPS_MEMBERS_DIR or BRIS_LEADS_DIR.

set -euo pipefail

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
source "$REPO_DIR/scripts/env.sh"

EVAL="${BRIS_EVALUATION_DIR:-$HOME/bris-runs/evaluation}"
OUT="${BRIS_LEADS_DIR:-$HOME/bris-runs/leads}"
mkdir -p "$OUT"

python3 "$REPO_DIR/scripts/lead_scores.py" \
  --model "baseline=$EVAL/baseline/nordic_*.nc" \
  --model "control2000=$EVAL/control2000/nordic_*.nc" \
  --model "controlfft2000=$EVAL/controlfft2000/nordic_*.nc" \
  --model "tailA2000=$EVAL/tailA2000/nordic_*.nc" \
  --model "tailC2000=$EVAL/tailC2000/nordic_*.nc" \
  --model "tailAfft2000=$EVAL/tailAfft2000/nordic_*.nc" \
  --comparator controlfft2000 --primary tailC2000 \
  --meps-members "${BRIS_MEPS_MEMBERS_DIR:-$HOME/bris-runs/meps-ens}" \
  --observations "${BRIS_OBSERVATIONS_DIR:-$HOME/bris-runs/observations}" \
  --cache "$OUT/cases.npz" --out "$OUT/report.json" \
  2>&1 | tee "$OUT/report.log"
