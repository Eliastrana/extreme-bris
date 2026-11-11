#!/usr/bin/env bash
#SBATCH --job-name=long-lead-scores
#SBATCH --propagate=NONE
#SBATCH --partition=fpgaq,defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/long-lead-scores-%j.out
#SBATCH --error=logs/long-lead-scores-%j.err
#
# Six-hour rain from +6 to +60 h, every second day of the test year
# (evaluation/long_lead_plan.json):
#
#   sbatch bris/slurm/long_lead_scores.sbatch
#
# Bris is the existing 60 h baseline run, cut to the same days by the cycle
# list; the other four models are the forecasts in ~/bris-runs/long-lead.

set -euo pipefail

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
source "$REPO_DIR/scripts/env.sh"

LL="${BRIS_LONG_LEAD_DIR:-$HOME/bris-runs/long-lead}"
OUT="${BRIS_LONG_LEAD_OUT:-$HOME/bris-runs/long-lead/scores}"
mkdir -p "$OUT"

python3 "$REPO_DIR/scripts/lead_scores.py" \
  --model "baseline=$HOME/bris-runs/evaluation/baseline/nordic_*.nc" \
  --model "control2000=$LL/control2000/nordic_*.nc" \
  --model "controlfft2000=$LL/controlfft2000/nordic_*.nc" \
  --model "tailA2000=$LL/tailA2000/nordic_*.nc" \
  --model "tailC2000=$LL/tailC2000/nordic_*.nc" \
  --comparator controlfft2000 --primary tailC2000 \
  --leads 6,12,18,24,30,36,42,48,54,60 --step-days 2 --no-meps \
  --plan evaluation/long_lead_plan.json \
  --meps-members "${BRIS_MEPS_MEMBERS_DIR:-$HOME/bris-runs/meps-ens}" \
  --observations "${BRIS_OBSERVATIONS_DIR:-$HOME/bris-runs/observations}" \
  --cache "$OUT/cases.npz" --out "$OUT/report.json" \
  2>&1 | tee "$OUT/report.log"
