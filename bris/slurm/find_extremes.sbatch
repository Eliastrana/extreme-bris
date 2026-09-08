#!/usr/bin/env bash
#SBATCH --job-name=find-extremes
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/extremes-%j.out
#SBATCH --error=logs/extremes-%j.err
#
# Rank every state of a dataset by how extreme its precipitation is.
#
#   mkdir -p logs
#   sbatch --export=ALL,DATASET=$BRIS_DATA_DIR/meps-2p5km-year1-6h-v1.zarr,\
# OUT=$HOME/bris-runs/extremes/meps-year1.json \
#     bris/slurm/find_extremes.sbatch
#
# WHY A JOB. Every state is one zarr chunk holding all 98 variables, so asking
# for precipitation alone still decompresses the whole chunk. A year is around
# 257 GB of reads. That is not a login-node task.
#
# WORKERS. The cost is reading and decompressing, and blosc releases the GIL,
# so threads help. Sixteen CPUs with eight workers leaves room for the
# decompression itself.

set -uo pipefail

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
[[ -f "$REPO_DIR/scripts/env.sh" ]] && source "$REPO_DIR/scripts/env.sh"
DATA_ENV="${BRIS_DATA_ENV_DIR:-$HOME/bris-data-env}"

: "${DATASET:?set DATASET=<input.zarr>}"
: "${OUT:?set OUT=<output.json>}"
WORKERS="${WORKERS:-8}"
STRIDE="${STRIDE:-20}"
VAR="${VAR:-tp}"

[[ -d "$DATASET" ]] || { echo "ERROR: no dataset at $DATASET" >&2; exit 1; }

echo "=== node    : $(hostname)"
echo "=== dataset : $DATASET"
echo "=== out     : $OUT"
echo "=== var     : $VAR   stride: $STRIDE   workers: $WORKERS"

start=$(date +%s)
"$DATA_ENV/bin/python" "$REPO_DIR/scripts/find_extreme_states.py" \
    "$DATASET" -o "$OUT" --var "$VAR" --stride "$STRIDE" --workers "$WORKERS"
rc=$?
elapsed=$(( $(date +%s) - start ))
printf '\n=== finished rc=%d in %02d:%02d:%02d\n' \
    "$rc" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))

# rc=3 is the zero gate, and it is a real outcome rather than a crash: the
# field is present, finite and empty of weather. Say so in words, because
# "job failed" sends you looking for a bug in the ranking instead of in the
# dataset it ranked.
case $rc in
  0) echo "=== OK" ;;
  3) echo "=== NO PRECIPITATION IN THIS DATASET - nothing to rank" ;;
  *) echo "=== FAILED (rc=$rc)" ;;
esac
exit $rc
