#!/usr/bin/env bash
#SBATCH --job-name=tendencies
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/tendencies-%A_%a.out
#SBATCH --error=logs/tendencies-%A_%a.err
#
# One stage of the tendency statistics for one dataset. Driven by
# bris/slurm/tendencies.sh, which submits the three stages in order.
#
#   sbatch bris/slurm/tendencies.sbatch init     <dataset>
#   sbatch --array=1-10 bris/slurm/tendencies.sbatch load <dataset> 10
#   sbatch bris/slurm/tendencies.sbatch finalise <dataset>
#
# WHAT THESE STATISTICS ARE FOR. anemoi's loss weights each variable by how
# much it changes over one timestep, so that a field which barely moves does
# not compete for the model's attention with one that does. Without them every
# variable is weighted equally, and geopotential at 50 hPa counts as much as
# precipitation. The datasets built here have none, and the training run says
# so:
#
#   Dataset has no tendency statistics! Are you sure you want to use a
#   tendency scaler?
#
# It is not fatal and it does not bias the comparison, since both arms are
# affected identically. It is one of two places where this setup departs from
# the objective Bris was pretrained under, and it is the one that can be fixed.
#
# THE DELTA IS THE MODEL'S TIMESTEP. Six hours, matching data.frequency and the
# rollout step. A different delta would compute a real quantity that the loss
# would then not find.
#
# PARTS RUN IN PARALLEL ON PURPOSE. Each writes its own slice, which is what
# --parts exists for. The finalise stage combines them and must not start until
# every part has finished, which is what the dependency in tendencies.sh is for.

set -uo pipefail

STAGE="${1:-}"
DATASET="${2:-}"
NPARTS="${3:-10}"
DELTA="${DELTA:-6h}"

[[ -n "$STAGE" && -n "$DATASET" ]] || {
    echo "usage: sbatch $0 <init|load|finalise> <dataset.zarr> [nparts]" >&2
    exit 1
}
[[ -d "$DATASET" ]] || { echo "ERROR: no dataset at $DATASET" >&2; exit 1; }

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
[[ -f "$REPO_DIR/scripts/env.sh" ]] && source "$REPO_DIR/scripts/env.sh"
ANEMOI="${BRIS_ENV_DIR:-$HOME/bris-env}/.venv/bin/anemoi-datasets"
[[ -x "$ANEMOI" ]] || { echo "ERROR: missing $ANEMOI" >&2; exit 1; }

export PYTHONUNBUFFERED=1

echo "=== stage   : $STAGE"
echo "=== dataset : $(basename "$DATASET")"
echo "=== delta   : $DELTA"
echo "=== node    : $(hostname)"

start=$(date +%s)
case "$STAGE" in
    init)
        "$ANEMOI" init-additions "$DATASET" --delta "$DELTA"
        ;;
    load)
        PART="${SLURM_ARRAY_TASK_ID:-1}/$NPARTS"
        echo "=== part    : $PART"
        "$ANEMOI" load-additions "$DATASET" --delta "$DELTA" --parts "$PART"
        ;;
    finalise)
        "$ANEMOI" finalise-additions "$DATASET" --delta "$DELTA"
        ;;
    *)
        echo "ERROR: unknown stage $STAGE" >&2; exit 1 ;;
esac
rc=$?
elapsed=$(( $(date +%s) - start ))
printf '\n=== finished rc=%d in %02d:%02d:%02d\n' \
    "$rc" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
exit $rc
