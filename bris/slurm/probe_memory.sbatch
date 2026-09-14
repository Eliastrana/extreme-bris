#!/usr/bin/env bash
#SBATCH --job-name=probe-mem
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/probe-mem-%j.out
#SBATCH --error=logs/probe-mem-%j.err
#
# How many ensemble members fit on one card, and how long a step takes.
#
#   mkdir -p logs && sbatch bris/slurm/probe_memory.sbatch
#   SIZES="4 3 2" sbatch bris/slurm/probe_memory.sbatch
#
# WHY ONE JOB TRIES SEVERAL SIZES. A card on this cluster has taken anywhere
# from thirty seconds to a quoted six days. Submitting one ensemble size per job
# and waiting for each verdict in turn could take weeks to learn one number.
# So this tries the largest size first, stops at the first that runs, and only
# falls back to a smaller one when the larger ran out of memory. Any other
# failure stops it, because a smaller ensemble would not fix a broken config and
# the slot is better spent reading the traceback.
#
# WHAT IT RECORDS. Peak card memory for each attempt, sampled by nvidia-smi
# while it runs, and the time per step from the last attempt. Those two numbers
# decide the real run: how many members it can afford, and how many steps fit
# in a walltime.
#
# Run scripts/dry_run_training.py first. Nothing here should fail before the
# first batch; if it does, that was a queue slot spent on something the login
# node could have told you.

set -uo pipefail

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
SIZES="${SIZES:-4 2}"
STEPS="${STEPS:-5}"
ARM="${ARM:-finetune_tail}"
cd "$REPO_DIR" || exit 1
mkdir -p logs

echo "=== node   : $(hostname)"
echo "=== card   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=== sizes  : $SIZES (largest first, stopping at the first that fits)"
echo "=== steps  : $STEPS per attempt"

verdict="none fit"
for n in $SIZES; do
    attempt="logs/probe-mem-${SLURM_JOB_ID:-local}-ens$n.log"
    samples="$(mktemp)"
    echo
    echo "=== ensemble $n"

    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "$samples" &
    smi=$!

    start=$(date +%s)
    bash bris/slurm/finetune.sbatch "$ARM" \
        training.max_steps="$STEPS" \
        training.ensemble_size_per_device="$n" \
        dataloader.limit_batches.training="$STEPS" \
        dataloader.limit_batches.validation=1 \
        > "$attempt" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start ))

    kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
    peak=$(sort -n "$samples" | tail -1); rm -f "$samples"

    echo "  exit $rc after ${elapsed}s, peak card memory ${peak:-?} MiB"
    rate=$(grep -oE '[0-9.]+(s/it|it/s)' "$attempt" | tail -1)
    [[ -n "$rate" ]] && echo "  last reported rate: $rate"
    grep -E "tail term, forward" "$attempt" | tail -2 | sed 's/^/  /'

    if [[ $rc -eq 0 ]]; then
        verdict="ensemble $n fits"
        break
    fi
    if grep -q "OutOfMemoryError" "$attempt"; then
        echo "  out of memory; trying the next size down"
        continue
    fi
    echo "  failed for a reason other than memory. Stopping; read $attempt"
    verdict="stopped at ensemble $n: not a memory failure"
    break
done

echo
echo "=== VERDICT: $verdict"
