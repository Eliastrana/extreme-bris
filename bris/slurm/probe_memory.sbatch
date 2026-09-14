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
# Find a configuration that fits on one card, and what a step costs there.
#
#   mkdir -p logs && sbatch bris/slurm/probe_memory.sbatch
#
#   CONFIGS="training.ensemble_size_per_device=4;
#            training.ensemble_size_per_device=2;
#            training.ensemble_size_per_device=2 model.processor.cpu_offload=True" \
#   EXTRA="model.encoder.num_chunks=32 model.decoder.num_chunks=32" \
#   ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=32 \
#   sbatch bris/slurm/probe_memory.sbatch
#
# CONFIGS is a semicolon-separated list of override sets, tried in order.
# EXTRA is added to every one of them. Without CONFIGS it falls back to SIZES,
# a list of ensemble sizes, which is how it started.
#
# WHY ONE JOB TRIES SEVERAL. A card on this cluster has taken anywhere from
# thirty seconds to a quoted six days. One configuration per job, waiting for
# each verdict in turn, could take weeks to learn one number. So this tries the
# most wanted configuration first, stops at the first that runs, and only moves
# on when the previous one ran out of memory. Any other failure stops it,
# because a smaller configuration would not fix a broken one and the slot is
# better spent reading the traceback.
#
# WHAT IT RECORDS. Peak card memory for each attempt, sampled by nvidia-smi
# while it runs; the last training rate and loss it reported; and the tail
# term's own value, which is what sets its weight in the combined loss.
#
# Run scripts/dry_run_training.py with the same overrides first. Nothing here
# should fail before the first batch.

set -uo pipefail

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
STEPS="${STEPS:-5}"
ARM="${ARM:-finetune_tail}"
EXTRA="${EXTRA:-}"
cd "$REPO_DIR" || exit 1
mkdir -p logs

if [[ -n "${CONFIGS:-}" ]]; then
    IFS=';' read -ra attempts <<< "$CONFIGS"
else
    attempts=()
    for n in ${SIZES:-4 2}; do
        attempts+=("training.ensemble_size_per_device=$n")
    done
fi
read -ra extra_args <<< "$EXTRA"

echo "=== node    : $(hostname)"
echo "=== card    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=== steps   : $STEPS per attempt"
echo "=== extra   : ${EXTRA:-none}"
echo "=== mapper chunks outside training: ${ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER:-as ANEMOI_INFERENCE_NUM_CHUNKS}"
for i in "${!attempts[@]}"; do
    echo "=== try $((i + 1))   : $(echo "${attempts[$i]}" | xargs)"
done

verdict="none fit"
for i in "${!attempts[@]}"; do
    read -ra spec_args <<< "${attempts[$i]}"
    attempt="logs/probe-mem-${SLURM_JOB_ID:-local}-try$((i + 1)).log"
    samples="$(mktemp)"
    echo
    echo "=== try $((i + 1)): ${spec_args[*]}"

    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "$samples" &
    smi=$!

    start=$(date +%s)
    bash bris/slurm/finetune.sbatch "$ARM" \
        training.max_steps="$STEPS" \
        dataloader.limit_batches.training="$STEPS" \
        dataloader.limit_batches.validation=1 \
        ${extra_args[@]+"${extra_args[@]}"} \
        ${spec_args[@]+"${spec_args[@]}"} \
        > "$attempt" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start ))

    kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
    peak=$(sort -n "$samples" | tail -1); rm -f "$samples"

    echo "  exit $rc after ${elapsed}s, peak card memory ${peak:-?} MiB"
    # Rates from the sanity check are validation, not training; only trust
    # the ones on Epoch lines.
    rate=$(grep -E "^Epoch|Epoch 0" "$attempt" | grep -oE '[0-9.]+(s/it|it/s)' | tail -1)
    [[ -n "$rate" ]] && echo "  last training rate: $rate"
    loss=$(grep -oE 'train_[a-z_]*loss[a-z_]*=[0-9.eE+-]+' "$attempt" | tail -2 | xargs)
    [[ -n "$loss" ]] && echo "  last training loss: $loss"
    grep -E "tail term, forward" "$attempt" | tail -3 | sed -E 's/.*reported: /  tail term: /; s/ \(before.*//'

    if [[ $rc -eq 0 ]]; then
        verdict="try $((i + 1)) fits: ${spec_args[*]}"
        break
    fi
    if grep -q "OutOfMemoryError" "$attempt"; then
        where=$(grep -oE 'anemoi/models/(layers|models)/[a-z_]+\.py", line [0-9]+' "$attempt" | tail -1)
        echo "  out of memory${where:+ at $where}; trying the next"
        continue
    fi
    echo "  failed for a reason other than memory. Stopping; read $attempt"
    verdict="stopped at try $((i + 1)): not a memory failure"
    break
done

echo
echo "=== VERDICT: $verdict"
