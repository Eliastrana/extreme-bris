#!/usr/bin/env bash
#SBATCH --job-name=bris-fc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=h200q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/bris-fc-%j.out
#SBATCH --error=logs/bris-fc-%j.err
#
# The evaluation forecasts: one run per day at 00 UTC, over a date range, from
# one checkpoint. This is what the three models are judged on.
#
#   sbatch --export=ALL,BRIS_TAG=baseline,BRIS_START=2025-09-08,BRIS_END=2026-09-06,\
#BRIS_LAM_ZARR=meps-2p5km-year1-6h-v1.zarr,BRIS_GLOBAL_ZARR=od-an-n320-year1-6h-v1.zarr \
#          bris/slurm/bris_forecast.sbatch
#
#   sbatch --export=ALL,BRIS_TAG=tail,BRIS_CHECKPOINT=$HOME/bris-models/tail/tail-step003000_inference.ckpt,...
#
# ONE CALL PER DAY, NOT ONE PER RANGE. bris opens the dataset at the config's
# frequency and forecasts from every state in it, so a range at 6 h spacing
# would produce four forecasts a day. The model is therefore reloaded per day,
# about a minute against five minutes of forecasting, which is the cheaper of
# the two mistakes.
#
# RESUMABLE. A day whose output file already exists is skipped, so a job that
# hits the walltime, or a node that dies, is continued by resubmitting it.
#
# WHICH INPUT FILE. The year datasets do not overlap: year 2 ends 7 September
# 2025 and year 1 starts the next day. The test period spans both, so it is run
# as two jobs per model rather than by merging the datasets.

set -uo pipefail

ENV_DIR="${BRIS_ENV_DIR:-$HOME/bris-env}"
REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
export BRIS_MODEL_DIR="${BRIS_MODEL_DIR:-$HOME/bris-models}"
export BRIS_DATA_DIR="${BRIS_DATA_DIR:-$HOME/bris-data}"

TAG="${BRIS_TAG:?set BRIS_TAG, e.g. baseline, control or tail}"
CKPT="${BRIS_CHECKPOINT:-$BRIS_MODEL_DIR/bris-forecaster/bris-crpsfft_inference.ckpt}"
START="${BRIS_START:?set BRIS_START, e.g. 2025-09-08}"
END="${BRIS_END:?set BRIS_END, e.g. 2026-09-06}"
HOUR="${BRIS_HOUR:-00}"
LEADTIMES="${BRIS_LEADTIMES:-11}"
MEMBERS="${BRIS_MEMBERS:-4}"
LAM="${BRIS_LAM_ZARR:?set BRIS_LAM_ZARR}"
GLOBAL="${BRIS_GLOBAL_ZARR:?set BRIS_GLOBAL_ZARR}"
CONFIG="${BRIS_CONFIG:-$REPO_DIR/bris/configs/config_inference_eval.yaml}"
OUTPUT_DIR="${BRIS_OUTPUT_DIR:-$HOME/bris-runs/evaluation/$TAG}"
export BRIS_OUTPUT_DIR="$OUTPUT_DIR"

export SLURM_NNODES="${SLURM_NNODES:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MLFLOW_TRACKING_URI="file://$OUTPUT_DIR/mlruns"
export HF_HUB_OFFLINE=1
export HYDRA_FULL_ERROR=1
# Read by anemoi at import time; see bris_smoke.sbatch.
export ANEMOI_INFERENCE_NUM_CHUNKS="${BRIS_NUM_CHUNKS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# shellcheck source=../../scripts/tls_env.sh
source "$REPO_DIR/scripts/tls_env.sh"

mkdir -p "$OUTPUT_DIR" logs

GPUS="${SLURM_GPUS_ON_NODE:-1}"
echo "=== node       : $(hostname), $GPUS gpu(s)"
echo "=== model      : $TAG"
echo "=== checkpoint : $CKPT"
echo "=== sha256     : $(sha256sum "$CKPT" | cut -c1-16)"
echo "=== dates      : $START .. $END at ${HOUR}Z, leadtimes $LEADTIMES, members $MEMBERS"
echo "=== inputs     : $LAM + $GLOBAL"
echo "=== output     : $OUTPUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint missing: $CKPT" >&2; exit 1; }
for ds in "$LAM" "$GLOBAL"; do
  [[ -e "$BRIS_DATA_DIR/$ds" ]] || { echo "ERROR: input missing: $BRIS_DATA_DIR/$ds" >&2; exit 2; }
done

RUN_CONFIG="$OUTPUT_DIR/config-${SLURM_JOB_ID:-manual}.yaml"
sed -e "s|^run_name: .*|run_name: bris_${TAG}|" \
    -e "s|^    checkpoint_path: .*|    checkpoint_path: ${CKPT}|" \
    -e "s|^    leadtimes: .*|    leadtimes: ${LEADTIMES}|" \
    -e "s|^  num_members: .*|  num_members: ${MEMBERS}|" \
    -e "s|^    dataset_lam: .*|    dataset_lam: ${LAM}|" \
    -e "s|^    dataset_global: .*|    dataset_global: ${GLOBAL}|" \
    "$CONFIG" > "$RUN_CONFIG"
grep -n "^run_name:\|checkpoint_path:\|^    leadtimes:\|^  num_members:\|^    dataset_lam:\|^    dataset_global:" "$RUN_CONFIG"
echo

cd "$ENV_DIR"
done_count=0 skipped=0 failed=0
started=$SECONDS
day="$START"
while [[ "$day" < "$END" || "$day" == "$END" ]]; do
  stamp="$(date -d "$day" +%Y-%m-%d)T${HOUR}Z"
  out="$OUTPUT_DIR/nordic_bris_${TAG}_${stamp}.nc"
  if [[ -f "$out" ]]; then
    skipped=$((skipped + 1))
  else
    t0=$SECONDS
    srun env SLURM_GPUS_PER_NODE="$GPUS" "$ENV_DIR/.venv/bin/bris" \
      --config "$RUN_CONFIG" \
      -sd "${day}T${HOUR}:00:00" \
      -ed "${day}T${HOUR}:00:00" \
      -l "$LEADTIMES" \
      -wd "$OUTPUT_DIR" > "$OUTPUT_DIR/log-${day}.txt" 2>&1
    rc=$?
    if [[ $rc -eq 0 && -f "$out" ]]; then
      done_count=$((done_count + 1))
      rm -f "$OUTPUT_DIR/log-${day}.txt"
      echo "  $day  $((SECONDS - t0))s  ($done_count done, $skipped skipped, $failed failed)"
    else
      failed=$((failed + 1))
      echo "  $day  FAILED rc=$rc, log kept at $OUTPUT_DIR/log-${day}.txt" >&2
    fi
  fi
  day="$(date -d "$day + 1 day" +%Y-%m-%d)"
done

echo
echo "=== $done_count forecast(s) in $((SECONDS - started))s, $skipped already present, $failed failed"
du -sh "$OUTPUT_DIR"
[[ $failed -eq 0 ]]
