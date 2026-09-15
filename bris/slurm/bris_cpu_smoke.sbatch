#!/usr/bin/env bash
#SBATCH --job-name=bris-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=fpgaq
#SBATCH --cpus-per-task=96
#SBATCH --time=12:00:00
#SBATCH --output=logs/bris-cpu-%j.out
#SBATCH --error=logs/bris-cpu-%j.err
#
# One forecast on CPU, from any inference checkpoint. For checking that a
# checkpoint loads and produces physical fields without waiting days for a GPU:
#
#   sbatch --export=ALL,BRIS_CHECKPOINT=$HOME/bris-models/control/control-step002793_inference.ckpt,BRIS_TAG=control \
#          bris/slurm/bris_cpu_smoke.sbatch
#   sbatch --export=ALL,BRIS_TAG=baseline bris/slurm/bris_cpu_smoke.sbatch
#
# RUN THE BASELINE BESIDE IT. Same date, same inputs, same config, same code
# path, so the only difference between the two outputs is the weights. The
# baseline forecasts from 4 September are not that reference: they predate the
# dewpoint repair of the inputs on 9 September.
#
# WHY CPU WORKS. bris picks the device itself. With no GPU visible it logs
# "Could not find gpu, using CPU", and Lightning's DDP falls back to gloo. The
# model asks for cuda autocast, which does nothing on CPU, while the trainer's
# bf16 setting becomes CPU autocast. Close to the GPU numbers, not identical:
# good for "does it load and is the rain physical", not for scoring.
#
# DO NOT POINT THIS AT A CHECKPOINT TRAINING IS STILL WRITING. inference-last.ckpt
# is rewritten at every save; copy it somewhere fixed first, as above.
#
# WHERE. fpgaq: four x86 AMD EPYC nodes, 96 cores and 500 GB each, usually
# idle. milanq's free cores sat behind a down node and GPU-partition overlap,
# and a40q, armq and aarchq are ARM, which this x86 environment cannot run on.
#
# LEADTIMES counts the analysis step. 7 is +0 .. +36 h, the shortest run that
# holds one whole 06-06 UTC day (+6 .. +30 h) for the daily gauges.

set -euo pipefail

ENV_DIR="${BRIS_ENV_DIR:-$HOME/bris-env}"
REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
export BRIS_MODEL_DIR="${BRIS_MODEL_DIR:-$HOME/bris-models}"
export BRIS_DATA_DIR="${BRIS_DATA_DIR:-$HOME/bris-data}"

CKPT="${BRIS_CHECKPOINT:-$BRIS_MODEL_DIR/bris-forecaster/bris-crpsfft_inference.ckpt}"
TAG="${BRIS_TAG:-$(basename "$CKPT" .ckpt)}"
DATE="${BRIS_DATE:-2025-10-04T00:00:00}"
STAMP="$(echo "$DATE" | tr -d ':-' | cut -c1-11)"          # 20251004T00
LEADTIMES="${BRIS_LEADTIMES:-7}"
LAM="${BRIS_LAM_ZARR:-meps-2p5km-${STAMP}Z-6h-v1.zarr}"
GLOBAL="${BRIS_GLOBAL_ZARR:-od-an-n320-${STAMP}Z-6h-v1.zarr}"
CONFIG="${BRIS_CONFIG:-$REPO_DIR/bris/configs/config_inference_smoke.yaml}"
OUTPUT_DIR="${BRIS_OUTPUT_DIR:-$HOME/bris-runs/cpu-smoke/${STAMP}Z-${TAG}}"
export BRIS_OUTPUT_DIR="$OUTPUT_DIR"

# No GPU, and say so to everything that asks. devices=1 is one CPU process.
# SLURM_GPUS_PER_NODE, which the config reads as the device count, is set on the
# bris process only, below. Exported here, srun reads it as a GPU request of its
# own and refuses the step: "Step requested GRES but job doesn't have GRES".
export CUDA_VISIBLE_DEVICES=""
unset SLURM_GPUS_PER_NODE
export BRIS_GPUS_PER_MODEL=1
export SLURM_NNODES="${SLURM_NNODES:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"

export MLFLOW_TRACKING_URI="file://$OUTPUT_DIR/mlruns"
export HF_HUB_OFFLINE=1
export HYDRA_FULL_ERROR=1
# Read by anemoi at import time, so exported before python starts; see
# bris_smoke.sbatch. Memory is not the constraint on these nodes.
export ANEMOI_INFERENCE_NUM_CHUNKS="${BRIS_NUM_CHUNKS:-4}"

# shellcheck source=../../scripts/tls_env.sh
source "$REPO_DIR/scripts/tls_env.sh"

mkdir -p "$OUTPUT_DIR" logs

echo "=== node       : $(hostname), ${OMP_NUM_THREADS} threads"
echo "=== checkpoint : $CKPT"
echo "=== sha256     : $(sha256sum "$CKPT" | cut -c1-16)"
echo "=== date       : $DATE, leadtimes $LEADTIMES"
echo "=== inputs     : $LAM + $GLOBAL"
echo "=== output     : $OUTPUT_DIR"

[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint missing: $CKPT" >&2; exit 1; }
for ds in "$LAM" "$GLOBAL"; do
  [[ -e "$BRIS_DATA_DIR/$ds" ]] || { echo "ERROR: input missing: $BRIS_DATA_DIR/$ds" >&2; exit 2; }
done

# bris validates the file against its schema before resolving interpolations,
# so the per-run values are written into a copy, kept beside the output.
RUN_CONFIG="$OUTPUT_DIR/config-${SLURM_JOB_ID:-manual}.yaml"
sed -e "s|^run_name: .*|run_name: bris_${TAG}|" \
    -e "s|^    checkpoint_path: .*|    checkpoint_path: ${CKPT}|" \
    -e "s|^    leadtimes: .*|    leadtimes: ${LEADTIMES}|" \
    -e "s|^    dataset_lam: .*|    dataset_lam: ${LAM}|" \
    -e "s|^    dataset_global: .*|    dataset_global: ${GLOBAL}|" \
    "$CONFIG" > "$RUN_CONFIG"
grep -n "^run_name:\|checkpoint_path:\|^    leadtimes:\|^    dataset_lam:\|^    dataset_global:" "$RUN_CONFIG"
echo

cd "$ENV_DIR"
start=$SECONDS
# The environment's own entry point, not `uv run`: uv lives in ~/.local/bin,
# which is on PATH in a login shell and not in a batch job, where this failed
# with "env: 'uv': No such file or directory". It adds nothing here anyway; the
# environment is already built.
srun env SLURM_GPUS_PER_NODE=1 "$ENV_DIR/.venv/bin/bris" \
  --config "$RUN_CONFIG" \
  -sd "$DATE" \
  -ed "$DATE" \
  -l "$LEADTIMES" \
  -wd "$OUTPUT_DIR"
echo
echo "=== finished in $((SECONDS - start))s"

# The question this job answers, printed rather than left for later: is every
# field finite and in a physical range, and does the rain look like rain.
"$ENV_DIR/.venv/bin/python" - "$OUTPUT_DIR" <<'PY'
import glob, sys
import numpy as np, xarray as xr
for path in sorted(glob.glob(f"{sys.argv[1]}/nordic_*.nc")):
    print(f"--- {path.rsplit('/', 1)[-1]}")
    with xr.open_dataset(path) as ds:
        for name, da in ds.data_vars.items():
            if "time" not in da.dims or da.ndim < 3:
                continue
            for t in range(da.sizes["time"]):
                a = np.asarray(da.isel(time=t).values, dtype="float64")
                bad = int((~np.isfinite(a)).sum())
                print(f"  {name:32s} t{t:02d} mean {np.nanmean(a):10.4f} p99 {np.nanpercentile(a, 99):10.4f}"
                      f" max {np.nanmax(a):10.4f}  non-finite {bad}")
PY
