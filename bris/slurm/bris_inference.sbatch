#!/usr/bin/env bash
#SBATCH --job-name=bris-inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/bris-%j.out
#SBATCH --error=logs/bris-%j.err
# TODO: set the partition once `sinfo` confirms the name of the 8x A100 80GB
# (Supermicro 4124GO-NART) queue on eX3. Add e.g.  #SBATCH --partition=a100q
#
# Submit:
#   mkdir -p logs && sbatch bris/slurm/bris_inference.sbatch
#
# Override the date without editing this file:
#   sbatch --export=ALL,BRIS_DATES="2025-04-01T00:00:00" bris/slurm/bris_inference.sbatch
#
# ---------------------------------------------------------------------------
# WHY 8 GPUs
# ---------------------------------------------------------------------------
# met-no's config_inference.yaml sets num_gpus_per_model: 4 and num_members: 2.
# The model is sharded across 4 GPUs and two ensemble members run alongside, so
# the published configuration wants all 8 cards on the node — it is NOT a
# single-GPU job, despite inference being much lighter than training.
#
# Whether it can be squeezed onto fewer cards on 80 GB A100s (MET trained on
# 64 GB) is an open question. Test it deliberately rather than assuming:
#   sbatch --gpus-per-node=4 --export=ALL,BRIS_GPUS_PER_MODEL=4,BRIS_MEMBERS=1 ...
# and watch for OOM in the error log.
#
# ---------------------------------------------------------------------------
# RESUME
# ---------------------------------------------------------------------------
# Node walltime limits on eX3 are not yet known. This script runs one date per
# invocation and skips any date whose output already exists, so re-submitting
# after a timeout picks up where it left off rather than redoing finished work.

set -euo pipefail

ENV_DIR="${BRIS_ENV_DIR:-$HOME/bris-env}"
MODEL_DIR="${BRIS_MODEL_DIR:-$HOME/bris-models}"
REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
OUTPUT_DIR="${BRIS_OUTPUT_DIR:-$HOME/bris-runs/outputs}"
CONFIG="${BRIS_CONFIG:-$REPO_DIR/bris/configs/config_inference_ex3.yaml}"

# Dates to forecast, whitespace-separated. One is enough for the first milestone.
BRIS_DATES="${BRIS_DATES:-2025-04-01T00:00:00}"

mkdir -p "$OUTPUT_DIR" logs

# --- environment ------------------------------------------------------------

# bris-inference reads its parallelism from Slurm's environment. Under sbatch
# these are set for us; export them explicitly so the config's oc.env lookups
# resolve identically in an interactive srun too.
export SLURM_GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-${BRIS_GPUS_PER_NODE:-8}}"
export SLURM_NNODES="${SLURM_NNODES:-1}"

# Keep everything offline. The training config points MLflow at mlflow.ecmwf.int;
# inference should never phone home, and a compute node may have no egress anyway.
export MLFLOW_TRACKING_URI="file://$OUTPUT_DIR/mlruns"
export HF_HUB_OFFLINE=1
export ANEMOI_CONFIG_PATH="${ANEMOI_CONFIG_PATH:-$OUTPUT_DIR/.anemoi}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export HYDRA_FULL_ERROR=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "=== node   : $(hostname)"
echo "=== gpus   : $SLURM_GPUS_PER_NODE on $SLURM_NNODES node(s)"
echo "=== config : $CONFIG"
echo "=== output : $OUTPUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo

[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
[[ -d "$ENV_DIR" ]] || { echo "ERROR: env not found: $ENV_DIR (run scripts/setup_env.sh)" >&2; exit 1; }

# --- run --------------------------------------------------------------------

cd "$ENV_DIR"

for date in $BRIS_DATES; do
  stamp="$(echo "$date" | tr -d ':-' | cut -c1-11)"   # 20250401T00
  marker="$OUTPUT_DIR/.done_${stamp}"

  if [[ -f "$marker" ]]; then
    echo ">> $date already complete (marker $marker) — skipping"
    continue
  fi

  echo ">> forecasting $date"
  start=$SECONDS

  uv run bris \
    --config-path="$(dirname "$CONFIG")" \
    --config-name="$(basename "${CONFIG%.yaml}")" \
    start_date="$date" \
    end_date="$date" \
    output_path="$OUTPUT_DIR"

  echo ">> $date finished in $((SECONDS - start))s"
  touch "$marker"
done

echo
echo "=== outputs"
ls -lh "$OUTPUT_DIR"/*.nc 2>/dev/null || echo "  (no NetCDF written — check the error log)"
