#!/usr/bin/env bash
#SBATCH --job-name=bris-inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hgx2q
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:8
#SBATCH --time=04:00:00
#SBATCH --output=logs/bris-%j.out
#SBATCH --error=logs/bris-%j.err
#
# hgx2q is the single 8x A100 80GB node (Supermicro 4124GO-NART), 256 CPUs,
# 14-day limit. The 2x A100 nodes on a100q/milanq CANNOT run this config as
# published: num_gpus_per_model is 4, so two cards are not enough. Use them for
# the environment build and checkpoint inspection only.
#
# hgx2q is a single shared node — check what is free before asking for all 8:
#   sinfo -p hgx2q -o "%n %G %C %t"
#   squeue -p hgx2q
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
#   sbatch --gres=gpu:a100:4 --export=ALL,BRIS_MEMBERS=1 ...
# and watch for OOM in the error log.
#
# ---------------------------------------------------------------------------
# RESUME
# ---------------------------------------------------------------------------
# hgx2q allows 14 days, so walltime is not the pressure it was assumed to be —
# a 60h forecast will not come close. Resume is kept anyway because the node is
# shared and a job can still die for other reasons: one date per invocation,
# skipping any date already marked complete.

set -euo pipefail

ENV_DIR="${BRIS_ENV_DIR:-$HOME/bris-env}"
# Resolved by oc.env inside the config, so these must be exported.
export BRIS_MODEL_DIR="${BRIS_MODEL_DIR:-$HOME/bris-models}"
export BRIS_DATA_DIR="${BRIS_DATA_DIR:-$HOME/bris-data}"
MODEL_DIR="$BRIS_MODEL_DIR"
REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
OUTPUT_DIR="${BRIS_OUTPUT_DIR:-$HOME/bris-runs/outputs}"
CONFIG="${BRIS_CONFIG:-$REPO_DIR/bris/configs/config_inference_ex3.yaml}"

# Dates to forecast, whitespace-separated. One is enough for the first milestone.
BRIS_DATES="${BRIS_DATES:-2025-04-01T00:00:00}"

export BRIS_OUTPUT_DIR="$OUTPUT_DIR"

# shellcheck source=../../scripts/tls_env.sh
source "$REPO_DIR/scripts/tls_env.sh"

mkdir -p "$OUTPUT_DIR" logs

# --- environment ------------------------------------------------------------

# bris-inference reads its parallelism from Slurm's environment. Under sbatch
# these are set for us; export them explicitly so the config's oc.env lookups
# resolve identically in an interactive srun too.
# Prefer what Slurm actually granted; --gres does not always set
# SLURM_GPUS_PER_NODE, so fall back to counting visible devices rather than
# assuming 8 and mismatching the allocation.
if [[ -z "${SLURM_GPUS_PER_NODE:-}" ]]; then
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    export SLURM_GPUS_PER_NODE="$SLURM_GPUS_ON_NODE"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    export SLURM_GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    export SLURM_GPUS_PER_NODE=8
  fi
fi
export SLURM_NNODES="${SLURM_NNODES:-1}"

# Keep everything offline. The training config points MLflow at mlflow.ecmwf.int;
# inference should never phone home, and a compute node may have no egress anyway.
export MLFLOW_TRACKING_URI="file://$OUTPUT_DIR/mlruns"
export HF_HUB_OFFLINE=1
export ANEMOI_CONFIG_PATH="${ANEMOI_CONFIG_PATH:-$OUTPUT_DIR/.anemoi}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export HYDRA_FULL_ERROR=1
# Chunking has to be exported BEFORE python starts. anemoi's block.py reads
#   NUM_CHUNKS_INFERENCE = int(os.environ.get("ANEMOI_INFERENCE_NUM_CHUNKS", "1"))
# at module level, i.e. at import time. bris sets the same variable inside
# main(), long after anemoi has been imported, so the constant is already 1 and
# the config's inference_num_chunks has no effect — the log still says
# "chunked to 8" because bris prints its own value, not the model's.
export ANEMOI_INFERENCE_NUM_CHUNKS="${BRIS_NUM_CHUNKS:-4}"
# Reduces fragmentation, which the OOM message explicitly recommends.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
    --config "$CONFIG" \
    -sd "$date" \
    -ed "$date" \
    -wd "$OUTPUT_DIR"

  echo ">> $date finished in $((SECONDS - start))s"
  touch "$marker"
done

echo
echo "=== outputs"
ls -lh "$OUTPUT_DIR"/*.nc 2>/dev/null || echo "  (no NetCDF written — check the error log)"
