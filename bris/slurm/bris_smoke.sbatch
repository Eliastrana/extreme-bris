#!/usr/bin/env bash
#SBATCH --job-name=bris-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100q
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err
#
# Smoke test: does Bris load and step at all? Smallest allocation that could
# possibly work, on an idle non-flagship node. Not for real forecasts.
#
#   mkdir -p logs && sbatch bris/slurm/bris_smoke.sbatch
#
# ---------------------------------------------------------------------------
# WHAT THIS CHANGES vs THE PUBLISHED CONFIG
# ---------------------------------------------------------------------------
#   num_gpus_per_model : 4 -> 1     no model sharding
#   num_members        : 2 -> 1     single member, no ensemble
#   leadtimes          : 10 -> 2    12h instead of 60h
#
# MET published 4 GPUs per model because that is how they ran it on Leonardo's
# 64 GB A100s, not necessarily because 1024 channels cannot fit on one card at
# inference. Inference has no optimizer state, no activations retained for
# backward, and no ensemble here — so it is plausible this fits on a single
# 48 GB A40. Plausible, not established. That is the point of the job.
#
# a40q is NOT usable: those nodes are aarch64, and this environment is x86/CUDA
# from MET's lockfile. A job there dies with
#   x86_64-binfmt-P: Could not open '/lib64/ld-linux-x86-64.so.2'
# which looks like a missing library and is really a wrong architecture.
#
# If it OOMs, walk up the ladder (see README):
#   a100q 2x A100 40GB  ->  dgx2q 4x V100  ->  hgx2q 8x A100 80GB
#
# Escalate by overriding on submit, e.g. four V100s on the DGX-2:
#   sbatch --partition=dgx2q --gres=gpu:tesla:4 \
#          --export=ALL,BRIS_GPUS_PER_MODEL=4 bris/slurm/bris_smoke.sbatch

set -euo pipefail

ENV_DIR="${BRIS_ENV_DIR:-$HOME/bris-env}"
REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
OUTPUT_DIR="${BRIS_OUTPUT_DIR:-$HOME/bris-runs/smoke}"
CONFIG="${BRIS_CONFIG:-$REPO_DIR/bris/configs/config_inference_smoke.yaml}"

# The config resolves these with oc.env, so they must be exported, not just set.
export BRIS_MODEL_DIR="${BRIS_MODEL_DIR:-$HOME/bris-models}"
export BRIS_DATA_DIR="${BRIS_DATA_DIR:-$HOME/bris-data}"
export BRIS_OUTPUT_DIR="$OUTPUT_DIR"

GPUS_PER_MODEL="${BRIS_GPUS_PER_MODEL:-1}"
MEMBERS="${BRIS_MEMBERS:-1}"
LEADTIMES="${BRIS_LEADTIMES:-2}"
DATE="${BRIS_DATE:-2025-04-01T00:00:00}"

# shellcheck source=../../scripts/tls_env.sh
source "$REPO_DIR/scripts/tls_env.sh"

mkdir -p "$OUTPUT_DIR" logs

if [[ -z "${SLURM_GPUS_PER_NODE:-}" ]]; then
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    export SLURM_GPUS_PER_NODE="$SLURM_GPUS_ON_NODE"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    export SLURM_GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    export SLURM_GPUS_PER_NODE=1
  fi
fi
export SLURM_NNODES="${SLURM_NNODES:-1}"

export MLFLOW_TRACKING_URI="file://$OUTPUT_DIR/mlruns"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export HYDRA_FULL_ERROR=1
# Makes OOM report the real allocation site rather than a cascade of retries.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== node    : $(hostname)"
echo "=== gpus    : $SLURM_GPUS_PER_NODE  (per-model: $GPUS_PER_MODEL, members: $MEMBERS)"
echo "=== leadtimes: $LEADTIMES  (${DATE})"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo

[[ -d "$ENV_DIR" ]] || { echo "ERROR: env not found: $ENV_DIR (sbatch setup_and_inspect.sbatch)" >&2; exit 1; }
[[ -f "$CONFIG" ]]  || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }

# --- preflight --------------------------------------------------------------
# Fail here, loudly, rather than 40 lines into an anemoi-datasets traceback.

CKPT="$BRIS_MODEL_DIR/bris-forecaster/bris-crpsfft_inference.ckpt"
[[ -f "$CKPT" ]] || {
  echo "ERROR: checkpoint missing: $CKPT" >&2
  echo "       ./scripts/download_models.sh   (4.7 GB, run on the login node)" >&2
  exit 1
}

missing=0
# Read the dataset names out of the config rather than repeating them here —
# they were duplicated, drifted apart, and the job then looked for a file the
# config had stopped naming.
mapfile -t datasets < <(grep -E "^\s+dataset_(lam|global):" "$CONFIG" | awk '{print $2}')
for ds in "${datasets[@]}"; do
  if [[ ! -e "$BRIS_DATA_DIR/$ds" ]]; then
    echo "MISSING INPUT: $BRIS_DATA_DIR/$ds" >&2
    missing=1
  fi
done
if [[ "$missing" -eq 1 ]]; then
  cat >&2 <<'NODATA'

No input data — the forecast cannot run yet.

This is the open work, not a misconfiguration. Bris needs initial conditions on
a cutout of MEPS 2.5 km inside and N320 outside. The N320 atmospheric component
MET used is MARS class od, which we do not have access to; the plan is to
substitute public ERA5 N320. See docs/INPUTS.md.

Everything upstream of this point (environment, checkpoint, Slurm wiring) can
still be verified with:

    sbatch bris/slurm/setup_and_inspect.sbatch

NODATA
  exit 2
fi

cd "$ENV_DIR"

start=$SECONDS
# bris takes a plain --config file and argparse flags — not Hydra overrides.
# There is no flag for num_gpus_per_model or num_members, which is why the
# smoke values live in their own config file.
uv run bris \
  --config "$CONFIG" \
  -sd "$DATE" \
  -ed "$DATE" \
  -l "$LEADTIMES" \
  -wd "$OUTPUT_DIR"

echo
echo "=== finished in $((SECONDS - start))s"
ls -lh "$OUTPUT_DIR"/*.nc 2>/dev/null || echo "  (no NetCDF written — check the error log)"
