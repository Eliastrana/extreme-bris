#!/usr/bin/env bash
#SBATCH --job-name=tail-weight
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/tail-weight-%j.out
#SBATCH --error=logs/tail-weight-%j.err
#
# Measure the tail term's weight on the extreme states. No training, no weight
# updates, one card, about twenty minutes. See scripts/measure_tail_weight.py.
#
#   scripts/measure_tail_weight.py --plan-only     # first, on the login node
#   mkdir -p logs && sbatch bris/slurm/measure_tail_weight.sbatch
#
# The environment matches bris/slurm/finetune.sbatch, so the model runs exactly
# as it does in training: same chunking outside training, members one at a
# time, same allocator setting and seed.

set -uo pipefail

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
[[ -f "$REPO_DIR/scripts/env.sh" ]] && source "$REPO_DIR/scripts/env.sh"
VENV="${BRIS_ENV_DIR:-$HOME/bris-env}/.venv"

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ANEMOI_BASE_SEED="${ANEMOI_BASE_SEED:-20260909}"
export ANEMOI_INFERENCE_NUM_CHUNKS="${ANEMOI_INFERENCE_NUM_CHUNKS:-16}"
export ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER="${ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER:-32}"
export XBRIS_SEQUENTIAL_MEMBERS=1

OUT="$HOME/bris-runs/tail-weight/measure-${SLURM_JOB_ID:-local}.json"
mkdir -p "$(dirname "$OUT")"
echo "=== node : $(hostname)"
echo "=== card : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

"$VENV/bin/python" "$REPO_DIR/scripts/measure_tail_weight.py" --out "$OUT" "$@"
rc=$?
echo "=== finished rc=$rc"
exit $rc
