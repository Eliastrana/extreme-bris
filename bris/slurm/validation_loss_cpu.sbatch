#!/usr/bin/env bash
#SBATCH --job-name=val-loss
#SBATCH --partition=fpgaq,defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=16:00:00
#SBATCH --output=logs/val-loss-%j.out
#SBATCH --error=logs/val-loss-%j.err
#
# One shard of the validation loss of one checkpoint, on CPU.
#
#   sbatch bris/slurm/validation_loss_cpu.sbatch baseline 1/2
#   sbatch bris/slurm/validation_loss_cpu.sbatch control 1/2 \
#       ~/bris-runs/train/checkpoint/<run>/anemoi-by_step-epoch_001-step_003000.ckpt
#
# Without a checkpoint it scores the config's warm start, untouched Bris.
# Compare when every shard is in:
#   scripts/validation_loss.py --compare ~/bris-runs/validation-loss/*.json
#
# 96 cores and two partitions: fpgaq's nodes have 96, and defq's n001 usually
# has a little under 128 free, so asking for 96 lets either take a shard.

set -uo pipefail

LABEL="${1:?usage: sbatch $0 label k/n [checkpoint]}"
SHARD="${2:?usage: sbatch $0 label k/n [checkpoint]}"
CKPT="${3:-}"

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
VENV="${BRIS_ENV_DIR:-$HOME/bris-env}/.venv"
CORES="${SLURM_CPUS_PER_TASK:-$(nproc)}"

export OMP_NUM_THREADS="$CORES" MKL_NUM_THREADS="$CORES" SLURM_CPUS_PER_TASK="$CORES"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export ANEMOI_BASE_SEED="${ANEMOI_BASE_SEED:-20260909}"
export ANEMOI_INFERENCE_NUM_CHUNKS=16 ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=32
export XBRIS_SEQUENTIAL_MEMBERS=1

echo "=== node  : $(hostname), $CORES cores, $(free -g | awk 'NR==2{print $7}') GB free"
echo "=== label : $LABEL, shard $SHARD, weights ${CKPT:-warm start from the config}"

"$VENV/bin/python" "$REPO_DIR/scripts/validation_loss.py" --cpu \
  --label "$LABEL" --shard "$SHARD" ${CKPT:+--checkpoint "$CKPT"}
rc=$?
echo "=== finished rc=$rc"
exit $rc
