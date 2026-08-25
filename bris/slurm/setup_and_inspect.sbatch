#!/usr/bin/env bash
#SBATCH --job-name=bris-setup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err
#
# Build the inference environment and dump the checkpoint metadata.
# Tasks 1 and 3 in one batch job.
#
#   mkdir -p logs && sbatch bris/slurm/setup_and_inspect.sbatch
#   tail -f logs/setup-<jobid>.out
#
# NO GPU IS REQUESTED, DELIBERATELY.
#
# Installing wheels is CPU and I/O work, and reading checkpoint metadata happens
# on the CPU (torch.load with map_location="cpu"). Asking for a GPU here would
# hold a card for the whole install for no reason. eX3's guidance is to submit
# batch jobs rather than sit in interactive sessions precisely because a
# forgotten interactive shell can lock a GPU — this sidesteps that entirely.
#
# The GPU is first exercised by bris_smoke.sbatch, which needs one.
#
# Consequence: torch.cuda.is_available() is False in the verification output
# below. That is expected here and is not a failure.

set -euo pipefail

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
ENV_DIR="${BRIS_ENV_DIR:-$HOME/bris-env}"
MODEL_DIR="${BRIS_MODEL_DIR:-$HOME/bris-models}"
CKPT="$MODEL_DIR/bris-forecaster/bris-crpsfft_inference.ckpt"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ENV_DIR/.uv-cache}"

echo "=== node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK:-?}"
echo

command -v uv >/dev/null 2>&1 || {
  echo "ERROR: uv not on PATH. On the login node, once:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}

# --- task 1: environment ----------------------------------------------------
echo ">>> building environment"
"$REPO_DIR/scripts/setup_env.sh" -d "$ENV_DIR"

# --- task 3: checkpoint metadata -------------------------------------------
echo
if [[ -f "$CKPT" ]]; then
  echo ">>> inspecting $CKPT"
  cd "$ENV_DIR"
  uv run python "$REPO_DIR/scripts/inspect_checkpoint.py" "$CKPT" \
      --json "$ENV_DIR/ckpt-metadata.json"
  echo
  echo ">>> metadata written to $ENV_DIR/ckpt-metadata.json"
  echo ">>> copy the variable order and multistep_input into docs/INPUTS.md"
else
  echo "SKIPPED checkpoint inspection: $CKPT not found."
  echo "Run ./scripts/download_models.sh on the login node first (4.7 GB)."
fi
