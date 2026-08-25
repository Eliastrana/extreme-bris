# shellcheck shell=bash
#
# Shared paths for interactive use. Source this, do not execute it.
#
#   source ~/extreme-bris/scripts/env.sh
#
# run.sh sets these internally, but they are not exported to your shell, so
# documented commands like `cd $BRIS_ENV_DIR` silently become `cd` and run
# against the wrong environment. Sourcing this makes them work as written.
#
# Override any of them by exporting before sourcing.

export BRIS_REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
export BRIS_MODEL_DIR="${BRIS_MODEL_DIR:-$HOME/bris-models}"
export BRIS_DATA_DIR="${BRIS_DATA_DIR:-$HOME/bris-data}"
export BRIS_ENV_DIR="${BRIS_ENV_DIR:-$HOME/bris-env}"
export BRIS_RUN_DIR="${BRIS_RUN_DIR:-$HOME/bris-runs}"
export BRIS_CKPT="${BRIS_CKPT:-$BRIS_MODEL_DIR/bris-forecaster/bris-crpsfft_inference.ckpt}"

export PATH="$HOME/.local/bin:$PATH"

# eX3 asks for at most 8 threads on the login node.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# shellcheck source=tls_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tls_env.sh"
