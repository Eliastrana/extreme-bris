#!/usr/bin/env bash
#
# Build the Bris inference environment on an eX3 GPU node.
#
#   ./scripts/setup_env.sh                 # build in $HOME/bris-env
#   ./scripts/setup_env.sh -d /path        # custom location
#   ./scripts/setup_env.sh -n              # no --locked (allows resolver drift)
#
# Run this on a compute node, not the login node: the CUDA wheels are large and
# the resolve is CPU-heavy. Submit it as a batch job rather than sitting in an
# interactive session — eX3 asks for sbatch over srun, and an interactive shell
# that is forgotten or disconnected can hold a GPU:
#
#   mkdir -p logs && sbatch bris/slurm/setup_and_inspect.sbatch
#
# No GPU is needed here. Installing wheels is CPU work, so the batch job runs on
# defq without --gres, and torch.cuda.is_available() being False in the
# verification output below is expected rather than a problem.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# met-no/bris-forecaster pins the inference package as an SSH git dependency:
#
#   bris = { git = "ssh://git@github.com/metno/bris-inference.git", rev = "d1d27c1..." }
#
# `uv sync --locked` therefore tries to clone over SSH, which fails on any node
# without a GitHub SSH key. The repo is public, so we rewrite the transport to
# HTTPS with git's insteadOf. That leaves the URL *string* in uv.lock untouched,
# so the lockfile stays valid and --locked still works.

set -euo pipefail

ENV_DIR="${BRIS_ENV_DIR:-$HOME/bris-env}"
MODEL_DIR="${BRIS_MODEL_DIR:-$HOME/bris-models}"
LOCKED="--locked"
PY_VERSION="3.12.11"   # pyproject pins this exactly: requires-python = "==3.12.11"

while getopts ":d:nh" opt; do
  case "$opt" in
    d) ENV_DIR="$OPTARG" ;;
    n) LOCKED="" ;;
    h) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

# eX3 re-signs TLS; uv needs the system trust store or it fails UnknownIssuer.
# shellcheck source=tls_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tls_env.sh"

command -v uv >/dev/null 2>&1 || {
  echo "ERROR: uv not found. Install it first:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}

SRC="$MODEL_DIR/bris-forecaster"
if [[ ! -f "$SRC/pyproject.toml" || ! -f "$SRC/uv.lock" ]]; then
  echo "ERROR: need pyproject.toml and uv.lock in $SRC" >&2
  echo "       Run ./scripts/download_models.sh -c first." >&2
  exit 1
fi

# --- transport rewrite: ssh -> https for the public metno repo ---------------
# Scoped to this host's git config. Harmless and idempotent; remove with
#   git config --global --unset url."https://github.com/".insteadOf

if ! git config --global --get-all url."https://github.com/".insteadOf 2>/dev/null \
     | grep -qx "ssh://git@github.com/"; then
  echo ">> adding git insteadOf rewrite: ssh://git@github.com/ -> https://github.com/"
  git config --global --add url."https://github.com/".insteadOf "ssh://git@github.com/"
else
  echo ">> git insteadOf rewrite already present"
fi

mkdir -p "$ENV_DIR"
cp "$SRC/pyproject.toml" "$SRC/uv.lock" "$ENV_DIR/"

cd "$ENV_DIR"

# Keep uv's cache off the home quota — it holds several GB of CUDA wheels.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ENV_DIR/.uv-cache}"

echo ">> installing Python $PY_VERSION"
uv python install "$PY_VERSION"

echo ">> uv sync $LOCKED  (this pulls torch 2.6.0+cu124 and the CUDA runtime; several GB)"
# shellcheck disable=SC2086
uv sync $LOCKED

echo
echo ">> verifying"
uv run python - <<'PY'
import torch, sys
print(f"  python : {sys.version.split()[0]}")
print(f"  torch  : {torch.__version__}")
print(f"  cuda   : {torch.version.cuda}  available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu {i}  : {p.name}  {p.total_memory/1e9:.1f} GB")
else:
    print("  NOTE: no GPU visible — expected on a login node, a problem on a GPU node.")
import bris
print(f"  bris   : {getattr(bris, '__version__', 'installed')}")
PY

cat <<NEXT

Environment ready at $ENV_DIR

Run things with:
  cd $ENV_DIR && uv run <command>

Next:
  python scripts/inspect_checkpoint.py $MODEL_DIR/bris-forecaster/bris-crpsfft_inference.ckpt
NEXT
