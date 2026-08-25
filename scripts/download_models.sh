#!/usr/bin/env bash
#
# Download the Bris model checkpoints and training configs from Hugging Face.
#
# Both repos are Apache 2.0 and public — no HF token required.
#
#   ./scripts/download_models.sh              # everything (several GB)
#   ./scripts/download_models.sh -c           # configs only, fast first look
#   ./scripts/download_models.sh -d /path     # custom target directory
#   ./scripts/download_models.sh -i           # pip install huggingface_hub if missing
#
# Target directory can also be set with the BRIS_MODEL_DIR environment variable.
# Defaults to $HOME/bris-models.
#
# Note: checkpoints are a few GB, so BeeGFS is fine for these. The large zarr
# training datasets are what belong on the node-local NVMe.

set -euo pipefail

REPOS=(
  "met-no/bris-forecaster"              # CRPS-FFT: inference + training ckpt, configs
  "met-no/bris-forecaster-pretrained"   # global pretraining ckpt + stage configs
)

TARGET_DIR="${BRIS_MODEL_DIR:-$HOME/bris-models}"
CONFIGS_ONLY=0
INSTALL_DEPS=0

usage() {
  awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
  exit 0
}

while getopts ":d:chi" opt; do
  case "$opt" in
    d) TARGET_DIR="$OPTARG" ;;
    c) CONFIGS_ONLY=1 ;;
    i) INSTALL_DEPS=1 ;;
    h) usage ;;
    \?) echo "Unknown option: -$OPTARG" >&2; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

# --- locate the Hugging Face CLI -------------------------------------------
# huggingface_hub >= 0.34 ships `hf`; older versions ship `huggingface-cli`.

find_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    echo "hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli"
  else
    echo ""
  fi
}

HF_CLI="$(find_hf_cli)"

if [[ -z "$HF_CLI" && "$INSTALL_DEPS" -eq 1 ]]; then
  echo ">> huggingface_hub not found, installing to user site-packages"
  python3 -m pip install --user --upgrade huggingface_hub
  export PATH="$HOME/.local/bin:$PATH"
  HF_CLI="$(find_hf_cli)"
fi

if [[ -z "$HF_CLI" ]]; then
  cat >&2 <<'MISSING_CLI'
ERROR: neither `hf` nor `huggingface-cli` found on PATH.

Install it into your environment first:

    python3 -m pip install --user --upgrade huggingface_hub

or re-run this script with -i to let it do that for you.
MISSING_CLI
  exit 1
fi

# --- keep the HF cache off the home quota ----------------------------------
# Without this, huggingface_hub caches a second full copy under ~/.cache.

export HF_HOME="${HF_HOME:-$TARGET_DIR/.hf-cache}"

mkdir -p "$TARGET_DIR"

echo ">> CLI:      $HF_CLI"
echo ">> Target:   $TARGET_DIR"
echo ">> HF_HOME:  $HF_HOME"
if [[ "$CONFIGS_ONLY" -eq 1 ]]; then
  echo ">> Mode:     configs only"
else
  echo ">> Mode:     full download (several GB)"
fi
echo

# --- warn on low disk space ------------------------------------------------

avail_gb="$(df -BG --output=avail "$TARGET_DIR" 2>/dev/null | tail -n 1 | tr -dc '0-9' || echo "")"
if [[ -n "$avail_gb" && "$avail_gb" -lt 30 && "$CONFIGS_ONLY" -eq 0 ]]; then
  echo "WARNING: only ${avail_gb}G available at $TARGET_DIR." >&2
  echo "         Full checkpoints may not fit. Continuing anyway." >&2
  echo
fi

# --- download ---------------------------------------------------------------

for repo in "${REPOS[@]}"; do
  name="${repo##*/}"
  dest="$TARGET_DIR/$name"

  echo "=== $repo -> $dest"

  args=("download" "$repo" "--local-dir" "$dest")
  if [[ "$CONFIGS_ONLY" -eq 1 ]]; then
    args+=("--include" "configs/*" "*.yaml" "*.toml" "*.md")
  fi

  "$HF_CLI" "${args[@]}"
  echo
done

# --- report -----------------------------------------------------------------

echo "=== Done. Contents:"
for repo in "${REPOS[@]}"; do
  name="${repo##*/}"
  dest="$TARGET_DIR/$name"
  [[ -d "$dest" ]] || continue
  echo
  echo "--- $name  ($(du -sh "$dest" 2>/dev/null | cut -f1))"
  find "$dest" -type f -not -path '*/.cache/*' -printf '%10s  %p\n' 2>/dev/null \
    | sort -k2 \
    | sed "s|$dest/||"
done

cat <<NEXT_STEPS

Next:
  1. Read $TARGET_DIR/bris-forecaster/configs/config_training.yaml
     — it defines the dataset spec, variable list and normalisation setup.
  2. Set up the environment from the pinned lockfile:
       cd $TARGET_DIR/bris-forecaster && uv sync --locked
  3. Training needs the fork with the FFT-CRPS loss:
       https://github.com/evenmn/anemoi-core  (branch: feat/crps-fft-loss)
NEXT_STEPS
