#!/usr/bin/env bash
#
# One-command bootstrap for Bris inference on eX3.
#
#   git clone <this repo> && cd extreme-bris && ./run.sh
#
# Does every step that can currently succeed, in order, continuing past
# failures so that one broken step does not hide the state of the rest. Writes
# a report to $HOME/bris-runs/REPORT.md and prints it at the end.
#
#   ./run.sh              # full run
#   ./run.sh --no-gpu     # skip the GPU smoke job
#   ./run.sh --mail you@uio.no    # also mail the report if `mail` exists
#
# WHERE WORK HAPPENS, AND WHY
# ---------------------------
# Network-dependent steps (model download, uv sync) run on the LOGIN node with
# concurrency capped at 8 threads, which is what eX3 asks for. That is
# deliberate: compute nodes may have no outbound internet, and discovering that
# inside a queued job wastes a scheduling round-trip. The work is I/O bound
# rather than CPU bound, so it is a poor neighbour only if left uncapped.
#
# GPU work goes through sbatch, never an interactive srun, so nothing can be
# left holding a card.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BRIS_REPO_DIR="$REPO_DIR"
export BRIS_MODEL_DIR="${BRIS_MODEL_DIR:-$HOME/bris-models}"
export BRIS_DATA_DIR="${BRIS_DATA_DIR:-$HOME/bris-data}"
export BRIS_ENV_DIR="${BRIS_ENV_DIR:-$HOME/bris-env}"
RUN_DIR="${BRIS_RUN_DIR:-$HOME/bris-runs}"
REPORT="$RUN_DIR/REPORT.md"
LOG_DIR="$RUN_DIR/logs"

# eX3 login-node etiquette: at most 8 threads.
export OMP_NUM_THREADS=8
export UV_CONCURRENT_INSTALLS=8
export UV_CONCURRENT_DOWNLOADS=8
export PATH="$HOME/.local/bin:$PATH"

DO_GPU=1
MAIL_TO=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-gpu) DO_GPU=0; shift ;;
    --mail)   MAIL_TO="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$RUN_DIR" "$LOG_DIR" "$BRIS_DATA_DIR"

STEP_NAMES=(); STEP_STATUS=(); STEP_NOTES=(); STEP_LOGS=()
record() { STEP_NAMES+=("$1"); STEP_STATUS+=("$2"); STEP_NOTES+=("$3"); STEP_LOGS+=("${4:-}"); }

banner() { printf '\n\033[1m=== %s\033[0m\n' "$1"; }

# --- 1. uv ------------------------------------------------------------------
banner "1/6  uv"
if command -v uv >/dev/null 2>&1; then
  record "uv" "ok" "already present ($(uv --version 2>/dev/null))"
  echo "  already installed: $(uv --version)"
else
  if curl -LsSf https://astral.sh/uv/install.sh 2>>"$LOG_DIR/uv.log" | sh >>"$LOG_DIR/uv.log" 2>&1; then
    export PATH="$HOME/.local/bin:$PATH"
    record "uv" "ok" "installed $(uv --version 2>/dev/null)"
    echo "  installed: $(uv --version 2>/dev/null)"
  else
    record "uv" "FAILED" "see $LOG_DIR/uv.log — no outbound network from the login node?"
    echo "  FAILED (see $LOG_DIR/uv.log)"
  fi
fi

# --- 2. huggingface CLI -----------------------------------------------------
banner "2/6  huggingface CLI"
if command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; then
  record "hf-cli" "ok" "already present"
  echo "  already installed"
elif command -v uv >/dev/null 2>&1 && uv tool install huggingface_hub >>"$LOG_DIR/hf.log" 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  record "hf-cli" "ok" "installed via uv tool"
  echo "  installed"
else
  record "hf-cli" "FAILED" "see hf.log" "$LOG_DIR/hf.log"
  echo "  FAILED (see $LOG_DIR/hf.log)"
fi

# --- 3. checkpoints ---------------------------------------------------------
banner "3/6  model download (4.7 GB)"
CKPT="$BRIS_MODEL_DIR/bris-forecaster/bris-crpsfft_inference.ckpt"
if [[ -f "$CKPT" ]]; then
  record "download" "ok" "already present ($(du -sh "$BRIS_MODEL_DIR" 2>/dev/null | cut -f1))"
  echo "  already downloaded"
elif "$REPO_DIR/scripts/download_models.sh" >>"$LOG_DIR/download.log" 2>&1; then
  record "download" "ok" "$(du -sh "$BRIS_MODEL_DIR" 2>/dev/null | cut -f1) in $BRIS_MODEL_DIR"
  echo "  done: $(du -sh "$BRIS_MODEL_DIR" 2>/dev/null | cut -f1)"
else
  record "download" "FAILED" "see download.log" "$LOG_DIR/download.log"
  echo "  FAILED (see $LOG_DIR/download.log)"
fi

# --- 4. environment ---------------------------------------------------------
banner "4/6  environment (uv sync --locked)"
if [[ -d "$BRIS_ENV_DIR/.venv" ]]; then
  record "env" "ok" "already built at $BRIS_ENV_DIR"
  echo "  already built"
elif "$REPO_DIR/scripts/setup_env.sh" -d "$BRIS_ENV_DIR" >>"$LOG_DIR/env.log" 2>&1; then
  record "env" "ok" "built at $BRIS_ENV_DIR"
  echo "  built"
else
  note="see $LOG_DIR/env.log"
  grep -qi "permission denied (publickey)\|could not read from remote" "$LOG_DIR/env.log" 2>/dev/null \
    && note="SSH pin on metno/bris-inference not rewritten to HTTPS — $LOG_DIR/env.log"
  record "env" "FAILED" "$note" "$LOG_DIR/env.log"
  echo "  FAILED ($note)"
fi

# --- 5. checkpoint metadata -------------------------------------------------
banner "5/6  checkpoint metadata"
META="$RUN_DIR/ckpt-metadata.json"
if [[ ! -f "$CKPT" || ! -d "$BRIS_ENV_DIR/.venv" ]]; then
  record "metadata" "skipped" "needs both the checkpoint and the environment"
  echo "  skipped (missing checkpoint or environment)"
elif (cd "$BRIS_ENV_DIR" && uv run python "$REPO_DIR/scripts/inspect_checkpoint.py" \
        "$CKPT" --json "$META") >"$LOG_DIR/metadata.log" 2>&1; then
  nvars=$(grep -cE '^\s+[0-9]+\s+\S+$' "$LOG_DIR/metadata.log" 2>/dev/null || echo "?")
  record "metadata" "ok" "$nvars variables dumped to $META"
  echo "  done: $nvars variables -> $META"
else
  record "metadata" "FAILED" "see metadata.log" "$LOG_DIR/metadata.log"
  echo "  FAILED (see $LOG_DIR/metadata.log)"
fi

# --- 6. GPU smoke test ------------------------------------------------------
banner "6/6  GPU smoke test"
have_data=1
for ds in meps-2p5km-202501-202604-6h-v7.zarr era5-n320-2025-6h-v1.zarr; do
  [[ -e "$BRIS_DATA_DIR/$ds" ]] || have_data=0
done

if [[ "$DO_GPU" -eq 0 ]]; then
  record "smoke" "skipped" "--no-gpu given"
  echo "  skipped (--no-gpu)"
elif [[ "$have_data" -eq 0 ]]; then
  record "smoke" "blocked" "no input data in $BRIS_DATA_DIR — this is the open work, see docs/INPUTS.md"
  echo "  BLOCKED: no input data. This is expected today — see docs/INPUTS.md"
elif [[ ! -d "$BRIS_ENV_DIR/.venv" ]]; then
  record "smoke" "skipped" "environment not built"
  echo "  skipped (no environment)"
else
  echo "  submitting to a40q and waiting..."
  if sbatch --wait --output="$LOG_DIR/smoke-%j.out" --error="$LOG_DIR/smoke-%j.err" \
       "$REPO_DIR/bris/slurm/bris_smoke.sbatch" >>"$LOG_DIR/smoke-submit.log" 2>&1; then
    nc_count=$(ls "$HOME"/bris-runs/smoke/*.nc 2>/dev/null | wc -l | tr -d ' ')
    record "smoke" "ok" "$nc_count NetCDF file(s) written"
    echo "  done: $nc_count NetCDF file(s)"
  else
    record "smoke" "FAILED" "see smoke-*.err" "$(ls -t "$LOG_DIR"/smoke-*.err 2>/dev/null | head -1)"
    echo "  FAILED (see $LOG_DIR/smoke-*.err)"
  fi
fi

# --- report -----------------------------------------------------------------
{
  echo "# Bris on eX3 — bootstrap report"
  echo
  echo "- host: \`$(hostname)\`"
  echo "- when: $(date -u '+%Y-%m-%d %H:%M UTC')"
  echo "- repo: \`$REPO_DIR\` @ $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo
  echo "| step | status | detail |"
  echo "|---|---|---|"
  for i in "${!STEP_NAMES[@]}"; do
    echo "| ${STEP_NAMES[$i]} | ${STEP_STATUS[$i]} | ${STEP_NOTES[$i]} |"
  done
  echo
  # Anything that failed gets its log excerpted inline, so the report is
  # self-contained and can be pasted somewhere useful without chasing files.
  any_failed=0
  for i in "${!STEP_NAMES[@]}"; do
    [[ "${STEP_STATUS[$i]}" == "FAILED" ]] || continue
    any_failed=1
    echo "## Failure: ${STEP_NAMES[$i]}"
    echo
    lg="${STEP_LOGS[$i]}"
    if [[ -n "$lg" && -f "$lg" ]]; then
      echo '```'
      tail -25 "$lg"
      echo '```'
    else
      echo "(no log captured)"
    fi
    echo
  done

  if [[ "$any_failed" -eq 1 ]]; then
    echo "## Diagnostics"
    echo
    echo '```'
    echo "disk:"
    df -h "$HOME" 2>/dev/null | tail -2
    quota -s 2>/dev/null | tail -3 || echo "  (quota unavailable)"
    echo
    echo "reachability:"
    for u in https://pypi.org/simple/ https://files.pythonhosted.org/ \
             https://huggingface.co https://astral.sh; do
      printf "  %-34s " "$u"
      curl -sS -m 10 -o /dev/null -w "HTTP %{http_code}\n" "$u" 2>&1 | tail -1
    done
    echo
    echo "proxy env: ${http_proxy:-unset} / ${https_proxy:-unset}"
    echo '```'
    echo
  fi

  if [[ -f "$META" ]]; then
    echo "## Checkpoint metadata"
    echo
    echo "Written to \`$META\`. Variable order and normalisation statistics:"
    echo
    echo '```'
    head -40 "$LOG_DIR/metadata.log" 2>/dev/null
    echo '```'
    echo
  fi
  echo "## Next"
  echo
  if [[ "$have_data" -eq 0 ]]; then
    cat <<'NEXT'
Input data is the blocker, and it is real work rather than a missing flag.
Bris needs initial conditions on a cutout of MEPS 2.5 km inside and N320
outside. MET's N320 atmospheric source is MARS class `od`, which we do not have;
the plan is to substitute public ERA5 N320. See `docs/INPUTS.md`.

Build the checkpoint's variable order into that dataset — the metadata dumped
above is authoritative, the YAML is not.
NEXT
  else
    echo "Data present. Check the smoke output for NaNs and physical ranges (docs/INPUTS.md section 5)."
  fi
  echo
  echo "Logs: \`$LOG_DIR\`"
} > "$REPORT"

banner "REPORT"
cat "$REPORT"
echo
echo "Saved to $REPORT"

if [[ -n "$MAIL_TO" ]] && command -v mail >/dev/null 2>&1; then
  mail -s "Bris eX3 bootstrap report" "$MAIL_TO" < "$REPORT" && echo "Mailed to $MAIL_TO"
fi

for s in "${STEP_STATUS[@]}"; do [[ "$s" == "FAILED" ]] && exit 1; done
exit 0
