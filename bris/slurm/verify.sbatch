#!/usr/bin/env bash
#SBATCH --job-name=bris-verify
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify-%j.out
#SBATCH --error=logs/verify-%j.err
#
# Fetch ERA5 truth and score a forecast against it. CPU only - no GPU is held
# while a download waits.
#
#   mkdir -p logs
#   sbatch --export=ALL,BRIS_DATE=2025-04-01T00:00:00 bris/slurm/verify.sbatch
#
# Chain it behind the forecast:
#
#   jid=$(sbatch --parsable --partition=dgx2q --gres=gpu:tesla:4 \
#           --ntasks-per-node=4 \
#           --export=ALL,BRIS_DATE=2025-04-01T00:00:00,BRIS_LEADTIMES=10,\
# BRIS_GPUS_PER_MODEL=4,BRIS_OUTPUT_DIR=$HOME/bris-runs/20250401T00Z \
#           bris/slurm/bris_smoke.sbatch)
#   sbatch --dependency=afterok:$jid \
#          --export=ALL,BRIS_DATE=2025-04-01T00:00:00 bris/slurm/verify.sbatch
#
# EGRESS. Fetching truth needs outbound HTTPS to CDS, and it is not settled
# whether compute nodes have it (that is what check_egress.sbatch answers). If
# the fetch fails here, run it once on the login node - it is minutes, not the
# hours the input build takes - and resubmit. This script skips the fetch when
# the truth file is already there, so that costs nothing.

set -uo pipefail

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
source "$REPO_DIR/scripts/env.sh"

DATA_ENV="${BRIS_DATA_ENV:-$HOME/bris-data-env}"
DATE="${BRIS_DATE:-2025-04-01T00:00:00}"
LEADTIMES="${BRIS_LEADTIMES:-10}"
TAG="$(date -u -d "${DATE//T/ } UTC" +%Y%m%dT%HZ)"

FCDIR="${BRIS_OUTPUT_DIR:-$BRIS_RUN_DIR/$TAG}"
TRUTH="${BRIS_TRUTH:-$BRIS_RUN_DIR/truth/era5-truth-$TAG.nc}"
OUT="${BRIS_VERIFY_DIR:-$REPO_DIR/results/verify}"

echo "=== date      : $DATE  (+0h .. +$((LEADTIMES * 6))h)"
echo "=== forecast  : $FCDIR"
echo "=== truth     : $TRUTH"
echo "=== output    : $OUT"
echo

# --- 1. truth ---------------------------------------------------------------
if [[ -s "$TRUTH" ]]; then
  echo ">>> truth already present, skipping fetch"
else
  echo ">>> fetching ERA5 truth"
  if ! "$DATA_ENV/bin/python" "$REPO_DIR/scripts/fetch_verification.py" \
        --date "$DATE" --leadtimes "$LEADTIMES" -o "$TRUTH"; then
    echo >&2
    echo "Truth fetch failed. If the error mentions connection, DNS or TLS," >&2
    echo "this node has no egress. Run it once on the login node:" >&2
    echo >&2
    echo "  ~/bris-data-env/bin/python ~/extreme-bris/scripts/fetch_verification.py \\" >&2
    echo "      --date $DATE --leadtimes $LEADTIMES -o $TRUTH" >&2
    echo >&2
    echo "then resubmit this job - it will skip the fetch." >&2
    exit 1
  fi
fi

# --- 2. forecast files ------------------------------------------------------
shopt -s nullglob
globals=("$FCDIR"/global_*.nc)
shopt -u nullglob
if [[ ${#globals[@]} -eq 0 ]]; then
  echo "ERROR: no global_*.nc in $FCDIR" >&2
  echo "       The global file is the one scored: bris already interpolates it" >&2
  echo "       to 0.25 deg, which is the grid the truth comes on." >&2
  exit 2
fi
echo ">>> scoring ${globals[0]}"

# --- 3. score ---------------------------------------------------------------
mkdir -p "$OUT"
"$BRIS_ENV_DIR/.venv/bin/python" "$REPO_DIR/scripts/verify_forecast.py" \
    --forecast "${globals[0]}" --truth "$TRUTH" -o "$OUT"
rc=$?

echo
if [[ $rc -eq 0 ]]; then
  echo ">>> done. Plots in $OUT"
  ls -lh "$OUT"/*.png 2>/dev/null
else
  echo ">>> scoring failed (rc=$rc)" >&2
fi
exit $rc
