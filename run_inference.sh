#!/usr/bin/env bash
#
# End-to-end: build what is missing, run one Bris forecast, check the output.
#
#   ./run_inference.sh              # do everything that is not already done
#   ./run_inference.sh --check      # report state and exit, build nothing
#   ./run_inference.sh --date 2025-04-01T00:00:00
#
# HONEST STATUS
# -------------
# Stages 1-4 have all been run successfully on eX3. Stage 5, building the MEPS
# dataset, has NOT — its recipe is a first draft written against the archive's
# metadata and the anemoi-datasets source, not against a successful run. That is
# the gate. Stages 6-7 cannot have been tested either, because they need stage 5.
#
# So this script is honest about where it is: it will get you to the first real
# error in the MEPS build, and everything before that is known to work.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/scripts/env.sh"

DATA_ENV="${BRIS_DATA_ENV:-$HOME/bris-data-env}"
DATE="${BRIS_DATE:-2025-04-01T00:00:00}"
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --date)  DATE="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

ERA5="$BRIS_DATA_DIR/era5-n320-2025-6h-v1.zarr"
MEPS="$BRIS_DATA_DIR/meps-2p5km-2025-6h-v1.zarr"
OUT="${BRIS_OUTPUT_DIR:-$BRIS_RUN_DIR/smoke}"
LOGS="$BRIS_RUN_DIR/logs"
mkdir -p "$LOGS" "$OUT" logs

step() { printf '\n\033[1m=== %s\033[0m\n' "$1"; }
have() { [[ -e "$1" ]]; }

# A directory existing is not a dataset. anemoi-datasets creates the zarr shell
# before filling it, so an interrupted or failed build leaves something `test -e`
# accepts and open_dataset rejects. Every decision about whether a dataset is
# present has to go through this, not through `have`.
dataset_ok() {
  [[ -e "$1" ]] && "$DATA_ENV/bin/anemoi-datasets" inspect "$1" >/dev/null 2>&1
}

# Variable count, read from the zarr attributes. Parsing `inspect` output for it
# is a trap: the shape line separates fields with U+00D7, not an ASCII x.
n_vars() {
  "$DATA_ENV/bin/python" -c "
import sys, zarr
try:
    print(len(zarr.open(sys.argv[1], mode='r').attrs['variables']))
except Exception:
    print(0)
" "$1" 2>/dev/null
}

# Openable AND holding what the checkpoint needs. A dataset can be perfectly
# valid and still be missing a whole branch of the join, which is how a
# 17-variable MEPS dataset survived a run and skipped its own rebuild.
# 89 stored fields plus the 9 computed forcings, which have to be stored
# variables here: the inference config selects all 98 by name.
EXPECTED_VARS=98

dataset_complete() {
  dataset_ok "$1" && [[ "$(n_vars "$1")" == "$EXPECTED_VARS" ]]
}

# --- state ------------------------------------------------------------------
step "state"
printf "  %-26s %s\n" "environment"  "$([[ -d $BRIS_ENV_DIR/.venv ]] && echo ok || echo MISSING)"
printf "  %-26s %s\n" "data environment" "$([[ -d $DATA_ENV ]] && echo ok || echo MISSING)"
printf "  %-26s %s\n" "checkpoint"   "$(have "$BRIS_CKPT" && echo ok || echo MISSING)"
printf "  %-26s %s\n" "ERA5 (global)" "$(dataset_ok "$ERA5" && echo ok || echo MISSING)"
if dataset_ok "$MEPS"; then
  printf "  %-26s %s\n" "MEPS (LAM)" "ok"
elif have "$MEPS"; then
  printf "  %-26s %s\n" "MEPS (LAM)" "PARTIAL  <- failed build, will be removed"
else
  printf "  %-26s %s\n" "MEPS (LAM)" "MISSING  <- the gate"
fi

[[ "$CHECK_ONLY" -eq 1 ]] && exit 0

for req in "$BRIS_ENV_DIR/.venv" "$BRIS_CKPT" "$ERA5"; do
  have "$req" || { echo "ERROR: missing $req — run ./run.sh first" >&2; exit 1; }
done

# --- 5. MEPS ----------------------------------------------------------------
if dataset_complete "$MEPS"; then
  step "5/7  MEPS dataset — already built"
  "$DATA_ENV/bin/python" "$REPO_DIR/scripts/postprocess_meps.py" "$MEPS" --dry-run \
      >/dev/null 2>&1 || echo "  (already post-processed)"
else
  if have "$MEPS"; then
    step "5/7  MEPS dataset — discarding incomplete build ($(n_vars "$MEPS") of $EXPECTED_VARS variables)"
    rm -rf "$MEPS"
  fi
  step "5/7  MEPS dataset — building (UNTESTED RECIPE, expect to iterate)"
  # Wrapped rather than called directly: 0.5.24's json_tidy cannot serialise
  # the numpy integers MEPS carries in its coordinates. See the script.
  "$DATA_ENV/bin/python" "$REPO_DIR/scripts/anemoi_create.py" \
      "$REPO_DIR/bris/configs/meps_2p5km.yaml" "$MEPS" 2>&1 | tee "$LOGS/meps-build.log"
  # A directory appearing is not a dataset. anemoi-datasets creates the zarr
  # before it fills it, so a failed build leaves a shell that `test -e` accepts
  # and `open_dataset` then rejects with a bare AttributeError.
  if ! dataset_complete "$MEPS"; then
    echo
    echo "MEPS build produced $(n_vars "$MEPS") of $EXPECTED_VARS variables." >&2
    echo "MEPS build failed. This is the expected stopping point today." >&2
    echo "The last 20 lines are the thing to work from:" >&2
    tail -n 20 "$LOGS/meps-build.log" >&2
    [[ -e "$MEPS" ]] && echo "(removing the partial zarr at $MEPS)" >&2 && rm -rf "$MEPS"
    exit 2
  fi

  # Every physical conversion happens here rather than in the recipe, because
  # anemoi's conversion filters are GRIB-only. Skipping this leaves the dataset
  # holding grid-relative winds, relative humidity in the 2d column and m/s in
  # the w columns — all of which produce a forecast that runs and is wrong.
  step "5b/7  physical conversions"
  "$DATA_ENV/bin/python" "$REPO_DIR/scripts/postprocess_meps.py" "$MEPS" \
      || { echo "post-processing failed — the dataset is NOT usable as it stands" >&2; exit 5; }
fi

# --- 6. verify the inputs before spending a GPU ------------------------------
step "6/7  verifying both datasets"
"$DATA_ENV/bin/anemoi-datasets" inspect "$ERA5" | sed -n '1,12p'
echo
"$DATA_ENV/bin/anemoi-datasets" inspect "$MEPS" | sed -n '1,12p'

# The join must contribute both branches. A missing branch is silent: the build
# succeeds and the dataset is simply short, which only shows up as a variable
# count that does not match the checkpoint.
meps_vars=$(n_vars "$MEPS")
era5_vars=$(n_vars "$ERA5")
echo
echo "  variables: MEPS ${meps_vars:-?}, ERA5 ${era5_vars:-?} (both must be $EXPECTED_VARS)"
if [[ "${meps_vars:-0}" != "$EXPECTED_VARS" || "${era5_vars:-0}" != "$EXPECTED_VARS" ]]; then
  echo "  Variable count does not match the checkpoint's $EXPECTED_VARS." >&2
  echo "  The cutout cannot be assembled from these. Stopping." >&2
  exit 6
fi

# trim_edge needs a 2D field shape on the LAM side, and refuses anything else.
# Read it from the attributes: piping `inspect` into grep -q makes grep exit at
# the first match, which breaks the pipe and misreports the result.
meps_shape=$("$DATA_ENV/bin/python" -c "
import sys, zarr
try:
    print(len(zarr.open(sys.argv[1], mode='r').attrs.get('field_shape', [])))
except Exception:
    print(0)
" "$MEPS" 2>/dev/null)
if [[ "${meps_shape:-0}" == "2" ]]; then
  echo "  MEPS field shape is 2D — trim_edge will accept it"
else
  echo "  MEPS field shape is not 2D." >&2
  echo "  trim_edge will refuse with 'TrimEdge only works on regular grids'," >&2
  echo "  so there is no point spending a GPU allocation on this. Stopping." >&2
  exit 4
fi

# --- 7. forecast ------------------------------------------------------------
step "7/7  forecast"
echo "  submitting bris_smoke.sbatch and waiting"
sbatch --wait \
  --output="$LOGS/forecast-%j.out" --error="$LOGS/forecast-%j.err" \
  --export=ALL,BRIS_DATE="$DATE",BRIS_OUTPUT_DIR="$OUT" \
  "$REPO_DIR/bris/slurm/bris_smoke.sbatch"
rc=$?

nc_files=$(ls "$OUT"/*.nc 2>/dev/null | wc -l | tr -d ' ')
echo
if [[ "$rc" -ne 0 || "$nc_files" -eq 0 ]]; then
  echo "Forecast did not produce output. Last error log:" >&2
  tail -n 25 "$(ls -t "$LOGS"/forecast-*.err 2>/dev/null | head -n 1)" >&2
  exit 3
fi

step "output"
ls -lh "$OUT"/*.nc
echo
"$BRIS_ENV_DIR/.venv/bin/python" "$REPO_DIR/scripts/check_forecast.py" "$OUT"/*.nc
