#!/usr/bin/env bash
#
# End-to-end: build what is missing, run one Bris forecast, check the output.
#
#   ./run_inference.sh              # do everything that is not already done
#   ./run_inference.sh --check      # report state and exit, build nothing
#   ./run_inference.sh --date 2026-08-25T00:00:00
#
# --date drives everything: both recipes, both dataset names, the output
# directory and the inference. Datasets already built for that date are reused.
#
# STATUS
# ------
# The whole chain has run end to end on eX3 for 2025-04-01T00: both datasets
# built, forecast written, fields checked numerically and against geostrophic
# balance. What has NOT been exercised is any other date - the recipes were
# pinned until now, so every build so far used the same two states.
#
# So expect the first run at a new date to surface date-specific problems:
# a MEPS .ncml missing from the archive, or an ERA5 date not yet published.
# Both fail loudly rather than quietly.
#
# NOTE ON INPUTS: the global half is ERA5, not the operational analysis Bris
# was trained on. Fields are physical, but no number out of this belongs in a
# verification without saying that first. See docs/INPUTS.md.

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

# --- date-derived naming ----------------------------------------------------
# Everything below hangs off --date. Before this, --date reached only the
# inference: the zarr paths and the MEPS recipe were pinned to April 2025, so
# asking for another date ran the model against a dataset that did not contain
# it.
if ! T0="$(date -u -d "${DATE//T/ } UTC" +%Y-%m-%dT%H:%M:%S 2>/dev/null)"; then
  echo "ERROR: --date '$DATE' is not a date I can parse." >&2
  echo "       Expected YYYY-MM-DDTHH:MM:SS, e.g. 2026-08-25T00:00:00" >&2
  exit 1
fi
TAG="$(date -u -d "${DATE//T/ } UTC" +%Y%m%dT%HZ)"
# multistep_input = 2: the recipes must span t0-6h .. t0. One state is not enough.
PREV="$(date -u -d "${DATE//T/ } UTC - 6 hours" +%Y-%m-%dT%H:%M:%S)"

# The global half defaults to ERA5. BRIS_GLOBAL_ZARR points it at an already
# built dataset instead - specifically one from the operational analysis
# (class od), which is what Bris was actually trained on. Holding the LAM, the
# date and the model fixed and changing only this leaves the initialisation
# source as the single difference between two runs.
ERA5="${BRIS_GLOBAL_ZARR:-$BRIS_DATA_DIR/era5-n320-${TAG}-6h-v1.zarr}"
MEPS="$BRIS_DATA_DIR/meps-2p5km-${TAG}-6h-v1.zarr"

# The first case was built before any of this was date-driven. Retrieving ERA5
# through CDS costs hours of tape, so do not orphan a good dataset over a
# filename change.
#
# The guard belongs on the ERA5 line ALONE. Putting it on the whole block also
# disabled the MEPS fallback, which sent the script off to rebuild the LAM from
# a thredds date the archive no longer carries - the global override says
# nothing about where the LAM should come from.
if [[ "$T0" == "2025-04-01T00:00:00" ]]; then
  [[ -z "${BRIS_GLOBAL_ZARR:-}" && -e "$BRIS_DATA_DIR/era5-n320-2025-6h-v1.zarr" && ! -e "$ERA5" ]] && \
      ERA5="$BRIS_DATA_DIR/era5-n320-2025-6h-v1.zarr"
  [[ -e "$BRIS_DATA_DIR/meps-2p5km-2025-6h-v1.zarr" && ! -e "$MEPS" ]] && \
      MEPS="$BRIS_DATA_DIR/meps-2p5km-2025-6h-v1.zarr"
fi

RECIPES="$BRIS_RUN_DIR/recipes"
ERA5_RECIPE="$RECIPES/era5-${TAG}.yaml"
MEPS_RECIPE="$RECIPES/meps-${TAG}.yaml"
META="$BRIS_RUN_DIR/ckpt-metadata.json"
OUT="${BRIS_OUTPUT_DIR:-$BRIS_RUN_DIR/$TAG}"
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
step "state  —  t0 $T0   (inputs span $PREV .. $T0)"
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

for req in "$BRIS_ENV_DIR/.venv" "$BRIS_CKPT"; do
  have "$req" || { echo "ERROR: missing $req — run ./run.sh first" >&2; exit 1; }
done

mkdir -p "$RECIPES"

# ERA5 is a reanalysis, not an analysis. The preliminary stream (ERA5T) runs
# about five days behind real time and the final stream months behind, so a
# recent date is not slow to retrieve - it does not exist. Say so here rather
# than letting CDS return an empty result that builds a dataset with no data.
# Only ERA5 has this problem. The operational analysis is available within
# hours, so when BRIS_GLOBAL_ZARR names a prebuilt od dataset the lag guard
# below would reject dates that are perfectly retrievable.
AGE_DAYS=$(( ( $(date -u +%s) - $(date -u -d "${DATE//T/ } UTC" +%s) ) / 86400 ))
if [[ -n "${BRIS_GLOBAL_ZARR:-}" ]]; then
  :
elif (( AGE_DAYS < 0 )); then
  echo "ERROR: $T0 is in the future. ERA5 cannot initialise a forecast of the" >&2
  echo "       future; it is a reanalysis of the past." >&2
  exit 1
elif (( AGE_DAYS < 6 )); then
  echo "WARNING: $T0 is only $AGE_DAYS day(s) ago. ERA5T lags real time by" >&2
  echo "         roughly five days, so this date may not be published yet." >&2
  echo "         If the ERA5 build returns nothing, that is why - go further back." >&2
fi

# --- 4. ERA5 -----------------------------------------------------------------
if dataset_complete "$ERA5"; then
  step "4/7  ERA5 dataset — already built"
else
  have "$META" || { echo "ERROR: missing $META — run ./run.sh first" >&2; exit 1; }
  if have "$ERA5"; then
    step "4/7  ERA5 dataset — discarding incomplete build ($(n_vars "$ERA5") of $EXPECTED_VARS variables)"
    rm -rf "$ERA5"
  fi
  step "4/7  ERA5 dataset — generating recipe for $T0"
  "$DATA_ENV/bin/python" "$REPO_DIR/scripts/make_era5_recipe.py" \
      "$META" --date "$T0" --frequency 6h -o "$ERA5_RECIPE" \
      || { echo "recipe generation failed" >&2; exit 3; }
  step "4b/7 ERA5 dataset — building (CDS tape: this is the slow one)"
  "$DATA_ENV/bin/python" "$REPO_DIR/scripts/anemoi_create.py" \
      "$ERA5_RECIPE" "$ERA5" 2>&1 | tee "$LOGS/era5-build-$TAG.log"
  if ! dataset_complete "$ERA5"; then
    echo >&2
    echo "ERA5 build produced $(n_vars "$ERA5") of $EXPECTED_VARS variables." >&2
    tail -n 20 "$LOGS/era5-build-$TAG.log" >&2
    [[ -e "$ERA5" ]] && rm -rf "$ERA5"
    exit 3
  fi
fi

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
  step "5/7  MEPS dataset — building for $T0"
  # The recipe's OPeNDAP URLs are already {date:strftime(...)} templates, so
  # only the dates block is pinned. Substitute it into a per-date copy rather
  # than editing the checked-in recipe, which is hard-won and shared.
  sed -e "s|^  start: .*|  start: $PREV|" \
      -e "s|^  end: .*|  end: $T0|" \
      "$REPO_DIR/bris/configs/meps_2p5km.yaml" > "$MEPS_RECIPE"
  # Wrapped rather than called directly: 0.5.24's json_tidy cannot serialise
  # the numpy integers MEPS carries in its coordinates. See the script.
  "$DATA_ENV/bin/python" "$REPO_DIR/scripts/anemoi_create.py" \
      "$MEPS_RECIPE" "$MEPS" 2>&1 | tee "$LOGS/meps-build-$TAG.log"
  # A directory appearing is not a dataset. anemoi-datasets creates the zarr
  # before it fills it, so a failed build leaves a shell that `test -e` accepts
  # and `open_dataset` then rejects with a bare AttributeError.
  if ! dataset_complete "$MEPS"; then
    echo
    echo "MEPS build produced $(n_vars "$MEPS") of $EXPECTED_VARS variables." >&2
    echo "MEPS build failed. This is the expected stopping point today." >&2
    echo "The last 20 lines are the thing to work from:" >&2
    tail -n 20 "$LOGS/meps-build-$TAG.log" >&2
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

# The placement that actually completed: 4 x V100 on dgx2q, model sharded over
# all four. The sbatch file's own defaults (one A100 on hgx2q, GPUS_PER_MODEL=2)
# trip its ntasks guard, because Lightning needs one task per device. Encode
# what worked rather than leaving the caller to rediscover it.
PARTITION="${BRIS_PARTITION:-dgx2q}"
GRES="${BRIS_GRES:-gpu:tesla:4}"
NTASKS="${BRIS_NTASKS:-4}"
SHARD="${BRIS_GPUS_PER_MODEL:-4}"
LEADTIMES="${BRIS_LEADTIMES:-10}"        # 10 x 6h = 60h

echo "  $PARTITION  $GRES  ntasks=$NTASKS  shard=$SHARD  leadtimes=$LEADTIMES ($((LEADTIMES * 6))h)"
echo "  submitting and waiting"
sbatch --wait \
  --partition="$PARTITION" --gres="$GRES" --ntasks-per-node="$NTASKS" \
  --output="$LOGS/forecast-%j.out" --error="$LOGS/forecast-%j.err" \
  --export=ALL,BRIS_DATE="$DATE",BRIS_OUTPUT_DIR="$OUT",BRIS_GPUS_PER_MODEL="$SHARD",BRIS_LEADTIMES="$LEADTIMES",BRIS_GLOBAL_ZARR="$ERA5",BRIS_LAM_ZARR="$MEPS" \
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
