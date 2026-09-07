#!/usr/bin/env bash
#SBATCH --job-name=build-dataset
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/build-%j.out
#SBATCH --error=logs/build-%j.err
#
# Build an anemoi dataset as a batch job.
#
#   mkdir -p logs
#   sbatch --export=ALL,RECIPE=bris/configs/meps_year.yaml,\
# OUT=$BRIS_DATA_DIR/meps-2p5km-2025-2026-6h-v1.zarr,THREADS=4 \
#     bris/slurm/build_dataset.sbatch
#
# WHY THIS IS A JOB AND NOT A nohup ON THE LOGIN NODE.
# Every build so far has been run on the login node, on the assumption that
# compute nodes have no outbound internet. check_egress.sbatch was written to
# test exactly that and was never actually run. It is now: n001 reaches
# thredds.met.no (HTTP 200) and api.ecmwf.int (301). Compute nodes have egress.
#
# That matters at this scale. A one-week MEPS build measured 74 minutes; a year
# extrapolates to around 64 hours. That does not belong on a shared login node,
# where it competes with everyone else's editing and submitting, has no
# reserved memory, and dies with the node. defq has no wall-time limit, so the
# whole thing fits in one job.
#
# THREADS, NOT PROCESSES. The cost here is almost entirely waiting: each state
# opens one .ncml aggregation describing 67 underlying files, and parsing that
# dwarfs the transfer of the numbers themselves. Threads release the GIL during
# network I/O, which is where the time goes. Processes would also work but
# multiply the memory.
#
# Keep THREADS modest. thredds.met.no is a shared public service, not a private
# endpoint, and a build that hammers it for days is a bad neighbour.

set -uo pipefail

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
[[ -f "$REPO_DIR/scripts/env.sh" ]] && source "$REPO_DIR/scripts/env.sh"
DATA_ENV="${BRIS_DATA_ENV_DIR:-$HOME/bris-data-env}"

: "${RECIPE:?set RECIPE=<recipe.yaml>}"
: "${OUT:?set OUT=<output.zarr>}"
THREADS="${THREADS:-4}"

# KEEP THE GRIB CACHE OFF THE NODE'S LOCAL DISK. earthkit writes every field
# it retrieves into a temp dir under TMPDIR and never evicts any of it, so the
# cache grows monotonically for the whole build. On n001 that is a 40 GB
# partition shared with every other job on the node. A year of class od ran it
# to 22 GB by the third month and would have hit the wall around the seventh -
# after six hours of retrieval, with nothing kept.
#
# $HOME has hundreds of terabytes and already holds the zarr this writes, so
# the cache belongs there. Cleared on exit: it is scratch, not a result.
export TMPDIR="${BRIS_TMPDIR:-$HOME/bris-tmp}/${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR" || { echo "ERROR: cannot create $TMPDIR" >&2; exit 1; }
trap 'rm -rf "$TMPDIR"' EXIT

[[ -f "$RECIPE" ]] || { echo "ERROR: no recipe at $RECIPE" >&2; exit 1; }
[[ -x "$DATA_ENV/bin/python" ]] || { echo "ERROR: no data env at $DATA_ENV" >&2; exit 1; }

echo "=== node    : $(hostname)"
echo "=== recipe  : $RECIPE"
echo "=== out     : $OUT"
echo "=== threads : $THREADS"
echo "=== tmpdir  : $TMPDIR ($(df -h "$TMPDIR" | awk 'NR==2{print $4}') free)"
sed -n '/^dates:/,/^$/p' "$RECIPE"
echo

# A half-written zarr from a previous attempt is not a resumable state: it has
# the right shape and dates with nothing behind them, which is precisely the
# failure this week's finite check exists to catch.
[[ -e "$OUT" ]] && { echo "removing existing $OUT"; rm -rf "$OUT"; }

start=$(date +%s)
# The wrapper carries two patches the build cannot complete without: numpy
# integers made JSON-serialisable, and iterate_patterns handing each resolved
# URL only its own dates. See scripts/anemoi_create.py.
"$DATA_ENV/bin/python" "$REPO_DIR/scripts/anemoi_create.py" \
    "$RECIPE" "$OUT" --threads "$THREADS"
rc=$?
elapsed=$(( $(date +%s) - start ))
printf '\n=== build finished rc=%d in %02d:%02d:%02d\n' \
    "$rc" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))

# Structure is not data. A dataset with the right shape, the right dates and
# the right variable count can still be entirely NaN - that happened this week,
# ran a forecast to COMPLETED, and wrote 350 MB of nothing. Never report a
# build as usable without measuring what fraction of it is a number.
"$DATA_ENV/bin/python" - "$OUT" <<'PY'
import sys
import numpy as np
import zarr

try:
    g = zarr.open(sys.argv[1], mode="r")
    d = g["data"]
    dates = g["dates"][:]
    empty = []
    for s in range(d.shape[0]):
        a = np.asarray(d[s, :, 0, ::200], dtype="float64")
        if np.isfinite(a).mean() < 0.5:
            empty.append(str(dates[s]))
    print(f"=== shape   : {d.shape}")
    print(f"=== dates   : {len(dates)}  {dates[0]} .. {dates[-1]}")
    print(f"=== with data: {d.shape[0] - len(empty)} of {d.shape[0]}")
    if empty:
        print(f"=== EMPTY STATES ({len(empty)}): {empty[:8]}")
        sys.exit(3)
except Exception as exc:  # noqa: BLE001
    print(f"=== verification failed: {type(exc).__name__}: {exc}")
    sys.exit(3)
PY
vrc=$?
[[ $rc -eq 0 && $vrc -eq 0 ]] && echo "=== OK" || echo "=== NOT USABLE (build rc=$rc, verify rc=$vrc)"
exit $(( rc || vrc ))
