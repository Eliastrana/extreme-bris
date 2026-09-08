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
#
# MEMORY: ONE BUILD PER NODE. A MEPS year peaks at 56 GB resident. Slurm on
# this cluster reports RealMemory=1 and does not track memory at all, so it
# will happily pack several of these onto one node and let the kernel sort it
# out. Three submitted together landed on the same 132 GB node, asked for
# 169 GB between them, and were killed with rc=137 after 33 minutes each.
#
# Submit with --exclusive when running more than one at a time. There is no
# scarcity to justify packing: rome16q had nine idle nodes when this happened.
#
#   sbatch -p rome16q --exclusive --export=ALL,RECIPE=...,OUT=... this.sbatch

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

# MAKE A RETRY CHEAP. earthkit's default cache policy is "off", which means a
# fresh temp dir per run: every failure throws away everything retrieved. Both
# year-one builds died after five hours, one on a missing archive file and one
# on a full disk, and both would have to fetch the whole year again.
#
# Policy "user" keeps the cache at a fixed path across runs, so a resubmit
# re-reads what it already has and only fetches the rest. Eviction is governed
# by maximum-cache-disk-usage, 95% by default, which $HOME will not approach.
#
# This is safe only because the inputs are immutable: an archived MEPS cycle
# and a MARS analysis do not change after the fact. If a source is ever
# corrected in place, delete $BRIS_CACHE_DIR - nothing else notices staleness.
export EARTHKIT_DATA_CACHE_POLICY=user
export EARTHKIT_DATA_USER_CACHE_DIRECTORY="${BRIS_CACHE_DIR:-$HOME/bris-cache}"
mkdir -p "$EARTHKIT_DATA_USER_CACHE_DIRECTORY"

[[ -f "$RECIPE" ]] || { echo "ERROR: no recipe at $RECIPE" >&2; exit 1; }
[[ -x "$DATA_ENV/bin/python" ]] || { echo "ERROR: no data env at $DATA_ENV" >&2; exit 1; }

echo "=== node    : $(hostname)"
echo "=== recipe  : $RECIPE"
echo "=== out     : $OUT"
echo "=== threads : $THREADS"
# Slurm cannot enforce this, so at least record it. A build killed by the
# kernel leaves rc=137 and no explanation anywhere else.
echo "=== memory  : $(free -g 2>/dev/null | awk 'NR==2{print $7" GB available of "$2" GB"}') (a MEPS year peaks near 56 GB)"
echo "=== tmpdir  : $TMPDIR ($(df -h "$TMPDIR" | awk 'NR==2{print $4}') free)"
echo "=== cache   : $EARTHKIT_DATA_USER_CACHE_DIRECTORY ($(du -sh "$EARTHKIT_DATA_USER_CACHE_DIRECTORY" 2>/dev/null | cut -f1) held)"
sed -n '/^dates:/,/^$/p' "$RECIPE"
echo

# TRY TO CONTINUE BEFORE THROWING AWAY. anemoi records which monthly groups
# are finished inside the dataset and its load step skips them, but nothing
# used that: this line deleted the target before every build, so a failure at
# hour four cost four hours. It happened twice.
#
# anemoi_resume.py continues only when the recipe stored in the dataset is
# identical to the one being submitted, and exits 4 otherwise - including when
# nothing is finished yet, where a full build costs no more. Any 4 falls
# through to the wipe below, so the safe path is also the default.
resumed=0
start=$(date +%s)
if [[ -e "$OUT" && "${RESUME:-1}" != "0" ]]; then
    echo "=== trying to resume $OUT"
    "$DATA_ENV/bin/python" "$REPO_DIR/scripts/anemoi_resume.py" \
        "$RECIPE" "$OUT" --threads "$THREADS"
    rr=$?
    if [[ $rr -eq 0 ]]; then
        resumed=1
        rc=0
    elif [[ $rr -ne 4 ]]; then
        echo "=== resume failed with rc=$rr; not falling back to a wipe" >&2
        exit $rr
    fi
fi

# A half-written zarr that cannot be resumed is not a starting point: it has
# the right shape and dates with nothing behind them, which is precisely the
# failure the finite check exists to catch.
if [[ $resumed -eq 0 ]]; then
[[ -e "$OUT" ]] && { echo "removing existing $OUT"; rm -rf "$OUT"; }

# The wrapper carries the patches the build cannot complete without: numpy
# integers made JSON-serialisable, iterate_patterns handing each resolved URL
# only its own dates, and the accumulation stream picked by date. See
# scripts/anemoi_patches.py.
"$DATA_ENV/bin/python" "$REPO_DIR/scripts/anemoi_create.py" \
    "$RECIPE" "$OUT" --threads "$THREADS"
rc=$?
fi
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

    # A state the recipe declared missing is empty on purpose. Six of year one
    # are: 2026-09-02 has no 00Z or 06Z cycle in the archive, and 2026-01-31
    # serves half the pressure levels the recipe asks for. Counting those as
    # failures would report every correct build as NOT USABLE, and a check
    # that cries wolf is a check nobody reads.
    declared = {np.datetime64(x, "s")
                for x in g.attrs.get("missing_dates", [])}

    empty = []
    for s in range(d.shape[0]):
        if np.datetime64(dates[s], "s") in declared:
            continue
        a = np.asarray(d[s, :, 0, ::200], dtype="float64")
        if np.isfinite(a).mean() < 0.5:
            empty.append(str(dates[s]))

    expected = d.shape[0] - len(declared)
    print(f"=== shape   : {d.shape}")
    print(f"=== dates   : {len(dates)}  {dates[0]} .. {dates[-1]}")
    if declared:
        print(f"=== declared missing: {len(declared)}")
    print(f"=== with data: {expected - len(empty)} of {expected}")
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
