#!/usr/bin/env bash
#
# How far along is a running build_dataset job?
#
#   scripts/build_progress.sh              # every build-dataset job of mine
#   scripts/build_progress.sh 1381399      # one job
#
# WHY THIS EXISTS. A build prints one line per retrieved state into a log that
# grows to tens of megabytes, and the tqdm bar in the .err file restarts on
# every monthly group - so the "17:48 remaining" it shows is the remaining time
# for the current month, not for the job. Reading either one directly gives a
# number that looks like progress and is not. This reads the frontier date
# instead: the newest date the job has asked for, as a fraction of the range in
# the recipe. That is the same quantity for MEPS over OPeNDAP and for MARS.
#
# It also reports free space in the node's TMPDIR. earthkit caches every
# retrieved GRIB there and never evicts it, so a MARS build is a slow leak
# against a local disk that is much smaller than the one the zarr lands on.
# Running out of it is the likeliest way for a long build to die.

set -uo pipefail

export PATH=/cm/shared/apps/slurm/current/bin:$PATH
LOGS="${BRIS_LOG_DIR:-$HOME/extreme-bris/logs}"
PY="${BRIS_DATA_ENV_DIR:-$HOME/bris-data-env}/bin/python"

jobs=("$@")
if [[ ${#jobs[@]} -eq 0 ]]; then
    mapfile -t jobs < <(squeue --me -h -n build-dataset -o "%i")
fi
[[ ${#jobs[@]} -eq 0 ]] && { echo "no build-dataset jobs running"; exit 0; }

# Days between two ISO dates, via seconds. Used to place the frontier date
# inside the recipe's range.
secs() { date -d "${1//T/ }" +%s 2>/dev/null; }

for job in "${jobs[@]}"; do
    out="$LOGS/build-$job.out"
    read -r state elapsed node < <(
        squeue -h -j "$job" -o "%T %M %N" 2>/dev/null || echo "GONE - -")

    echo "=========== $job  $state  $elapsed  $node"
    [[ -f "$out" ]] || { echo "  no log at $out"; continue; }

    sed -n 's/^=== \(recipe\|out\) *: /  \1: /p' "$out"

    # The sbatch header echoes the recipe's dates block, so start and end are
    # in the log. Exclude those lines when looking for the frontier, or the
    # end date would always match and every job would read as finished.
    start=$(sed -n 's/^ *start: *//p' "$out" | head -1)
    end=$(sed -n 's/^ *end: *//p' "$out" | head -1)
    # The MARS client prefixes every line it relays with the wall-clock time it
    # polled, so today's date sits at the start of thousands of lines. Left in,
    # it is always the newest date in the file and every MARS build reads as
    # finished. Strip the prefix before looking for the frontier.
    # Skip the recipe header the sbatch echoes. It now carries a `missing:`
    # list, and those dates are in the future relative to the frontier: reading
    # them as progress reported a job 33 seconds old as 98% done.
    seen=$(sed -e 's/^20[0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9:]\{8\} //' "$out" \
           | sed -e '/^dates:/,/^[^ #-]/d' \
           | grep -v '^ *\(start\|end\|frequency\):' \
           | grep -oE '20[0-9]{2}-[0-9]{2}-[0-9]{2}(T[0-9]{2}:[0-9]{2}:[0-9]{2})?' \
           | sort | tail -1)

    if [[ -n "$start" && -n "$end" && -n "$seen" ]]; then
        s=$(secs "$start"); e=$(secs "$end"); f=$(secs "$seen")
        if [[ -n "$s" && -n "$e" && -n "$f" && "$e" -gt "$s" ]]; then
            pct=$(( (f - s) * 100 / (e - s) ))
            echo "  dates : $start .. $end"
            echo "  at    : $seen  (~${pct}%)"
        fi
    fi

    # Log silence is the signal worth seeing. MARS spends most of its time
    # queued at ECMWF with nothing to print, so a long gap is normal - but it
    # is also what a hang looks like, and only the length tells them apart.
    for f in "$out" "$LOGS/build-$job.err"; do
        [[ -f "$f" ]] && printf '  %-4s: last write %s\n' \
            "$(basename "$f" | sed 's/.*\.//')" \
            "$(date -d "@$(stat -c %Y "$f")" +%H:%M:%S)"
    done

    zarr=$(sed -n 's/^=== out *: //p' "$out" | head -1)
    [[ -n "$zarr" && -e "$zarr" ]] && echo "  zarr  : $(du -sh "$zarr" | cut -f1)"

    # The frontier above is the furthest-ahead worker, not the job. With four
    # threads, four monthly groups run at once, so the newest date in the log
    # belongs to whichever group happens to lead - four minutes in it read
    # December and called the job 23% done.
    #
    # anemoi keeps the real answer in the dataset: one completion flag per
    # group, and the number of states in each. That is the same array the
    # loader consults to decide what to skip, so it cannot drift from reality.
    [[ -n "$zarr" && -d "$zarr/_build" ]] && "$PY" - "$zarr" <<'PYEOF'
import sys
import zarr
try:
    b = zarr.open(sys.argv[1], mode="r")["_build"]
    flags, lengths = b["flags"][:], b["lengths"][:]
    done = int(sum(int(n) for f, n in zip(flags, lengths) if f))
    total = int(sum(int(n) for n in lengths))
    print(f"  built : {done} of {total} states  ({100*done//total}%), "
          f"{int(flags.sum())} of {len(flags)} groups")
except Exception as exc:
    print(f"  built : unreadable ({type(exc).__name__})")
PYEOF

    # TMPDIR is node-local, so this has to run on the node that holds the job.
    # The GRIB cache moved off the node in the disk fix, so reporting the
    # node's /tmp told us nothing: it read 0G while MARS was pulling gigabytes
    # into $HOME. Report where the fields actually land, and how fresh the
    # newest one is - for MARS that is the only sign of life there is, because
    # the client buffers its log and can look stalled while transferring.
    cache="${BRIS_CACHE_DIR:-$HOME/bris-cache}"
    if [[ -d "$cache" ]]; then
        newest=$(find "$cache" -type f -newermt "-10 minutes" -printf '%T@\n' 2>/dev/null \
                 | sort -n | tail -1)
        printf '  cache : %s' "$(du -sh "$cache" | cut -f1)"
        if [[ -n "$newest" ]]; then
            printf ', last write %s\n' "$(date -d "@${newest%.*}" +%H:%M:%S)"
        else
            printf ', nothing written in 10 min\n'
        fi
    fi
done
