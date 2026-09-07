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

    # TMPDIR is node-local, so this has to run on the node that holds the job.
    # One line for the node, not for the job: the caches are unlabelled temp
    # dirs, so there is no honest way to attribute them to one job of several
    # sharing a node. What matters is the headroom either way.
    [[ "$state" == "RUNNING" ]] && srun --jobid="$job" --overlap --quiet bash -c '
        c=$(du -sc /tmp/tmp*/ 2>/dev/null | tail -1 | cut -f1)
        [[ -n "$c" ]] && echo "  caches: $((c/1048576))G in /tmp on $(hostname) (all jobs)"
        echo "  /tmp  : $(df -h /tmp | awk "NR==2{print \$4\" free of \"\$2}")"
    ' 2>/dev/null
done
