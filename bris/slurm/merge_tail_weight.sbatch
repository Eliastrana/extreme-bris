#!/usr/bin/env bash
#SBATCH --job-name=tailw-merge
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/tailw-merge-%j.out
#SBATCH --error=logs/tailw-merge-%j.err
#
# Merge the CPU shards of the tail-weight measurement once they have all ended,
# and only then retire the queued GPU measurement.
#
#   sbatch --dependency=afterany:J1:J2:J3:J4 bris/slurm/merge_tail_weight.sbatch 4 GPUJOB
#
# WHY A JOB AND NOT A WAITING TERMINAL. The first attempt waited for the shards
# inside an ssh session, and the connection through the NAT dropped half an
# hour in. The shards kept running, as Slurm jobs do, but nothing was left to
# merge them. A dependency job needs nobody connected.
#
# The GPU job is cancelled only if every shard produced output and the merge
# succeeded. A partial merge leaves it queued as the fallback it is.

set -uo pipefail

N="${1:?usage: sbatch --dependency=afterany:... $0 <number of shards> [gpu job to cancel]}"
GPU_JOB="${2:-}"
REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
VENV="${BRIS_ENV_DIR:-$HOME/bris-env}/.venv"
DIR="$HOME/bris-runs/tail-weight"

files=()
for k in $(seq 1 "$N"); do
    f="$DIR/cpu-shard-$k-of-$N.json"
    if [[ -f "$f" ]]; then files+=("$f"); else echo "MISSING $f"; fi
done
[[ ${#files[@]} -gt 0 ]] || { echo "no shard produced output"; exit 1; }

"$VENV/bin/python" "$REPO_DIR/scripts/measure_tail_weight.py" \
    --merge "${files[@]}" --out "$DIR/measure-cpu-merged.json"
rc=$?

if [[ -n "$GPU_JOB" ]]; then
    if [[ $rc -eq 0 && ${#files[@]} -eq $N ]]; then
        scancel "$GPU_JOB" && echo "=== cancelled the queued GPU measurement $GPU_JOB"
    else
        echo "=== left GPU measurement $GPU_JOB queued: merge rc=$rc, ${#files[@]} of $N shards"
    fi
fi
exit $rc
