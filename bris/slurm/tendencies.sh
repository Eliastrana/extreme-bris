#!/usr/bin/env bash
#
# Submit the three tendency stages for one dataset, chained by dependency.
#
#   bris/slurm/tendencies.sh ~/bris-data/meps-2p5km-year3-6h-v1.zarr
#   bris/slurm/tendencies.sh            # all six
#
# init runs alone, load runs as an array of parts across nodes, finalise waits
# for every part. Slurm enforces the order, so the whole thing can be fired and
# left. Nothing here needs a GPU, which is the point: it is work that can
# happen while the cards are unavailable.

set -uo pipefail

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
NPARTS="${NPARTS:-10}"
JOB="$REPO_DIR/bris/slurm/tendencies.sbatch"
mkdir -p "$REPO_DIR/logs"

datasets=("$@")
if [[ ${#datasets[@]} -eq 0 ]]; then
    for y in 1 2 3; do
        datasets+=("$HOME/bris-data/meps-2p5km-year$y-6h-v1.zarr")
        datasets+=("$HOME/bris-data/od-an-n320-year$y-6h-v1.zarr")
    done
fi

for ds in "${datasets[@]}"; do
    name=$(basename "$ds")
    [[ -d "$ds" ]] || { echo "skipping $name: not there" >&2; continue; }

    init=$(sbatch --parsable "$JOB" init "$ds") || exit 1
    load=$(sbatch --parsable --dependency=afterok:"$init" \
                  --array=1-"$NPARTS" "$JOB" load "$ds" "$NPARTS") || exit 1
    fin=$(sbatch --parsable --dependency=afterok:"$load" \
                 "$JOB" finalise "$ds") || exit 1
    echo "$name: init $init -> load $load (1-$NPARTS) -> finalise $fin"
done

echo
echo "Watch with: squeue --me"
echo "A part that fails leaves finalise held rather than run on half the data."
