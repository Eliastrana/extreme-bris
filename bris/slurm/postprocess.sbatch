#!/usr/bin/env bash
#SBATCH --job-name=postprocess
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/postprocess-%j.out
#SBATCH --error=logs/postprocess-%j.err
#
# Apply the conversions anemoi's filters cannot, to one built dataset.
#
#   mkdir -p logs
#   sbatch bris/slurm/postprocess.sbatch ~/bris-data/meps-2p5km-year3-6h-v1.zarr
#
# One dataset per job, so the three years can run at the same time on
# different nodes. They touch different files and share only the filesystem.
#
# CHECK IT FIRST, ON THE LOGIN NODE. A dry run reads a single state and prints
# the rotation angle range and the humidity scale, which are the two things
# that decide whether the conversions are being applied correctly:
#
#   scripts/postprocess_meps.py <dataset> --dry-run
#
# NO MEMORY REQUEST, ON PURPOSE. The script holds one variable of one state at
# a time, a few megabytes, so whatever the partition hands out by default is
# ample. An explicit 32 GB was refused outright by defq, and a job that will
# not submit is worse than one that asks for too little. The line below prints
# what it actually got.
#
# TIME IS THE CONSTRAINT, NOT MEMORY. A year file means reading and rewriting
# roughly half a terabyte through zarr, bounded by the filesystem rather than
# by this node.
#
# To run somewhere else without editing this file:
#   sbatch -p milanq bris/slurm/postprocess.sbatch <dataset>
#
# INTERRUPTION IS SAFE NOW, AND WAS NOT BEFORE. Each variable is recorded as
# it completes. A job that hits the walltime can be resubmitted and will skip
# what is already converted. Without that, a partly converted dataset looks
# untouched, and a second run would rotate and convert those variables twice.

set -uo pipefail

DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
    echo "usage: sbatch $0 <path to .zarr>" >&2
    exit 1
fi
[[ -d "$DATASET" ]] || { echo "ERROR: no dataset at $DATASET" >&2; exit 1; }

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
[[ -f "$REPO_DIR/scripts/env.sh" ]] && source "$REPO_DIR/scripts/env.sh"

echo "=== dataset : $DATASET"
echo "=== node    : $(hostname)"
echo "=== memory  : $(free -g | awk 'NR==2{print $7" GB available of "$2" GB"}')"

# Python block-buffers stdout when it is a file, so a job that prints one line
# per variable shows nothing for hours and looks stalled. The progress is also
# written into the dataset attributes, which is the authoritative record, but a
# log you can tail is worth the flush.
export PYTHONUNBUFFERED=1

start=$(date +%s)
"$REPO_DIR/scripts/postprocess_meps.py" "$DATASET"
rc=$?
elapsed=$(( $(date +%s) - start ))
printf '\n=== finished rc=%d in %02d:%02d:%02d\n' \
    "$rc" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))

if [[ $rc -eq 0 ]]; then
    echo "=== CONVERTED - now check the rotation:"
    echo "    scripts/check_rotation.py $DATASET"
else
    echo "=== BLOCKED (rc=$rc). Resubmitting resumes from where it stopped." >&2
fi
exit $rc
