#!/usr/bin/env bash
#SBATCH --job-name=train-smoke
#SBATCH --partition=a100q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err
#
# Can anemoi-training load the published Bris weights and take a step?
#
#   mkdir -p logs && sbatch bris/slurm/train_smoke.sbatch
#
# This is a feasibility probe, not a training run. Two steps on one month of
# data, no checkpoint written, nothing logged remotely. The whole fine-tuning
# plan depends on the answer and none of it is worth designing until it is
# yes, so it is worth finding out cheaply and early.
#
# NCCL_P2P_DISABLE: two A100s in one node deadlocked in a collective during an
# earlier inference run here, 677 MiB resident at 100% utilisation and no
# progress for 150 seconds. Disabling peer-to-peer fixed it and costs little
# at this scale.

set -uo pipefail

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
[[ -f "$REPO_DIR/scripts/env.sh" ]] && source "$REPO_DIR/scripts/env.sh"
VENV="${BRIS_ENV_DIR:-$HOME/bris-env}/.venv"

CONFIG_DIR="$REPO_DIR/bris/train"
CONFIG_NAME="${CONFIG_NAME:-smoke}"

for f in "$VENV/bin/anemoi-training" "$CONFIG_DIR/$CONFIG_NAME.yaml" \
         "$HOME/bris-runs/graphs/n320_2p5k_7p10.pt"; do
    [[ -e "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

echo "=== node    : $(hostname)"
echo "=== config  : $CONFIG_DIR/$CONFIG_NAME.yaml"
echo "=== gpus    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tr '\n' '; ')"
echo "=== memory  : $(free -g | awk 'NR==2{print $7" GB available of "$2" GB"}')"

export NCCL_P2P_DISABLE=1
export HYDRA_FULL_ERROR=1
# anemoi writes its own config dump next to the run; keep it out of the repo.
mkdir -p "$HOME/bris-runs/train"
cd "$HOME/bris-runs/train" || exit 1

start=$(date +%s)
"$VENV/bin/anemoi-training" train \
    --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME"
rc=$?
elapsed=$(( $(date +%s) - start ))
printf '\n=== finished rc=%d in %02d:%02d:%02d\n' \
    "$rc" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))

# The only thing this run can tell you. A non-zero rc with a traceback is a
# useful answer too: it names the next thing to fix.
[[ $rc -eq 0 ]] && echo "=== THE MACHINERY TURNS OVER" \
                || echo "=== BLOCKED (rc=$rc) - read the traceback above"
exit $rc
