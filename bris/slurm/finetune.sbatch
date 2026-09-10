#!/usr/bin/env bash
#SBATCH --job-name=finetune
#SBATCH --partition=a100q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.err
#
# One job script, both arms of the experiment.
#
#   mkdir -p logs
#   sbatch bris/slurm/finetune.sbatch finetune        # control
#   sbatch bris/slurm/finetune.sbatch finetune_tail   # treatment
#
# Anything after the arm name is passed to anemoi as an override, which is how
# to probe one of these cheaply before committing two days to it. The tail arm
# is the one to probe, because it inherits everything the control arm has and
# adds a loss class that has never been instantiated:
#
#   sbatch -t 02:00:00 bris/slurm/finetune.sbatch finetune_tail \
#       training.max_steps=2 dataloader.limit_batches.training=2 \
#       dataloader.limit_batches.validation=1
#
# A run with overrides is not the run these configs describe, so the check for
# written weights below drops to a warning for it.
#
# RUN THESE TWO FIRST, ON THE LOGIN NODE. Both take seconds and both catch
# things that otherwise surface after the queue wait:
#
#   scripts/diff_configs.py finetune finetune_tail
#   scripts/tail_threshold.py --mm 20
#
# THE SEED IS FIXED AND SHARED. Both arms have to draw the same samples in the
# same order, or part of any difference between them is which states each one
# happened to see. This is the cheapest control in the whole design.
#
# THE GRES ASKS FOR CARDS, NOT FOR A100s. Naming the type pins the job to
# a100q, which is two cards in the whole cluster, and the queue there is deep.
# Written generically it can also land on hgx2q, dgx2q or h200q, and the
# partition list is what decides:
#
#   sbatch -p a100q,hgx2q,h200q --gres=gpu:1 -t 02:00:00 \
#       bris/slurm/finetune.sbatch finetune_tail training.max_steps=2 ...
#
# Slurm takes whichever of those can start it first. Leave the AMD partitions
# out: torch here is built for CUDA. Leave a40q and gh200q out too, since
# n009-n012 and gh001-gh002 are aarch64 and this environment is x86.
#
# NCCL_P2P_DISABLE: two A100s in one node deadlocked in a collective during an
# earlier run here, resident at full utilisation with no progress for 150
# seconds. Disabling peer-to-peer fixed it and costs little at this scale.

set -uo pipefail

CONFIG_NAME="${1:-${CONFIG_NAME:-finetune}}"
EXTRA=("${@:2}")

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
[[ -f "$REPO_DIR/scripts/env.sh" ]] && source "$REPO_DIR/scripts/env.sh"
VENV="${BRIS_ENV_DIR:-$HOME/bris-env}/.venv"
PY="$VENV/bin/python"
CONFIG_DIR="$REPO_DIR/bris/train"
CONFIG="$CONFIG_DIR/$CONFIG_NAME.yaml"

# xbris.losses is referenced by _target_ in the tail config, so hydra has to be
# able to import it. Nothing installs this package; the path is the whole
# mechanism.
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

for f in "$VENV/bin/anemoi-training" "$CONFIG" \
         "$HOME/bris-runs/graphs/n320_2p5k_7p10.pt" \
         "${BRIS_CKPT:-$HOME/bris-models/bris-forecaster/bris-crpsfft_inference.ckpt}"; do
    [[ -e "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

# A null threshold is caught by the loss too, but only after the model is
# built and the datasets are opened. grep costs nothing and answers now.
if grep -qE '^\s*tail_threshold:\s*null\s*$' "$CONFIG"; then
    echo "ERROR: tail_threshold is still null in $CONFIG" >&2
    echo "       run: scripts/tail_threshold.py --mm 20   and paste the number" >&2
    exit 1
fi

echo "=== arm     : $CONFIG_NAME"
[[ ${#EXTRA[@]} -gt 0 ]] && echo "=== override: ${EXTRA[*]}"
echo "=== node    : $(hostname)"
echo "=== memory  : $(free -g | awk 'NR==2{print $7" GB available of "$2" GB"}')"

# 21 GB of the card was reserved but unallocated when this first ran out of
# memory, which is fragmentation rather than genuine need. Expandable segments
# let the allocator grow a block instead of hunting for a contiguous one.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_P2P_DISABLE=1
export HYDRA_FULL_ERROR=1
export ANEMOI_BASE_SEED="${ANEMOI_BASE_SEED:-20260909}"
echo "=== seed    : $ANEMOI_BASE_SEED"

# Units, before anything expensive. The MEPS recipe renames
# precipitation_amount_acc to tp without rescaling kg/m2 to metres, and if
# that is still true the two halves of the cutout differ by a factor of a
# thousand at the seam. A run on that data trains and converges and answers a
# question nobody asked.
echo "=== checking precipitation units across the cutout"
"$PY" "$REPO_DIR/scripts/tail_threshold.py" \
      --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME" --mm 20
units_rc=$?
if [[ $units_rc -ne 0 ]]; then
    echo "ERROR: precondition failed (rc=$units_rc). Not starting a two-day run." >&2
    exit $units_rc
fi

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NGPU="${NGPU:-1}"
[[ "$NGPU" -lt 1 ]] && { echo "ERROR: no GPU visible" >&2; exit 1; }
echo "=== gpus    : $NGPU allocated"

# The allocation decides, not the config file. Asking sbatch for one card
# while the config said two is a mismatch nothing catches until Lightning
# tries to address a GPU it was never given.
#
# num_gpus_per_model stays at one: the ensemble is split across cards, the
# model is not sharded.
OUT_DIR="$HOME/bris-runs/train/$CONFIG_NAME"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR" || exit 1

start=$(date +%s)
"$VENV/bin/anemoi-training" train \
    --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME" \
    hardware.num_gpus_per_node="$NGPU" \
    hardware.num_gpus_per_ensemble="$NGPU" \
    hardware.num_gpus_per_model=1 \
    "${EXTRA[@]}"
rc=$?
elapsed=$(( $(date +%s) - start ))
printf '\n=== finished rc=%d in %02d:%02d:%02d\n' \
    "$rc" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))

# A run that exits clean having written no weights has produced nothing to
# score, and the only sign of it is a directory that is empty in a way nobody
# looks at until the next arm is already queued.
ckpt_dir="$HOME/bris-runs/train/checkpoint"
n_ckpt=$(find "$ckpt_dir" -name '*.ckpt' -newermt "@$start" 2>/dev/null | wc -l)
echo "=== wrote $n_ckpt checkpoint(s) under $ckpt_dir"
if [[ $rc -eq 0 && $n_ckpt -eq 0 ]]; then
    if [[ ${#EXTRA[@]} -gt 0 ]]; then
        echo "=== nothing saved, which is expected for an overridden run"
    else
        echo "=== FINISHED WITH NOTHING SAVED - check diagnostics.checkpoint" >&2
        exit 3
    fi
fi

[[ $rc -eq 0 ]] && echo "=== $CONFIG_NAME DONE" \
                || echo "=== $CONFIG_NAME BLOCKED (rc=$rc) - read the traceback above"
exit $rc
