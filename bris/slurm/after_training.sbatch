#!/usr/bin/env bash
#SBATCH --job-name=after-train
#SBATCH --propagate=NONE
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/after-train-%j.out
#SBATCH --error=logs/after-train-%j.err
#
# What happens after a screening run finishes, without anyone watching.
#
#   sbatch --dependency=afterok:TRAINJOB --export=ALL,ARM=tailA2000,TRAIN_JOB=TRAINJOB \
#       bris/slurm/after_training.sbatch
#
# WHY. The H200s come free for a few days at a time between week-long
# reservations and jobs. A finished run whose forecasts wait for someone to
# notice can lose the card to the next job in the queue. So each training job
# gets this as its follow-on:
#
#   1. find the run's step-2000 inference checkpoint, through the "Checkpoints
#      path" its log printed, and copy it to ~/bris-models/<arm>/
#   2. submit the validation forecasts from it, exactly as control2000's were
#      made: every second day of April to July 2025, 7 lead times, 4 members
#   3. if every arm in CHAIN/expected.txt has now submitted its forecasts,
#      submit the screening, after all of them
#
# --propagate=NONE, here and in every script in bris/slurm. Since 2026-09-29
# the login node caps address space at 16 GB (limits.conf, "* hard as"), and
# Slurm copies the submitting shell's limits into the job. Loading Bris needs
# far more address space than that, so every job submitted from the login
# node after that date failed with "cannot allocate memory" on nodes with
# terabytes free. The compute nodes' own limits are unlimited; NONE uses them.
# A job submitted from inside a job inherits the cap too, which is why the
# forecast and screening submissions below rely on it as well.
#
# The list of arms is a file, not an argument, so an arm added to the round
# later (a plan amendment) is one line in that file, not resubmitted jobs.

set -uo pipefail

ARM="${ARM:?set ARM, e.g. tailA2000}"
TRAIN_JOB="${TRAIN_JOB:?set TRAIN_JOB, the training job id}"
STEP="${STEP:-002000}"
REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
# One chain per screening round. Defaults are round 2's, which this was
# written for; a later round passes its own directory, plan, fixed models and
# output, so its files never meet the earlier round's.
CHAIN="${CHAIN_DIR:-$HOME/bris-runs/tail-variants/chain}"
PLAN="${PLAN:-evaluation/screening_round2_plan.json}"
FIXED_MODELS="${FIXED_MODELS:-control2000 tail2000}"
OUT_DIR="${OUT_DIR:-$HOME/bris-runs/tail-variants}"
V="$HOME/bris-runs/validation-forecasts"
export PATH=/cm/shared/apps/slurm/21.08.8/bin:$PATH
cd "$REPO_DIR" || exit 1

# DRY_RUN=1 prints every submission instead of making it, and keeps its own
# chain directory and copies nothing, so the logic can be tested on an old run.
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    CHAIN="${CHAIN_DIR:-$CHAIN-dry}"
    sbatch() { echo "DRY sbatch $*" >&2; echo "$RANDOM"; }
    cp() { echo "DRY cp $*" >&2; }
fi
mkdir -p "$CHAIN" logs

log="logs/finetune-$TRAIN_JOB.out"
# The last path, not the first: a job the cluster requeues (g003 crashed on
# 2026-10-03 and restarted tail-e from the beginning) may append a second run
# to the same log, and only the last one reached the step this looks for.
run_dir="$(grep -o 'Checkpoints path: .*' "$log" 2>/dev/null | tail -1 | sed 's/Checkpoints path: //')"
[[ -n "$run_dir" && -d "$run_dir" ]] || { echo "ERROR: no checkpoint path in $log" >&2; exit 1; }
ckpt="$(ls "$run_dir"/inference-anemoi-by_step-epoch_*-step_${STEP}.ckpt 2>/dev/null | head -1)"
[[ -n "$ckpt" ]] || { echo "ERROR: no step-$STEP inference checkpoint in $run_dir" >&2; exit 1; }

dest="$HOME/bris-models/$ARM/$ARM-step${STEP}_inference.ckpt"
mkdir -p "$(dirname "$dest")"
cp -n "$ckpt" "$dest"
echo "=== $ARM: $ckpt -> $dest"

# Either an H200 or a whole A100 80 GB will do for inference; a card on
# either may be held for a week, so asking for both is what starts it soonest.
fc=$(sbatch --parsable --partition=h200q,hgx2q --gres=gpu:1 --time=04:00:00 \
    --export=ALL,BRIS_TAG=$ARM,BRIS_CHECKPOINT=$dest,BRIS_OUTPUT_DIR=$V/$ARM,BRIS_START=2025-04-01,BRIS_END=2025-07-31,BRIS_STEP_DAYS=2,BRIS_LEADTIMES=7,BRIS_LAM_ZARR=meps-2p5km-year2-6h-v1.zarr,BRIS_GLOBAL_ZARR=od-an-n320-year2-6h-v1.zarr \
    bris/slurm/bris_forecast.sbatch) || { echo "ERROR: forecast submission failed" >&2; exit 1; }
echo "$fc" > "$CHAIN/$ARM.forecast-job"
echo "=== $ARM: forecasts submitted as $fc"

# The last arm to get here submits the screening. flock, so two arms that
# finish together cannot both decide they were last, or both decide they were not.
exec 9> "$CHAIN/.lock"
flock 9
[[ -f "$CHAIN/screening-job" ]] && { echo "=== screening already submitted"; exit 0; }
deps=""
models=""
while read -r arm; do
    [[ -z "$arm" ]] && continue
    [[ -f "$CHAIN/$arm.forecast-job" ]] || { echo "=== waiting for $arm; not last"; exit 0; }
    deps="$deps:$(cat "$CHAIN/$arm.forecast-job")"
    models="$models --model \"$arm=$V/$arm/nordic_*.nc\""
done < "$CHAIN/expected.txt"
for fixed in $FIXED_MODELS; do
    models="$models --model \"$fixed=$V/$fixed/nordic_*.nc\""
done
sj=$(sbatch --parsable --propagate=NONE --dependency=afterany$deps --partition=defq --cpus-per-task=8 --time=02:00:00 \
    --job-name=screen --output="$OUT_DIR/screening.log" \
    --wrap "source scripts/env.sh; export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1; git pull -q; ~/bris-env/.venv/bin/python scripts/screen_variants.py --plan $PLAN $models --observations $HOME/bris-runs/observations-val --cache $OUT_DIR/screening-cases.npz --out $OUT_DIR/screening.json")
echo "$sj" > "$CHAIN/screening-job"
echo "=== all arms in; screening submitted as $sj, after${deps//:/ }"
