#!/usr/bin/env bash
#SBATCH --job-name=tailw-cpu
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=14:00:00
#SBATCH --output=logs/tailw-cpu-%j.out
#SBATCH --error=logs/tailw-cpu-%j.err
#
# One shard of the tail-weight measurement, on a whole CPU node.
#
#   sbatch bris/slurm/measure_tail_weight_cpu.sbatch 1/4
#   sbatch -p milanq bris/slurm/measure_tail_weight_cpu.sbatch 4/4
#   scripts/measure_tail_weight.py --merge ~/bris-runs/tail-weight/cpu-shard-*.json
#
# WHY A CPU. The measurement does not train, and the only cards large enough
# for this model are two H200s that can be held for a week. A 128-core node
# does one state in about 43 minutes in full precision, so four shards on four
# nodes finish all 46 states in roughly nine hours.
#
# --exclusive takes the whole node, because every core is used and a second
# job sharing it would slow both down without either saying so.

set -uo pipefail

SHARD="${1:?usage: sbatch $0 k/n}"
REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
VENV="${BRIS_ENV_DIR:-$HOME/bris-env}/.venv"

CORES="$(nproc)"
export OMP_NUM_THREADS="$CORES" MKL_NUM_THREADS="$CORES" SLURM_CPUS_PER_TASK="$CORES"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export ANEMOI_BASE_SEED="${ANEMOI_BASE_SEED:-20260909}"
export ANEMOI_INFERENCE_NUM_CHUNKS=16 ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=32
export XBRIS_SEQUENTIAL_MEMBERS=1

OUT="$HOME/bris-runs/tail-weight/cpu-shard-${SHARD/\//-of-}.json"
echo "=== node  : $(hostname), $CORES cores, $(free -g | awk 'NR==2{print $7}') GB free"
echo "=== shard : $SHARD -> $OUT"

"$VENV/bin/python" "$REPO_DIR/scripts/measure_tail_weight.py" --cpu --shard "$SHARD" --out "$OUT"
rc=$?
echo "=== finished rc=$rc"
exit $rc
