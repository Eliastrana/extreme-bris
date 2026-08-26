#!/usr/bin/env bash
#SBATCH --job-name=egress-check
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/egress-%j.out
#SBATCH --error=logs/egress-%j.err
#
# Does a compute node have outbound internet?
#
#   mkdir -p logs && sbatch bris/slurm/check_egress.sbatch
#
# This decides where the dataset build can run. Building from CDS means hours to
# days of waiting on the MARS tape archive, which should not sit on the login
# node. But if compute nodes have no egress, it has to — and then it belongs in
# a nohup'd background process with the thread cap eX3 asks for, not a job.
#
# The login node is known to have egress: uv, pip and git all work there.

set -uo pipefail

REPO_DIR="${BRIS_REPO_DIR:-$HOME/extreme-bris}"
[[ -f "$REPO_DIR/scripts/env.sh" ]] && source "$REPO_DIR/scripts/env.sh"

echo "=== node: $(hostname)"
echo "=== partition: ${SLURM_JOB_PARTITION:-?}"
echo

fail=0
for u in https://pypi.org/simple/ \
         https://cds.climate.copernicus.eu/ \
         https://thredds.met.no/thredds/catalog/catalog.html \
         https://huggingface.co ; do
  printf "  %-52s " "$u"
  code=$(curl -sS -m 20 -o /dev/null -w "%{http_code}" "$u" 2>&1 | tail -1)
  echo "HTTP $code"
  [[ "$code" == 2* || "$code" == 3* ]] || fail=1
done

echo
echo "  DNS:   $(getent hosts pypi.org >/dev/null 2>&1 && echo ok || echo FAIL)"
echo "  proxy: ${https_proxy:-unset}"
echo

if [[ "$fail" -eq 0 ]]; then
  echo "VERDICT: compute nodes have egress — build datasets in a batch job."
else
  echo "VERDICT: no (or partial) egress — dataset builds must run on the login"
  echo "         node, backgrounded, at 8 threads. See docs/INPUTS.md."
fi
