# Frozen evaluation plan

Frozen 22 September 2026, before the three-model inference was scored.  The
machine-readable source of truth is `evaluation/evaluation_plan.json`; its full
SHA-256 is embedded in every result report.

## Question and primary endpoint

The causal comparison is **tail arm minus control arm**.  Both arms received
the same additional training and differ only in the tail loss.  The published,
untouched Bris checkpoint is a baseline, not the causal control.

The primary endpoint is:

- forecasts initialised at 00 UTC;
- the 06--06 UTC 24-hour precipitation window ending at **+30 h**;
- fair threshold-weighted CRPS with the censoring transform
  `v(x) = max(20 mm, x)`;
- averaged over every station-day in the strict common intersection of
  baseline, control, tail, MEPS, and finite Frost observations;
- reported as tail minus control, so a negative difference favours the tail
  arm;
- with a paired 95% interval from seven-day calendar-block bootstrap samples.

The primary result is positive only when the upper confidence bound is below
zero.  This rule concerns the primary tail metric only.  Ordinary-weather
costs are reported as effect sizes and intervals because no scientifically
defensible non-inferiority margin was fixed in advance.

## Secondary endpoints

The +54 h window is reported separately; lead times are never pooled.  At both
leads the report includes:

- fair CRPS over the complete precipitation distribution;
- fair twCRPS at 20 and 50 mm;
- fair and empirical Brier scores at 20 and 50 mm;
- exact reliability rows for the attainable four-member probabilities;
- hit rate, false-alarm ratio, and probability of false detection at a 50%
  event-probability decision threshold;
- bias, MAE, and RMSE of each Bris ensemble mean;
- deterministic MEPS bias, MAE, RMSE, and contingency statistics.

The same paired block bootstrap is used for model differences in fair CRPS,
twCRPS, Brier scores, and ensemble-mean MAE.  Threshold statistics conditioned
on observed exceedances remain descriptive and are not primary evidence; that
would create the forecaster's dilemma.

## Common cases and missing data

No model is scored on a case unavailable to another model.  A case is one
cycle, lead, station, and valid 06 UTC day.  It is included only when:

1. all three Bris NetCDF files exist for the cycle;
2. all ensemble members are finite for every Bris arm;
3. the deterministic MEPS value is finite;
4. the Frost daily total is finite and passes the fixed quality and gauge
   screens; and
5. the station lies within 5 km of the forecast grid point.

`case-ledger.csv` records every expected cycle and lead, whether it was
included, its common station-day count, and the exclusion reason.  The frozen
period is 1 August 2025 through 6 September 2026 inclusive: **402**, not 403,
expected daily cycles.

## Ensemble handling

The scorer preserves the member dimension.  Fair CRPS and fair Brier remove
the leading finite-ensemble bias.  Empirical Brier scores and reliability use
the raw member fraction, which for four members can only be 0, 0.25, 0.5,
0.75, or 1.

Bris NetCDF precipitation is treated as a per-step amount, matching the
inference output convention already checked by the single-event scorer.  The
MEPS station cache is already differenced from the run-start accumulation.

Corresponding members should use the same inference seed across baseline,
control, and tail.  The evaluator records seed-related NetCDF attributes when
present and warns when they are absent.  In that case the inference job logs
must be retained as the seed record; scoring cannot reconstruct unrecorded
random draws.

MEPS is deterministic in this experiment.  It is therefore compared to Bris
ensemble means and members with deterministic measures.  It is not presented
as a probabilistic ensemble competitor.

## Dependence and uncertainty

Stations under one weather system are not independent.  Bootstrap sampling
therefore keeps all stations from a valid date together and resamples fixed
seven-day calendar blocks.  Intervals apply to paired score differences, not
to two independently estimated model means.

## Running the frozen evaluation

Quote wildcard paths so the evaluator receives each model as one argument:

```bash
scripts/evaluate_experiment.py \
  --model 'baseline=~/bris-runs/evaluation/baseline/nordic_*.nc' \
  --model 'control=~/bris-runs/evaluation/control/nordic_*.nc' \
  --model 'tail=~/bris-runs/evaluation/tail/nordic_*.nc' \
  --meps ~/bris-runs/meps/precipitation_daily.npz \
  --observations ~/bris-runs/observations \
  --out-dir ~/bris-runs/evaluation-score
```

The command writes:

- `case-ledger.csv` — audit trail for included and excluded cycles;
- `report.json` — complete scores, reliability tables, provenance, and paired
  intervals; and
- `summary.md` — the compact results table and pre-registered primary result.

Do not alter the plan after looking at results.  A necessary correction should
create a new version of the JSON plan and document why the old evaluation was
invalid rather than silently replacing it.

On eX3, keep the NetCDF I/O off the login node.  Once the three inference job
IDs are known, the CPU scoring job can be chained behind all of them:

```bash
sbatch --dependency=afterok:${BASE_JOB}:${CONTROL_JOB}:${TAIL_JOB} \
  bris/slurm/evaluate_experiment.sbatch
```
