"""Pure NumPy scoring helpers for the Bris extreme-weather evaluation.

The functions in this module deliberately know nothing about NetCDF, Frost, or
file layouts.  That keeps the statistical core small enough to test with
hand-calculated synthetic examples before the real test-period forecasts are
opened.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _ensemble_and_truth(ensemble, observations) -> tuple[np.ndarray, np.ndarray]:
    ens = np.asarray(ensemble, dtype="float64")
    obs = np.asarray(observations, dtype="float64")
    if ens.ndim != 2:
        raise ValueError(f"ensemble must be cases by members, got shape {ens.shape}")
    if obs.ndim != 1 or obs.shape[0] != ens.shape[0]:
        raise ValueError(
            f"observations must have one value per case, got {obs.shape} for {ens.shape}"
        )
    if ens.shape[1] < 1:
        raise ValueError("ensemble has no members")
    if not np.isfinite(ens).all() or not np.isfinite(obs).all():
        raise ValueError("scores require the caller's common-case finite mask")
    return ens, obs


def fair_crps_cases(ensemble, observations) -> np.ndarray:
    """Fair CRPS for every case, with finite-ensemble bias removed.

    The fair estimator requires at least two members.  Pair differences are
    accumulated without materialising a cases x members x members array, which
    matters for roughly half a million station-days.
    """
    ens, obs = _ensemble_and_truth(ensemble, observations)
    members = ens.shape[1]
    if members < 2:
        raise ValueError("fair CRPS requires at least two ensemble members")
    skill = np.abs(ens - obs[:, None]).mean(axis=1)
    pair_sum = np.zeros(ens.shape[0], dtype="float64")
    for i, j in combinations(range(members), 2):
        pair_sum += np.abs(ens[:, i] - ens[:, j])
    return skill - pair_sum / (members * (members - 1))


def threshold_weighted_crps_cases(ensemble, observations, threshold: float) -> np.ndarray:
    """Fair twCRPS using the censoring transform v(x) = max(threshold, x)."""
    ens, obs = _ensemble_and_truth(ensemble, observations)
    return fair_crps_cases(np.maximum(ens, threshold), np.maximum(obs, threshold))


def exceedance_probability(ensemble, threshold: float) -> np.ndarray:
    ens = np.asarray(ensemble, dtype="float64")
    if ens.ndim != 2:
        raise ValueError("ensemble must be cases by members")
    return (ens >= threshold).mean(axis=1)


def brier_cases(ensemble, observations, threshold: float) -> np.ndarray:
    ens, obs = _ensemble_and_truth(ensemble, observations)
    probability = exceedance_probability(ens, threshold)
    event = (obs >= threshold).astype("float64")
    return (probability - event) ** 2


def fair_brier_cases(ensemble, observations, threshold: float) -> np.ndarray:
    """Finite-ensemble-unbiased Brier score for an exceedance event."""
    ens, obs = _ensemble_and_truth(ensemble, observations)
    members = ens.shape[1]
    if members < 2:
        raise ValueError("fair Brier score requires at least two members")
    hits = (ens >= threshold).sum(axis=1).astype("float64")
    event = (obs >= threshold).astype("float64")
    return hits * (hits - 1) / (members * (members - 1)) - 2 * hits / members * event + event


def deterministic_error_cases(forecast, observations) -> dict[str, np.ndarray]:
    fc = np.asarray(forecast, dtype="float64")
    obs = np.asarray(observations, dtype="float64")
    if fc.shape != obs.shape or fc.ndim != 1:
        raise ValueError("forecast and observations must be equal one-dimensional arrays")
    error = fc - obs
    return {"error": error, "absolute_error": np.abs(error), "squared_error": error ** 2}


def strict_common_mask(observations, deterministic, ensembles: list[np.ndarray]) -> np.ndarray:
    """Cases finite in truth, the deterministic benchmark, and every member."""
    obs = np.asarray(observations, dtype="float64")
    det = np.asarray(deterministic, dtype="float64")
    if obs.ndim != 1 or det.shape != obs.shape:
        raise ValueError("observations and deterministic benchmark must be equal 1-D arrays")
    common = np.isfinite(obs) & np.isfinite(det)
    for ensemble in ensembles:
        ens = np.asarray(ensemble, dtype="float64")
        if ens.ndim != 2 or ens.shape[1] != obs.size:
            raise ValueError("each ensemble must be members by the same cases as observations")
        common &= np.isfinite(ens).all(axis=0)
    return common


def contingency(forecast_event, observed_event) -> dict:
    forecast = np.asarray(forecast_event, dtype=bool)
    observed = np.asarray(observed_event, dtype=bool)
    if forecast.shape != observed.shape:
        raise ValueError("forecast and observed events must have the same shape")
    hits = int((forecast & observed).sum())
    misses = int((~forecast & observed).sum())
    false = int((forecast & ~observed).sum())
    correct = int((~forecast & ~observed).sum())
    return {
        "observed": hits + misses,
        "forecast": hits + false,
        "hits": hits,
        "misses": misses,
        "false_alarms": false,
        "correct_negatives": correct,
        "hit_rate": hits / (hits + misses) if hits + misses else None,
        "false_alarm_ratio": false / (hits + false) if hits + false else None,
        "probability_of_false_detection": false / (false + correct) if false + correct else None,
    }


def reliability_table(probabilities, events) -> list[dict]:
    """One reliability row per attainable ensemble probability."""
    probability = np.asarray(probabilities, dtype="float64")
    event = np.asarray(events, dtype=bool)
    if probability.shape != event.shape:
        raise ValueError("probabilities and events must have the same shape")
    rows = []
    for value in np.unique(np.round(probability, 12)):
        keep = np.isclose(probability, value, atol=1e-12, rtol=0)
        rows.append({
            "forecast_probability": float(value),
            "cases": int(keep.sum()),
            "observed_frequency": float(event[keep].mean()),
            "mean_brier": float(((probability[keep] - event[keep]) ** 2).mean()),
        })
    return rows


def deaccumulate(series, convention: str = "auto") -> tuple[np.ndarray, str]:
    """Return per-step amounts for a time x member x station array."""
    values = np.asarray(series, dtype="float64")
    if values.ndim != 3:
        raise ValueError("series must be time by member by station")
    if convention not in {"auto", "per_step", "cumulative"}:
        raise ValueError(f"unknown accumulation convention {convention!r}")
    chosen = convention
    if chosen == "auto":
        finite = np.where(np.isfinite(values), values, 0.0)
        rising = np.diff(finite, axis=0) >= -1e-6
        chosen = "cumulative" if rising.size and rising.mean() > 0.98 else "per_step"
    if chosen == "cumulative":
        values = np.diff(values, axis=0, prepend=np.zeros_like(values[:1]))
    return np.maximum(values, 0.0), chosen


def daily_sums(step_amounts, leads_hours, end_leads_hours) -> np.ndarray:
    """24 h sums ending at selected leads.

    A value at lead L is the amount in the interval ending at L.  Consequently
    the 06--06 UTC day ending at +30 h consists of the four steps ending at
    +12, +18, +24, and +30 h.  The analysis at +0 h and the +6 h step are not
    part of that day.
    """
    step = np.asarray(step_amounts, dtype="float64")
    leads = np.asarray(leads_hours, dtype=int)
    ends = np.asarray(end_leads_hours, dtype=int)
    if step.ndim != 3 or step.shape[0] != leads.size:
        raise ValueError("step amounts must be time by member by station and match leads")
    if leads.size < 2:
        raise ValueError("at least two lead times are required")
    spacing = np.diff(leads)
    if not np.all(spacing == spacing[0]) or int(spacing[0]) <= 0:
        raise ValueError("lead times must be strictly increasing and evenly spaced")
    step_hours = int(spacing[0])
    if 24 % step_hours:
        raise ValueError(f"{step_hours} h steps do not divide a 24 h window")
    count = 24 // step_hours
    where = {int(lead): i for i, lead in enumerate(leads)}
    out = []
    for end in ends:
        wanted = [int(end) - step_hours * (count - 1 - i) for i in range(count)]
        if wanted[0] <= 0:
            raise ValueError(f"day ending at +{int(end)} h reaches the analysis step")
        missing = [lead for lead in wanted if lead not in where]
        if missing:
            raise ValueError(f"day ending at +{int(end)} h is missing leads {missing}")
        out.append(step[[where[lead] for lead in wanted]].sum(axis=0))
    return np.stack(out, axis=0)  # end lead, member, station


def paired_block_bootstrap(
    differences,
    valid_times,
    *,
    block_days: int = 7,
    replicates: int = 5000,
    seed: int = 20260922,
    confidence: float = 0.95,
) -> dict:
    """Paired confidence interval with all stations from a date kept together.

    Calendar dates are assigned to consecutive fixed-width blocks.  Blocks,
    rather than station-days, are resampled, so hundreds of gauges under the
    same weather system are never treated as hundreds of independent events.
    """
    diff = np.asarray(differences, dtype="float64")
    dates = np.asarray(valid_times).astype("datetime64[D]")
    if diff.ndim != 1 or dates.shape != diff.shape:
        raise ValueError("differences and valid_times must be equal one-dimensional arrays")
    good = np.isfinite(diff) & ~np.isnat(dates)
    diff, dates = diff[good], dates[good]
    if not diff.size:
        raise ValueError("no finite paired differences")
    if block_days < 1 or replicates < 1:
        raise ValueError("block_days and replicates must be positive")
    origin = dates.min()
    block_id = ((dates - origin).astype(int) // block_days).astype(int)
    blocks = np.unique(block_id)
    sums = np.array([diff[block_id == block].sum() for block in blocks])
    counts = np.array([(block_id == block).sum() for block in blocks])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(blocks), size=(replicates, len(blocks)))
    estimates = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(estimates, [alpha, 1.0 - alpha])
    return {
        "mean_difference": float(diff.mean()),
        "confidence": float(confidence),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "block_days": int(block_days),
        "blocks": int(len(blocks)),
        "replicates": int(replicates),
        "seed": int(seed),
        "cases": int(diff.size),
        "valid_dates": int(np.unique(dates).size),
    }


def ensemble_metric_cases(name: str, ensemble, observations) -> np.ndarray:
    """Dispatch a machine-readable metric name to its per-case values."""
    if name == "fair_crps":
        return fair_crps_cases(ensemble, observations)
    if name == "ensemble_mean_mae":
        ens, obs = _ensemble_and_truth(ensemble, observations)
        return np.abs(ens.mean(axis=1) - obs)
    for prefix, function in (
        ("twcrps_", threshold_weighted_crps_cases),
        ("fair_brier_", fair_brier_cases),
        ("brier_", brier_cases),
    ):
        if name.startswith(prefix):
            return function(ensemble, observations, float(name[len(prefix):]))
    raise ValueError(f"unknown ensemble metric {name!r}")


def summarize_ensemble(
    ensemble,
    observations,
    *,
    thresholds: list[float],
    event_probability: float = 0.5,
) -> dict:
    ens, obs = _ensemble_and_truth(ensemble, observations)
    mean = ens.mean(axis=1)
    errors = deterministic_error_cases(mean, obs)
    report = {
        "cases": int(obs.size),
        "members": int(ens.shape[1]),
        "observed_mean": float(obs.mean()),
        "ensemble_mean": float(mean.mean()),
        "ensemble_mean_bias": float(errors["error"].mean()),
        "ensemble_mean_mae": float(errors["absolute_error"].mean()),
        "ensemble_mean_rmse": float(np.sqrt(errors["squared_error"].mean())),
        "fair_crps": float(fair_crps_cases(ens, obs).mean()),
        "thresholds": {},
        "member_deterministic": [],
    }
    for member in range(ens.shape[1]):
        member_errors = deterministic_error_cases(ens[:, member], obs)
        report["member_deterministic"].append({
            "member": member,
            "bias": float(member_errors["error"].mean()),
            "mae": float(member_errors["absolute_error"].mean()),
            "rmse": float(np.sqrt(member_errors["squared_error"].mean())),
        })
    for threshold in thresholds:
        key = f"{threshold:g}"
        probability = exceedance_probability(ens, threshold)
        event = obs >= threshold
        report["thresholds"][key] = {
            "twcrps": float(threshold_weighted_crps_cases(ens, obs, threshold).mean()),
            "fair_brier": float(fair_brier_cases(ens, obs, threshold).mean()),
            "brier": float(((probability - event) ** 2).mean()),
            "reliability": reliability_table(probability, event),
            "probability_decision": {
                "cutoff": float(event_probability),
                **contingency(probability >= event_probability, event),
            },
            "ensemble_mean_decision": contingency(mean >= threshold, event),
        }
    return report


def summarize_deterministic(forecast, observations, *, thresholds: list[float]) -> dict:
    fc = np.asarray(forecast, dtype="float64")
    obs = np.asarray(observations, dtype="float64")
    errors = deterministic_error_cases(fc, obs)
    report = {
        "cases": int(obs.size),
        "observed_mean": float(obs.mean()),
        "forecast_mean": float(fc.mean()),
        "bias": float(errors["error"].mean()),
        "mae": float(errors["absolute_error"].mean()),
        "rmse": float(np.sqrt(errors["squared_error"].mean())),
        "thresholds": {},
    }
    for threshold in thresholds:
        event = obs >= threshold
        forecast_event = fc >= threshold
        report["thresholds"][f"{threshold:g}"] = {
            "brier": float(((forecast_event.astype(float) - event) ** 2).mean()),
            "contingency": contingency(forecast_event, event),
        }
    return report
