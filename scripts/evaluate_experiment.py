#!/usr/bin/env python3
"""Joint, result-blind evaluation of the three Bris arms and MEPS.

Example (quote globs so this script, rather than the shell, expands them):

    scripts/evaluate_experiment.py \
      --model 'baseline=~/bris-runs/evaluation/baseline/nordic_*.nc' \
      --model 'control=~/bris-runs/evaluation/control/nordic_*.nc' \
      --model 'tail=~/bris-runs/evaluation/tail/nordic_*.nc' \
      --meps ~/bris-runs/meps/precipitation_daily.npz \
      --observations ~/bris-runs/observations \
      --out-dir ~/bris-runs/evaluation-score

The machine-readable plan fixes the period, leads, thresholds, primary metric,
comparison, bootstrap, and common-case rule before results are read.  Command
line arguments supply paths only.  Every included and excluded cycle/lead is
written to case-ledger.csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("xarray", "numpy")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.evaluation import (  # noqa: E402
    daily_sums,
    deaccumulate,
    ensemble_metric_cases,
    paired_block_bootstrap,
    strict_common_mask,
    summarize_deterministic,
    summarize_ensemble,
)
from xbris.stations import load_observations, nearest  # noqa: E402


PRECIPITATION = "precipitation_amount"
STAMP = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<hour>\d{2})Z")
COMPACT_STAMP = re.compile(r"(?P<date>\d{8})T(?P<hour>\d{2})Z")


def cycle_key(value) -> str:
    return str(np.datetime64(value, "s"))


def cycle_from_name(path: Path) -> str:
    match = STAMP.search(path.name)
    if match:
        return f"{match.group('date')}T{match.group('hour')}:00:00"
    match = COMPACT_STAMP.search(path.name)
    if match:
        date = dt.datetime.strptime(match.group("date"), "%Y%m%d").date()
        return f"{date.isoformat()}T{match.group('hour')}:00:00"
    raise ValueError(
        f"cannot find a YYYY-MM-DDTHHZ cycle stamp in {path.name}; "
        "rename the file or add its naming convention to cycle_from_name"
    )


def files_for_spec(spec: str) -> list[Path]:
    expanded = str(Path(spec).expanduser())
    path = Path(expanded)
    if path.is_dir():
        files = sorted(path.glob("nordic_*.nc"))
    else:
        files = [Path(item) for item in sorted(glob.glob(expanded))]
    if not files:
        raise SystemExit(f"no NetCDF files matched {spec!r}")
    return files


def index_model_files(spec: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in files_for_spec(spec):
        key = cycle_from_name(path)
        if key in index:
            raise SystemExit(f"two forecast files claim cycle {key}: {index[key]} and {path}")
        index[key] = path
    return index


def expected_cycles(start: str, end: str, hour: int) -> list[str]:
    first = dt.date.fromisoformat(start)
    last = dt.date.fromisoformat(end)
    if last < first:
        raise ValueError("evaluation period ends before it starts")
    out = []
    day = first
    while day <= last:
        out.append(f"{day.isoformat()}T{hour:02d}:00:00")
        day += dt.timedelta(days=1)
    return out


def member_dimension(da: xr.DataArray) -> str | None:
    names = ("ensemble", "member", "number", "realization")
    for dim in da.dims:
        lower = dim.lower()
        if any(name in lower for name in names):
            return dim
    return None


class BrisReader:
    """Read only gauge cells from one model's gridded forecast files."""

    def __init__(self, obs: dict, max_distance_km: float, accumulation: str) -> None:
        self.obs = obs
        self.max_distance_km = max_distance_km
        self.accumulation = accumulation
        self._row: np.ndarray | None = None
        self._col: np.ndarray | None = None
        self._keep: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None

    def _match_grid(self, ds: xr.Dataset, path: Path) -> None:
        latitude = np.asarray(ds["latitude"].values, dtype="float64")
        longitude = np.asarray(ds["longitude"].values, dtype="float64")
        if latitude.ndim != 2 or longitude.shape != latitude.shape:
            raise ValueError(f"{path.name}: latitude/longitude are not matching 2-D arrays")
        idx, distance = nearest(
            self.obs["lat"], self.obs["lon"], latitude.ravel(), longitude.ravel()
        )
        self._keep = distance <= self.max_distance_km
        self._row, self._col = np.divmod(idx[self._keep], latitude.shape[1])
        self._shape = latitude.shape

    def read(self, path: Path, end_leads: list[int]) -> tuple[np.ndarray, dict]:
        with xr.open_dataset(path) as ds:
            if PRECIPITATION not in ds:
                raise ValueError(f"{path.name}: missing {PRECIPITATION}")
            if self._keep is None:
                self._match_grid(ds, path)
            else:
                shape = tuple(int(v) for v in ds["latitude"].shape)
                if shape != self._shape:
                    raise ValueError(f"{path.name}: grid changed from {self._shape} to {shape}")

            da = ds[PRECIPITATION]
            member = member_dimension(da)
            protected = {"time", "y", "x"} | ({member} if member else set())
            for dim in list(da.dims):
                if dim in protected:
                    continue
                if da.sizes[dim] != 1:
                    raise ValueError(
                        f"{path.name}: unsupported non-singleton dimension {dim}={da.sizes[dim]}"
                    )
                da = da.isel({dim: 0}, drop=True)
            if not {"time", "y", "x"}.issubset(da.dims):
                raise ValueError(f"{path.name}: expected time/y/x dimensions, got {da.dims}")
            if member is None:
                member = "ensemble_member"
                da = da.expand_dims({member: [0]}, axis=1)

            # Read the whole field once and pick the gauge cells in memory. The
            # files are uncompressed, and netCDF4 serves a scattered-point
            # selection by seeking through the file point by point: 8.8 s for
            # 20 gauges, against 1.1 s for the whole field and all 715, with
            # identical values. The scattered read put a full evaluation at
            # about a day.
            field = np.asarray(da.transpose("time", member, "y", "x").values)
            steps = field[:, :, self._row, self._col].astype("float64")
            times = np.asarray(ds["time"].values).astype("datetime64[s]")
            attrs = {
                key: str(value)
                for key, value in ds.attrs.items()
                if "seed" in key.lower() or "random" in key.lower()
            }

        if times.size < 2:
            raise ValueError(f"{path.name}: fewer than two time steps")
        leads = ((times - times[0]) / np.timedelta64(1, "h")).astype(int)
        steps, convention = deaccumulate(steps, self.accumulation)
        sums = daily_sums(steps, leads, end_leads)
        full = np.full(
            (len(end_leads), sums.shape[1], len(self.obs["stations"])),
            np.nan,
            dtype="float64",
        )
        full[:, :, self._keep] = sums
        return full, {
            "members": int(sums.shape[1]),
            "accumulation": convention,
            "seed_attributes": attrs,
        }


class MepsReader:
    """Daily sums from the station-point MEPS archive file."""

    def __init__(self, path: Path, obs: dict, max_distance_km: float) -> None:
        self.path = path
        with np.load(path, allow_pickle=False) as data:
            self.values = np.asarray(data["values"], dtype="float64")
            stations = data["stations"].astype(str)
            self.cycles = np.array(
                [np.datetime64(value) for value in data["cycles"]], dtype="datetime64[s]"
            )
            self.leads = np.asarray(data["leads"], dtype=int)
            distance = (
                np.asarray(data["distance_km"], dtype="float64")
                if "distance_km" in data.files
                else np.zeros(stations.size)
            )
            self.accumulation = str(data["accumulation"]) if "accumulation" in data.files else ""
        where = {station: i for i, station in enumerate(stations)}
        self.order = np.array([where.get(str(station), -1) for station in obs["stations"]])
        self.keep = self.order >= 0
        positions = np.flatnonzero(self.keep)
        matched = self.order[positions]
        self.keep[positions] = distance[matched] <= max_distance_km
        self.cycle_index = {cycle_key(value): i for i, value in enumerate(self.cycles)}

    def has_cycle(self, cycle: str) -> bool:
        return cycle in self.cycle_index

    def read(self, cycle: str, end_leads: list[int]) -> np.ndarray:
        ci = self.cycle_index[cycle]
        stations = np.flatnonzero(self.keep)
        source = self.order[stations]
        steps = self.values[source, ci, :].T[:, None, :]  # time, member=1, station
        sums = daily_sums(steps, self.leads, end_leads)
        full = np.full((len(end_leads), len(self.keep)), np.nan, dtype="float64")
        full[:, stations] = sums[:, 0, :]
        return full


def load_screened_observations(path: Path, max_quality: int) -> tuple[dict, list[str]]:
    obs = load_observations(path / "precipitation_daily.npz", max_quality=max_quality)
    screen = path / "precipitation_daily_check.json"
    screened: list[str] = []
    if screen.exists():
        flagged = json.loads(screen.read_text())["flagged"]
        drop = np.isin(obs["stations"], list(flagged))
        obs["values"] = np.array(obs["values"], dtype="float64", copy=True)
        obs["values"][drop] = np.nan
        screened = sorted(str(value) for value in obs["stations"][drop])
    return obs, screened


def observation_for_time(obs: dict, valid_time: np.datetime64) -> np.ndarray | None:
    position = np.flatnonzero(obs["times"] == valid_time)
    if not position.size:
        return None
    return np.asarray(obs["values"][:, int(position[0])], dtype="float64")


def metric_names(thresholds: list[float]) -> list[str]:
    out = ["fair_crps", "ensemble_mean_mae"]
    for threshold in thresholds:
        out += [f"twcrps_{threshold:g}", f"fair_brier_{threshold:g}", f"brier_{threshold:g}"]
    return out


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_ledger(path: Path, rows: list[dict], labels: list[str]) -> None:
    fields = ["cycle", "valid_end", "lead_hours", "included", "common_cases",
              "observations_present", "meps_present"]
    for label in labels:
        fields += [f"{label}_present", f"{label}_members", f"{label}_accumulation"]
    fields.append("reasons")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_summary(report: dict) -> str:
    primary = report.get("primary_result")
    lines = [
        "# Bris extreme-weather evaluation",
        "",
        f"Plan SHA-256: `{report['plan_sha256']}`",
        "",
        "All scores use the strict common intersection recorded in `case-ledger.csv`.",
        "Negative paired differences favour the first model named in a comparison.",
        "",
    ]
    for lead, scores in report["by_lead_hours"].items():
        lines += [f"## +{lead} h", "", f"Common station-days: {scores['cases']:,}", ""]
        lines.append("| Model | fair CRPS | twCRPS 20 | twCRPS 50 | mean MAE |")
        lines.append("|---|---:|---:|---:|---:|")
        for label, model in scores["models"].items():
            lines.append(
                f"| {label} | {model['fair_crps']:.4f} | "
                f"{model['thresholds']['20']['twcrps']:.4f} | "
                f"{model['thresholds']['50']['twcrps']:.4f} | "
                f"{model['ensemble_mean_mae']:.4f} |"
            )
        meps = scores["meps"]
        lines.append(f"| MEPS (deterministic) | — | — | — | {meps['mae']:.4f} |")
        lines.append("")
    if primary:
        lines += [
            "## Pre-registered primary comparison",
            "",
            f"{primary['comparison']} at +{primary['lead_hours']} h, {primary['metric']}.",
            f"Mean difference: {primary['mean_difference']:.6f} "
            f"({primary['confidence']:.0%} block-bootstrap CI "
            f"{primary['ci_lower']:.6f} to {primary['ci_upper']:.6f}).",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=PATH_OR_GLOB")
    parser.add_argument("--meps", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=root / "evaluation" / "evaluation_plan.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    labels = list(plan["required_models"])
    specs = {}
    for item in args.model:
        if "=" not in item:
            raise SystemExit(f"--model must be LABEL=PATH_OR_GLOB, got {item!r}")
        label, spec = item.split("=", 1)
        if label in specs:
            raise SystemExit(f"model {label!r} was supplied twice")
        specs[label] = spec
    if set(specs) != set(labels):
        raise SystemExit(f"plan requires models {labels}; received {sorted(specs)}")

    observation_plan = plan["observations"]
    obs, screened = load_screened_observations(args.observations.expanduser(), observation_plan["max_quality"])
    thresholds = [float(value) for value in plan["thresholds_mm"]]
    leads = [int(value) for value in plan["leads_hours"]]
    accumulation = plan["forecast_accumulation"]
    max_distance = float(observation_plan["max_distance_km"])

    indexes = {label: index_model_files(specs[label]) for label in labels}
    readers = {label: BrisReader(obs, max_distance, accumulation) for label in labels}
    meps = MepsReader(args.meps.expanduser(), obs, max_distance)
    period = plan["period"]
    cycles = expected_cycles(period["start"], period["end"], int(period["cycle_hour_utc"]))
    if len(cycles) != int(period["expected_cycles"]):
        raise SystemExit(
            f"plan says {period['expected_cycles']} cycles but its dates define {len(cycles)}"
        )
    if not plan.get("strict_common_cases"):
        raise SystemExit("this evaluator requires strict_common_cases=true")
    print(f"=== frozen plan {args.plan} ({hashlib.sha256(plan_bytes).hexdigest()[:12]})")
    print(f"=== {len(cycles)} expected cycles, leads {leads}, models {labels}, strict common cases\n")

    ledger: list[dict] = []
    collected = {
        lead: {"observations": [], "valid_times": [], "station_indices": [],
               "models": {label: [] for label in labels}, "meps": []}
        for lead in leads
    }
    provenance = {label: {"member_counts": set(), "accumulation": set(), "seed_attributes": []}
                  for label in labels}

    for number, cycle in enumerate(cycles, 1):
        present = {label: cycle in indexes[label] for label in labels}
        meps_present = meps.has_cycle(cycle)
        missing = [label for label, value in present.items() if not value]
        if not meps_present:
            missing.append("meps")
        if missing:
            for lead in leads:
                valid = np.datetime64(cycle) + np.timedelta64(lead, "h")
                row = {
                    "cycle": cycle, "valid_end": str(valid), "lead_hours": lead,
                    "included": False, "common_cases": 0,
                    "observations_present": observation_for_time(obs, valid) is not None,
                    "meps_present": meps_present,
                    "reasons": ";".join(f"missing_{name}" for name in missing),
                }
                for label in labels:
                    row[f"{label}_present"] = present[label]
                ledger.append(row)
            continue

        try:
            model_daily = {}
            model_meta = {}
            for label in labels:
                model_daily[label], model_meta[label] = readers[label].read(indexes[label][cycle], leads)
                expected_members = int(plan["ensemble_members"])
                if model_meta[label]["members"] != expected_members:
                    raise ValueError(
                        f"{label} has {model_meta[label]['members']} members, "
                        f"plan requires {expected_members}"
                    )
                provenance[label]["member_counts"].add(model_meta[label]["members"])
                provenance[label]["accumulation"].add(model_meta[label]["accumulation"])
                if model_meta[label]["seed_attributes"]:
                    provenance[label]["seed_attributes"].append({
                        "cycle": cycle, **model_meta[label]["seed_attributes"]
                    })
            meps_daily = meps.read(cycle, leads)
        except Exception as exc:  # noqa: BLE001 - failure belongs in the ledger
            reason = f"read_error:{type(exc).__name__}:{str(exc).replace(';', ',')[:180]}"
            print(f"  {cycle} {reason}", file=sys.stderr)
            for lead in leads:
                row = {
                    "cycle": cycle,
                    "valid_end": str(np.datetime64(cycle) + np.timedelta64(lead, "h")),
                    "lead_hours": lead, "included": False, "common_cases": 0,
                    "observations_present": False, "meps_present": True, "reasons": reason,
                }
                for label in labels:
                    row[f"{label}_present"] = True
                ledger.append(row)
            continue

        for li, lead in enumerate(leads):
            valid = np.datetime64(cycle) + np.timedelta64(lead, "h")
            truth = observation_for_time(obs, valid)
            common = (
                np.zeros(len(obs["stations"]), dtype=bool)
                if truth is None
                else strict_common_mask(
                    truth,
                    meps_daily[li],
                    [model_daily[label][li] for label in labels],
                )
            )
            count = int(common.sum())
            reasons = []
            if truth is None:
                reasons.append("observation_time_absent")
            elif not np.isfinite(truth).any():
                reasons.append("no_finite_observations")
            if count == 0 and not reasons:
                reasons.append("no_strict_common_station_days")
            row = {
                "cycle": cycle,
                "valid_end": str(valid),
                "lead_hours": lead,
                "included": count > 0,
                "common_cases": count,
                "observations_present": truth is not None,
                "meps_present": True,
                "reasons": ";".join(reasons),
            }
            for label in labels:
                row[f"{label}_present"] = True
                row[f"{label}_members"] = model_meta[label]["members"]
                row[f"{label}_accumulation"] = model_meta[label]["accumulation"]
            ledger.append(row)
            if not count:
                continue
            slot = collected[lead]
            slot["observations"].append(truth[common])
            slot["valid_times"].append(np.full(count, valid, dtype="datetime64[s]"))
            slot["station_indices"].append(np.flatnonzero(common))
            slot["meps"].append(meps_daily[li, common])
            for label in labels:
                slot["models"][label].append(model_daily[label][li][:, common].T)

        if number % 10 == 0 or number == len(cycles):
            print(f"  {number}/{len(cycles)} cycles read")

    out = args.out_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_ledger(out / "case-ledger.csv", ledger, labels)

    report = {
        "schema_version": 1,
        "plan": plan,
        "plan_path": str(args.plan),
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "paths": {"models": specs, "meps": str(args.meps), "observations": str(args.observations)},
        "screened_out_stations": screened,
        "quality_values_dropped": int(obs["quality_dropped"]),
        "strict_common_sources": [*labels, "meps", "observations"],
        "expected_cycles": len(cycles),
        "ledger_rows": len(ledger),
        "provenance": {},
        "by_lead_hours": {},
        "comparisons": {},
    }
    for label in labels:
        report["provenance"][label] = {
            "member_counts": sorted(provenance[label]["member_counts"]),
            "accumulation": sorted(provenance[label]["accumulation"]),
            "seed_attributes": provenance[label]["seed_attributes"],
            "seed_metadata_found": bool(provenance[label]["seed_attributes"]),
        }

    arrays = {}
    for lead in leads:
        slot = collected[lead]
        if not slot["observations"]:
            raise SystemExit(f"no common cases at +{lead} h; inspect {out / 'case-ledger.csv'}")
        obs_values = np.concatenate(slot["observations"])
        valid_times = np.concatenate(slot["valid_times"])
        station_indices = np.concatenate(slot["station_indices"])
        models = {label: np.concatenate(slot["models"][label], axis=0) for label in labels}
        meps_values = np.concatenate(slot["meps"])
        arrays[lead] = {"observations": obs_values, "valid_times": valid_times,
                        "station_indices": station_indices, "models": models, "meps": meps_values}
        report["by_lead_hours"][str(lead)] = {
            "cases": int(obs_values.size),
            "valid_dates": int(np.unique(valid_times.astype("datetime64[D]")).size),
            "stations": int(np.unique(station_indices).size),
            "models": {
                label: summarize_ensemble(
                    models[label], obs_values, thresholds=thresholds,
                    event_probability=float(plan["event_probability_cutoff"]),
                )
                for label in labels
            },
            "meps": summarize_deterministic(meps_values, obs_values, thresholds=thresholds),
        }

    bootstrap = plan["bootstrap"]
    metrics = metric_names(thresholds)
    for first, second in plan["comparisons"]:
        comparison = f"{first}_minus_{second}"
        report["comparisons"][comparison] = {}
        for lead in leads:
            data = arrays[lead]
            lead_report = {}
            for metric in metrics:
                first_cases = ensemble_metric_cases(
                    metric, data["models"][first], data["observations"]
                )
                second_cases = ensemble_metric_cases(
                    metric, data["models"][second], data["observations"]
                )
                lead_report[metric] = paired_block_bootstrap(
                    first_cases - second_cases,
                    data["valid_times"],
                    block_days=int(bootstrap["block_days"]),
                    replicates=int(bootstrap["replicates"]),
                    seed=int(bootstrap["seed"]),
                    confidence=float(bootstrap["confidence"]),
                )
            report["comparisons"][comparison][str(lead)] = lead_report

    primary = plan["primary"]
    comparison = f"{primary['first']}_minus_{primary['second']}"
    metric = primary["metric"]
    if primary.get("threshold_mm") is not None:
        metric = f"{metric}_{float(primary['threshold_mm']):g}"
    primary_result = dict(report["comparisons"][comparison][str(primary["lead_hours"])][metric])
    primary_result.update({
        "comparison": comparison,
        "lead_hours": int(primary["lead_hours"]),
        "metric": metric,
        "lower_is_better": True,
    })
    report["primary_result"] = primary_result

    json_dump(out / "report.json", report)
    (out / "summary.md").write_text(render_summary(report))
    print(f"\nwrote {out / 'case-ledger.csv'}")
    print(f"wrote {out / 'report.json'}")
    print(f"wrote {out / 'summary.md'}")
    if not all(report["provenance"][label]["seed_metadata_found"] for label in labels):
        print("\nWARNING: one or more NetCDF sets expose no seed metadata. "
              "Use the inference job logs to verify common member seeds across models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
