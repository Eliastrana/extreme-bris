#!/usr/bin/env python3
"""Fetch ERA5 truth for the times a forecast is valid at.

    ~/bris-data-env/bin/python scripts/fetch_verification.py \
        --date 2025-04-01T00:00:00 --leadtimes 10 \
        -o ~/bris-runs/truth/era5-truth-20250401T00Z.nc

This is deliberately NOT the era5-complete/N320 route the input datasets use.
Verification does not need the model's native grid or its 89 fields - it needs
a handful of surface fields at 0.25 degrees, which is what the forecast's
global file is already interpolated to. `reanalysis-era5-single-levels` serves
those from disk rather than tape, so this takes minutes where the input build
takes hours.

What it gets, and why:

  2m_temperature              the field with the clearest signal
  mean_sea_level_pressure     synoptic placement - is the low where it should be
  10m_u/v_component_of_wind   combined into wind speed, to match the forecast's
                              `wind_speed_10m`

Precipitation is optional (--precip) because it costs more: ERA5 accumulates it
hourly while the forecast accumulates over 6h steps, so the hours in between
have to be fetched and summed. Everything else is instantaneous and needs only
the verification times themselves.

NOTE ON WHAT THIS PROVES. The forecast being checked was initialised from ERA5,
not from the operational analysis Bris was trained on. Verifying it against ERA5
is therefore self-consistent - it asks "did the model carry this state forward
correctly", not "is Bris skilful". The second question needs `od`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

INSTANT = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]


def verification_times(t0: dt.datetime, leadtimes: int, step_h: int) -> list[dt.datetime]:
    """t0 included: the forecast's own initial state is the persistence baseline."""
    return [t0 + dt.timedelta(hours=step_h * i) for i in range(leadtimes + 1)]


def as_request(times: list[dt.datetime], variables: list[str], hourly: bool) -> dict:
    days = sorted({t.date() for t in times})
    if hourly:
        hours = [f"{h:02d}:00" for h in range(24)]
    else:
        hours = sorted({f"{t.hour:02d}:00" for t in times})
    return {
        "product_type": "reanalysis",
        "variable": variables,
        "year": sorted({f"{d.year:04d}" for d in days}),
        "month": sorted({f"{d.month:02d}" for d in days}),
        "day": sorted({f"{d.day:02d}" for d in days}),
        "time": hours,
    }


def retrieve(client, request: dict, target: Path) -> None:
    """CDS renamed `format` to `data_format` in the 2024 migration. Try both."""
    for key in ("data_format", "format"):
        req = dict(request)
        req[key] = "netcdf"
        try:
            client.retrieve("reanalysis-era5-single-levels", req, str(target))
            return
        except Exception as exc:                     # noqa: BLE001
            if key == "format":
                raise
            low = str(exc).lower()
            if "data_format" not in low and "invalid" not in low and "unknown" not in low:
                raise
            print(f"  (data_format rejected, retrying with format) {exc}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2025-04-01T00:00:00", help="initialisation time t0")
    ap.add_argument("--leadtimes", type=int, default=10, help="number of steps (default 10)")
    ap.add_argument("--step-hours", type=int, default=6)
    ap.add_argument("--precip", action="store_true",
                    help="also fetch hourly total_precipitation, for 6h sums")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    try:
        import cdsapi
    except ModuleNotFoundError:
        print("ERROR: cdsapi not installed. Use ~/bris-data-env/bin/python.", file=sys.stderr)
        return 1

    t0 = dt.datetime.fromisoformat(args.date)
    times = verification_times(t0, args.leadtimes, args.step_hours)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"t0            {t0:%Y-%m-%d %H:%M}")
    print(f"valid times   {len(times)}: {times[0]:%m-%d %HZ} .. {times[-1]:%m-%d %HZ}"
          f"  (+0h .. +{args.leadtimes * args.step_hours}h)")
    print(f"variables     {', '.join(INSTANT)}")
    print(f"target        {args.out}\n")

    client = cdsapi.Client()

    print("requesting instantaneous fields...")
    retrieve(client, as_request(times, INSTANT, hourly=False), args.out)
    print(f"  wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    if args.precip:
        pt = args.out.with_name(args.out.stem + "-precip.nc")
        print("\nrequesting hourly total_precipitation...")
        print("  hourly because ERA5 accumulates over 1h and the forecast over"
              f" {args.step_hours}h; the sum is taken at scoring time.")
        retrieve(client, as_request(times, ["total_precipitation"], hourly=True), pt)
        print(f"  wrote {pt} ({pt.stat().st_size / 1e6:.1f} MB)")

    print("\nNext:")
    print("  python scripts/verify_forecast.py \\")
    print(f"      --forecast <global_*.nc> --truth {args.out} -o results/verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
