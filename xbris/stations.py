"""Pair gauges with grid points, on whichever grid is being scored.

There are two grids in this project and they are not the same. The built
datasets carry the full MEPS grid, 949 by 1069. The forecast files carry what
the model actually produced, which is the same grid minus the fifty-point edge
trim the dataloader applies. Matching a station on one and reading a value on
the other is an index error that produces a number rather than an exception.

So the grid is always taken from the thing being scored, and the matching lives
here rather than in either caller.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EARTH_R = 6371.0088          # km


def great_circle_km(lat1, lon1, lat2, lon2):
    """Haversine, broadcast over whatever shapes come in."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = p2 - p1
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_grid(path: Path, nx: int | None = None, ny: int | None = None):
    """Flat latitudes and longitudes, plus the 2-D shape, from a zarr or a NetCDF.

    Returns (lat, lon, (ny, nx)). The shape matters because a forecast file
    indexes by row and column while a dataset indexes by a single flat point.
    """
    path = Path(path)
    if path.suffix == ".zarr" or (path.is_dir() and (path / ".zattrs").exists()):
        import zarr

        z = zarr.open(str(path), mode="r")
        lat = np.asarray(z["latitudes"], dtype="float64")
        lon = np.asarray(z["longitudes"], dtype="float64")
        if nx and ny and lat.size == nx * ny:
            return lat, lon, (ny, nx)
        return lat, lon, (lat.size, 1)

    import xarray as xr

    with xr.open_dataset(path) as ds:
        lat2d = np.asarray(ds["latitude"].values, dtype="float64")
        lon2d = np.asarray(ds["longitude"].values, dtype="float64")
    if lat2d.ndim != 2:
        raise ValueError(f"{path.name} has {lat2d.ndim}-D latitude; expected 2-D")
    return lat2d.ravel(), lon2d.ravel(), lat2d.shape


def nearest(slat, slon, glat, glon, block: int = 16):
    """Index of and distance to the closest grid point, per station.

    Blocked over stations rather than built as one array: a million grid points
    against several hundred stations is billions of pairs, and everything that
    calls this is meant to run on a login node.
    """
    idx = np.empty(slat.size, dtype="int64")
    dist = np.empty(slat.size, dtype="float64")
    for i in range(0, slat.size, block):
        j = slice(i, min(i + block, slat.size))
        d = great_circle_km(slat[j, None], slon[j, None], glat[None, :], glon[None, :])
        idx[j] = d.argmin(axis=1)
        dist[j] = d.min(axis=1)
    return idx, dist


def load_observations(path: Path):
    """Station ids, coordinates, hourly values and their times."""
    with np.load(path, allow_pickle=False) as f:
        return {
            "stations": f["stations"],
            "lat": f["lat"].astype("float64"),
            "lon": f["lon"].astype("float64"),
            "values": f["values"],
            "times": np.array([np.datetime64(t.replace("Z", ""))
                               for t in f["times"]], dtype="datetime64[s]"),
            "element": str(f["element"]),
            "unit": str(f["unit"]),
        }


def accumulate(values, times, ends, hours: int):
    """Sum each station's hourly series over the `hours` ending at each of `ends`.

    Returns an array shaped (stations, len(ends)), NaN where the window is not
    complete. Incomplete is not the same as zero: a gauge that reported four of
    six hours during heavy rain would otherwise look drier than it was, and the
    states this project cares about are exactly the wet ones.
    """
    out = np.full((values.shape[0], len(ends)), np.nan, dtype="float64")
    step = np.timedelta64(1, "h")
    for k, end in enumerate(ends):
        want = np.array([end - step * (hours - 1 - i) for i in range(hours)],
                        dtype="datetime64[s]")
        cols = np.searchsorted(times, want)
        cols = np.clip(cols, 0, len(times) - 1)
        if not np.array_equal(times[cols], want):
            continue
        block = values[:, cols]
        complete = np.isfinite(block).all(axis=1)
        out[complete, k] = block[complete].sum(axis=1)
    return out
