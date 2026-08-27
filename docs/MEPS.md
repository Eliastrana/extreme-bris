# Building the MEPS side of the cutout

Surveyed 2026-08-26 against the public MEPS archive on thredds.met.no. Better
news than expected: the public archive is on the same grid MET trained on, so no
domain fitting is needed.

## The archive

    https://thredds.met.no/thredds/catalog/meps25epsarchive/YYYY/MM/DD/catalog.xml

Files are split by level type, one set per initialisation (00/06/12/18 UTC):

| File | Contents |
|---|---|
| `meps_det_sfc_YYYYMMDDThhZ.ncml` | surface — 114 variables |
| `meps_det_pl_YYYYMMDDThhZ.ncml` | pressure levels |
| `meps_det_ml_*`, `meps_det_hl_*` | model / height levels — not needed |

`det` is the control member, which is what MET uses. Licence is NLOD / CC BY 4.0,
no authentication.

For an initialisation at `2025-04-01T00` with `multistep_input: 2`, four files:
`sfc` and `pl` at both `20250331T18Z` and `20250401T00Z`.

## Grid — matches exactly

    x = 949, y = 1069        projection_lambert

This is precisely the pre-`trim_edge` extent derived from the checkpoint graph.
**The public archive is MET's training grid**, so there is nothing to fit or
approximate. It also arrives as a structured 2D grid, which is what `TrimEdge`
requires (it rejects anything whose `field_shape` is not two-dimensional).

## Pressure levels

    50, 100, 150, 200, 250, 300, 400, 500, 700, 800, 850, 925, 1000

Thirteen levels. All twelve the model needs are present; drop 800. **There is no
600 hPa** — exactly as Ingstad et al. state, and independent confirmation that
the drop list in the training config is an archive limitation.

## Variable mapping

MEPS uses CF standard names, ECMWF short names. Sixteen of seventeen map
directly:

| ECMWF | MEPS |
|---|---|
| `2t` | `air_temperature_2m` |
| `skt` | `air_temperature_0m` |
| `msl` | `air_pressure_at_sea_level` |
| `sp` | `surface_air_pressure` |
| `lsm` | `land_area_fraction` |
| `z` | `surface_geopotential` |
| `tcc` | `cloud_area_fraction` |
| `hcc` | `high_type_cloud_area_fraction` |
| `mcc` | `medium_type_cloud_area_fraction` |
| `lcc` | `low_type_cloud_area_fraction` |
| `tcw` | `lwe_thickness_of_atmosphere_mass_content_of_water_vapor` |
| `tp` | `precipitation_amount_acc` |
| `ssrd` | `integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time` |
| `strd` | `integral_of_surface_downwelling_longwave_flux_in_air_wrt_time` |
| `10u` | `x_wind_10m` — **see rotation below** |
| `10v` | `y_wind_10m` — **see rotation below** |
| `2d` | **absent** — derive from `relative_humidity_2m` or `specific_humidity_2m` with `air_temperature_2m` |

Upper-air `q t u v w z` are in the `pl` file under the corresponding CF names.

## Three ways to get this silently wrong

Each produces output that runs and looks plausible.

**Winds are grid-relative.** `x_wind_10m` and `y_wind_10m` are aligned with the
Lambert projection axes; ECMWF's `10u`/`10v` are earth-relative. They must be
rotated. MET's own tutorial lists "rotating winds in LAM models" as one of the
filters anemoi-datasets provides, which is exactly this. Skipping it gives wind
fields that are wrong by an angle growing with distance from the projection's
reference longitude — plausible everywhere, correct nowhere.

**Radiation must be the accumulated form.** MEPS carries both
`surface_downwelling_shortwave_flux_in_air` (instantaneous) and
`integral_of_..._wrt_time` (accumulated). ECMWF `ssrd`/`strd` are accumulated.
Picking the instantaneous field gives values of the wrong magnitude and the
wrong physical meaning.

**Lead time zero.** Take the analysis, lead time 0, for initial conditions.
Accumulated fields — `tp`, `ssrd`, `strd` — are zero there by construction. That
is harmless here: all three are **diagnostic**, so they are not model inputs
(`input.full` has 95 entries and excludes them). They still need to exist as
columns in the dataset.

## The .ncml files are aggregations, not data

Worth understanding before choosing an access route. Each
`meps_det_sfc_*.ncml` is an **8 KB NcML descriptor**, a `joinExisting`
aggregation over 67 per-leadtime files living on MET's internal Lustre:

    <netcdf location="/lustre/arkivB/.../meps_sfc_00_20250331T18Z.nc"/>
    <netcdf location="/lustre/arkivB/.../meps_sfc_01_20250331T18Z.nc"/>
    ...

Those paths are internal, and the individual files are not exposed in the
catalogue. So:

- `fileServer` on the `.ncml` returns the descriptor, not the data
- **`dodsC` is the only route**, because only THREDDS resolves the aggregation
- downloading first is not an available alternative

## Reading it: pydap, not netCDF4

The netCDF4 C library fails on these URLs with
`OSError: [Errno -68] NetCDF: I/O failure`, even with the system CA configured
in `~/.dodsrc` and with `curl` returning 200 for the same URL.

The fix is to use **pydap** as the Xarray engine. It is pure Python, goes
through `requests`, and therefore honours `REQUESTS_CA_BUNDLE` like the rest of
the stack — sidestepping the C library's DAP client entirely:

    opendap:
      url: ...
      options:
        engine: pydap

It needs installing separately:

    uv pip install --python ~/bris-data-env pydap

Level selection is supported on the source directly. From the anemoi-datasets
documentation: for Xarray-based sources `param` and `variable` are synonyms, as
are `level` and `levelist`. So dropping 800 hPa is a `level:` list, not a
separate filter step.

## Approach

`metno/anemoi-regional-tutorial` is MET's own walkthrough, by the Bris authors,
and covers extending anemoi-datasets with new sources and filters. Note that
MET do not build from thredds themselves — `bris-anemoi-demo` points at
pre-built zarr on Leonardo — so there is no published MEPS recipe to copy.
