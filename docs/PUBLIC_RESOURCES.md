# What is already public

Audited 2026-08-26, before asking MET for anything. Several things I had been
treating as unavailable are public, and one route I did not know about turns out
to be the intended one.

## Public — do not ask for these

### Code

| Repo | What |
|---|---|
| `metno/bris-inference` | the inference package (already installed) |
| `metno/bris-configs` | training + inference configs, **but not for CRPS-FFT** — only AIFS-CRPS, alpaca, boiling-blizzard |
| `metno/anemoi-regional-tutorial` | **MET's own guide to building regional datasets** |
| `metno/bris-anemoi-demo` | end-to-end demo: training + inference configs, env scripts, jobscripts |
| `metno/bris-fiab` | Bris in Forecast-in-a-Box, including moving the domain |
| `metno/anemoi-datasets` | MET fork; branch `feature/trimedge` — the `trim_edge` support our cutout uses |
| `evenmn/anemoi-core` | branch `feat/crps-fft-loss`, required for training |
| `ecmwf/anemoi-*` | datasets, models, training, graphs, inference, transform, utils |

### Checkpoints on Hugging Face (all Apache-2.0)

| Repo | Contents |
|---|---|
| `met-no/bris-forecaster` | CRPS-FFT inference + training ckpt, configs |
| `met-no/bris-forecaster-pretrained` | configs only — **no checkpoint** |
| `met-no/Bris-HourGlass` | 6h -> 1h temporal downscaler, 4 checkpoints |
| `met-no/bris_cloudy-skies` | **five further checkpoints**, 2025-02 through 2026-01, deterministic, ensemble, and a forecaster/interpolator pair, each with pinned dependencies |

`bris_cloudy-skies` was not in the project notes and is worth knowing about: it
carries a matched forecaster + interpolator pair for hourly output.

## Not public

No anemoi-format training dataset is downloadable anywhere:

- no `met-no` datasets on Hugging Face, and no weather anemoi datasets under any
  other account
- ECMWF's `ml-datasets` object store returns 403
- thredds.met.no publishes MEPS as GRIB/NetCDF, not anemoi zarr
- neither paper carries a data availability statement

## The route I did not know about

MET's own tutorial says it plainly:

> "To see what datasets are already available, checkout
> https://anemoi.ecmwf.int/datasets (requires ECMWF login credentials). The site
> provides download links to files in S3 buckets, and paths to where files are
> located on LUMI and Leonardo."

Verified: the site exists, `/datasets` redirects to a login (302) and the API
returns 401. So there **is** a catalogue of ready-made anemoi datasets — it is
gated behind ECMWF credentials rather than unpublished.

Norway is an ECMWF member state, and this work is UiO/Simula in collaboration
with MET, so credentials are plausibly obtainable.

## What this changes about the ask

Not "please send me your training data". Instead:

1. **Are the Bris stretched-grid datasets in the anemoi catalogue, and how do I
   get access?** Far smaller than a data transfer, and it is the route MET
   documents for exactly this.
2. **Is ERA5 (`ea`) an acceptable substitute for the operational analysis
   (`od`) for initial conditions?** Still the one question no file answers.

## What I should have been using already

- `anemoi-regional-tutorial` — MET's guide to `anemoi-datasets create` for a
  regional domain, by the Bris authors. Directly relevant to task 5, and the
  recipe format there is the reference `make_era5_recipe.py` should match.
- `metno/anemoi-datasets@feature/trimedge` — `trim_edge` is in our cutout
  config, so the fork may be required rather than optional.
- `bris-anemoi-demo` — a working end-to-end example to check our Slurm and env
  scripts against.
