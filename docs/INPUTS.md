# Minimum input for a single Bris forecast

What the CRPS-FFT checkpoint needs in order to produce one forecast, and how we
intend to assemble it on eX3 without MARS access.

Everything below is read off `configs/config_inference.yaml` and
`configs/config_training.yaml` in `met-no/bris-forecaster`. Where the configs
are silent, that is stated rather than guessed — the authoritative source is the
checkpoint itself, which is what `scripts/inspect_checkpoint.py` is for. **Run
that before building any data.**

## 1. Two datasets, combined as a cutout

Bris is a stretched-grid model. The input is a `cutout` of two zarr datasets:

| Role | Config key | Grid | Notes |
|---|---|---|---|
| Inner / LAM | `dataset_lam` | MEPS 2.5 km | `trim_edge: 50` |
| Outer / global | `dataset_global` | N320 (~31 km) | `select: ${selected_vars}` |

`trim_edge: 50` discards 50 cells from the MEPS boundary — roughly 125 km, the
relaxation zone where the LAM is nudged toward its host model and is not
physically trustworthy. This is not optional; it is part of how the model was
trained.

**Verified in anemoi-datasets 0.5.24** (the version the lockfile pins), reading
`TrimEdge` in `data/masked.py`:

- `trim_edge` is present upstream — MET's `feature/trimedge` fork has been
  merged, so the fork is not required after all
- a scalar is accepted and expanded to all four edges, so `trim_edge: 50` is
  valid
- **`field_shape` must be two-dimensional**, or it raises
  `TrimEdge only works on regular grids`

That last point is a hard requirement on how the MEPS dataset is built: it has
to carry a 2D field shape of 949 x 1069, not a flattened list of points. A
dataset that is otherwise correct will fail here if the grid is stored as an
unstructured sequence.

Both are combined with `min_distance_km: 0` and `adjust: all`.

## 2. Variables

`selected_vars` in the inference config lists **98** names. They split into:

- **89 that must exist in the input data**
  - 17 surface / single-level: `10u 10v 2d 2t hcc lcc mcc msl skt sp ssrd strd tcc tcw tp z`
    plus `lsm`
  - 6 upper-air fields on 12 pressure levels = 72:
    `q t u v w z` at `50 100 150 200 250 300 400 500 700 850 925 1000` hPa
- **9 computed by anemoi-datasets at load time**, not stored:
  `cos_julian_day sin_julian_day cos_latitude sin_latitude cos_longitude
  sin_longitude cos_local_time sin_local_time insolation`

Note there is no 600 hPa level: the training config explicitly drops
`u_600 v_600 w_600 q_600 z_600 t_600`, along with `sdor slor cp`.

The reason is archival, not a modelling choice. Ingstad et al. (HourGlass,
arXiv:2607.11457) state it directly, explaining why they could not fine-tune
Bris-HourGlass from the global model: *"MEPS does not have the 600 hPa pressure
level variables."* Any MEPS dataset built here will have the same gap by
construction, so the drop list is not something to try to fill in.

`z` appears twice with different meanings — as a surface field (orography
geopotential) and as a pressure-level field. Keep them distinct when building.

### Confirmed from the checkpoint (2026-08-25)

Read out of `bris-crpsfft_inference.ckpt` on eX3, not inferred from YAML:

| group | count | contents |
|---|---|---|
| prognostic | 84 | evolved and fed back each step |
| forcing | 11 | 9 computed + `lsm`, `z` |
| diagnostic | 3 | `tp`, `ssrd`, `strd` — predicted, never fed back |
| **model input** | **95** | prognostic + forcing |
| **model output** | **87** | prognostic + diagnostic |

The "87 variables" figure in the project notes is the **output** count.

What the **dataset** must store is a third number: **89**. The 9 sin/cos and
insolation terms are computed at load time, so from disk we need
84 prognostic + `lsm` + `z` + the 3 diagnostics = 89 fields.

    89 stored  ->  +9 computed  =  95 model inputs  ->  87 outputs

`tp` being **diagnostic** matters for this project directly: precipitation is
predicted but never fed back into the recurrent state. A tail-aware `twCRPS`
term on `tp` therefore acts on a diagnostic head, with a shorter and more
isolated gradient path than a prognostic variable would have. Worth knowing
before designing the loss, and a point in favour of a decoder-only fine-tune.

### multistep_input = 2

**One forecast needs two consecutive analysis times, not one.** For an
initialisation at `2025-04-01T00:00`, both of these must exist in the dataset:

    2025-03-31T18:00    (t-6h)
    2025-04-01T00:00    (t0)

This applies to both sides of the cutout — MEPS and the global N320 — and it
doubles the minimum data pull. Still trivial in volume, but it is a correctness
requirement rather than a convenience: initialising from a single state will
either fail outright or silently produce a garbage first step.

## 3. Time steps

`timestep: 6h`, `frequency: 6h`, `leadtimes: 10` → a 60-hour forecast.

The number of *input* states is not set in either config, so it falls back to
the Anemoi default (`multistep_input: 2`, i.e. t-6h and t0). **Confirm this from
the checkpoint before building initial conditions** — if it is 2, a single
forecast needs two consecutive analysis times, not one.

Volume is trivial either way: two states of 89 fields, not a training corpus.

## 4. The data problem, and the substitution

MET's global input is not one dataset but a **join**:

```yaml
join:
  - dataset: ${...}/${hardware.files.dataset_atm}    # IFS operational analysis
  - dataset: ${...}/${hardware.files.dataset_land}   # ERA5
    select: [lcc, mcc, hcc, tcc, strd, ssrd]
drop: [sdor, slor, cp, u_600, v_600, w_600, q_600, z_600, t_600]
```

| Source | Public? | Status |
|---|---|---|
| MEPS 2.5 km (`aifs-meps-2.5km-...-v7.zarr`) | yes, thredds.met.no | usable |
| ERA5 N320 land/cloud/radiation | yes, via CDS | usable — see below |
| IFS `aifs-od-an-oper-...-n320` | **no** — MARS class `od` | applied for, not granted |

So only the *atmospheric* part is blocked, and six of the surface fields
(`lcc mcc hcc tcc strd ssrd`) already come from ERA5 in MET's own setup.

**Plan: substitute ERA5 N320 (`ea`) for the IFS operational analysis (`od`).**
Same grid, same variable names, public. The model was trained on `od`, so
initialising from `ea` is a distribution shift and some skill loss is expected.
That is acceptable for the current milestone, which is *the model runs and the
output is physical* — not a verification score. It must be revisited before any
result is quoted, and certainly before fine-tuning.

### ERA5 must come from era5-complete, not the standard CDS datasets

Verified 2026-08-26. The standard `reanalysis-era5-single-levels` /
`-pressure-levels` datasets return a **regular 0.25 deg lat/lon grid**, which
does not match the checkpoint's graph. The global side has to be native N320 —
exactly the 536,600 nodes read out of the checkpoint — so regridding is not an
option.

Native N320 comes from **`reanalysis-era5-complete`**:

- API only, no web form
- served from the MARS tape archive: hours to days, even for two states
- needs a registered account, and the licence accepted once through the CDS web
  form before the API returns anything
- no extra licence beyond that

Two consequences. Submit the request as soon as the account exists rather than
waiting for the MEPS side — the tape queue is the long pole. And CDS now uses
ECMWF sign-in, so the ECMWF account requested for the anemoi catalogue may cover
this too; check before creating a second one.

**Accumulated fields need a separate source.** `tp`, `ssrd` and `strd` are
accumulations. ERA5's analysis stream does not carry them — they exist only in
the forecast stream, accumulated over a period. Requesting them with `type: an`
**drops them silently**: the build succeeds and produces a dataset with 86
variables instead of 89, with no error anywhere. It surfaces only as a count in
the log, or later as a failure to select `tp` when the cutout is assembled.

The recipe generator now emits a third join branch using the `accumulations`
source, which pulls from forecasts and accumulates over the dataset frequency.

**Route the request through CDS.** anemoi-datasets' `mars` source takes a
`use_cdsapi_dataset` argument; setting it to `reanalysis-era5-complete` sends a
MARS-style request (`class: ea`, `grid: N320`) over the CDS API rather than
needing direct MARS access. The recipe generator emits this.

**Build datasets in a separate environment.** `cdsapi` is not in the inference
lockfile, so the environment `setup_env.sh` builds cannot reach CDS at all.
Rather than perturb an environment that is verified for inference:

    source ~/extreme-bris/scripts/env.sh    # REQUIRED: eX3 re-signs TLS
    uv venv ~/bris-data-env
    uv pip install --python ~/bris-data-env \
        "anemoi-datasets[all]==0.5.24" "anemoi-utils==0.4.23" cdsapi

**Pin `anemoi-utils` too.** Pinning only anemoi-datasets is not enough: 0.5.24's
`fix_provenance` assumes `module_versions` values are strings, while newer
anemoi-utils records dicts, and the build dies with
`'dict' object has no attribute 'startswith'` in the `patch` step. The lockfile
pairs 0.5.24 with anemoi-utils 0.4.23.

That failure comes at step 7 of 9, after `init`, `load` and `finalise` — so the
data and statistics are already written and the dataset may well be usable.
`patch` only tidies provenance metadata, which nothing at inference reads. Check
with `anemoi-datasets inspect` before assuming a rebuild is needed.

**Pin 0.5.24, matching the inference lockfile.** A dataset written by a newer
anemoi-datasets may carry a format version the pinned reader refuses, and that
would only surface when the forecast is run — long after the dataset is built.
0.5.24 supports everything needed here: `use_cdsapi_dataset` is in its
`mars` source, and `trim_edge` is in its `TrimEdge`.

Without the first line `uv` fails with `invalid peer certificate: UnknownIssuer`,
because it uses its own bundled roots rather than the system trust store. Every
command here that reaches the network needs it.

Then put the CDS key in `~/.cdsapirc` and build with that environment.

## 4b. The datasets are not published — they have to be built

Checked 2026-08-25: there is no downloadable anemoi-format zarr for either side
of the cutout.

| Source | Status |
|---|---|
| `met-no` datasets on Hugging Face | none exist |
| ECMWF `ml-datasets` object store | HTTP 403 |
| data.met.no / thredds.met.no | MEPS is public, but as GRIB/NetCDF |

So task 5 is a **build**, not a download: `anemoi-datasets create` from raw
sources, on both sides. `scripts/make_era5_recipe.py` generates the global-side
recipe from the checkpoint metadata so the variable list, levels and date range
are exactly what the model expects:

    python scripts/make_era5_recipe.py ~/bris-runs/ckpt-metadata.json \
        --date 2025-04-01T00:00:00 -o bris/configs/era5_n320.yaml

It emits 89 stored fields, 17 surface and 6 x 12 upper-air, over the two states
`multistep_input: 2` requires. The `input:` stanza still needs checking against
the installed anemoi-datasets version, and ERA5 needs CDS credentials.

**But there is a catalogue.** MET's own regional tutorial points at
`https://anemoi.ecmwf.int/datasets`, which lists ready-made anemoi datasets with
S3 download links and paths on LUMI and Leonardo. It is gated behind ECMWF
login, not unpublished — verified: `/datasets` returns 302 to a login and the
API returns 401. Norway is an ECMWF member state, so credentials are plausibly
obtainable through UiO or Simula.

That reframes the ask: not "send me your data", but "are the Bris datasets in
the catalogue, and how do I get access". See `docs/PUBLIC_RESOURCES.md` for the
full audit of what is already public — including `metno/anemoi-regional-tutorial`,
MET's own guide to building regional datasets, which is the reference this
recipe generator should be checked against. That would replace days of dataset engineering with an email,
and it also settles whether ERA5 is an acceptable substitute for `od` in their
view. Building from scratch is the fallback, not the obvious first move.

## 4c. Inference needs initial conditions, not the training archive

An easy and expensive thing to get wrong. Running one forecast does **not**
require the datasets Bris was trained on.

| | states | volume |
|---|---|---|
| Training archive | 2020-2023+, 6-hourly, both grids | TB scale |
| **One forecast** | **2** (t-6h, t0) | **~1.1 GB** |

At 89 fields x 2 states x float32: MEPS 949 x 1069 = 0.72 GB, N320 = 0.39 GB.
That is small enough to build on demand and throw away, which suits a cluster
with no long-term storage.

Two conditions make this more than a download:

**Format.** It must be an anemoi-datasets zarr that `cutout` can open — the
config reads through anemoi-datasets, not raw GRIB. So still
`anemoi-datasets create`, just over a two-state range rather than an archive.

**Grid geometry — the actual risk.** `switch_graph: null` means the model uses
the graph baked into the checkpoint, which encodes fixed node positions for the
stretched grid. The MEPS dataset must therefore match MET's domain, projection
and 949 x 1069 extent exactly, or the graph will not align with the data. N320
is safe because it is a standard grid; MEPS is MET's own domain and is not.

### Grid, resolved from the checkpoint (2026-08-25)

Read out of the checkpoint graph with `scripts/dump_grid.py` and split with
`scripts/analyse_grid.py`. No longer inferred:

| | |
|---|---|
| data nodes | 1,359,281 |
| LAM (MEPS) | **822,681 = 849 x 969** |
| global (N320) | 536,600 |
| N320 latitude rows | **640** — confirms N320 |
| removed under the LAM | 5,480 |

**MEPS domain, post-`trim_edge: 50`:**

    latitude    51.119 N .. 74.131 N
    longitude   13.150 W .. 49.418 E

The pre-trim extent is therefore 949 x 1069, matching what `CRPSFFTLoss`
implied. The 5,480 global points dropped under the LAM agree with the ~5,350
expected from the footprint area against a 31 km cell, the small excess being
what a curved Lambert-conformal domain cuts out of a Gaussian grid.

Coordinates are saved in `grid.npz` and are the thing to validate a built
dataset against — node order included, since that is what the model is indexed
on.

### Four ways to get the grid definition, in order of cost

Asking MET is the obvious route but not the only one, and not the first.

**1. Read it out of the checkpoint.** The same fact that makes the grid a hard
constraint also solves it: `switch_graph: null` means the graph is stored in the
checkpoint, and an Anemoi graph carries explicit lat/lon coordinates for every
node. The grid is therefore already in hand.

    cd $BRIS_ENV_DIR && uv run python scripts/dump_grid.py \
        $BRIS_MODEL_DIR/bris-forecaster/bris-crpsfft_inference.ckpt \
        --npz ~/bris-runs/grid.npz

This gives the authoritative node coordinates — better than any description,
because it is what the model will actually be indexed against. It also confirms
or refutes the inferred 949 x 1069 directly.

**2. Dataset provenance in the checkpoint metadata.** anemoi-datasets records
the source specification of the dataset a model was trained on. Worth grepping
`ckpt-metadata.json` for the recipe before building anything:

    python -c "import json;d=json.load(open('ckpt-metadata.json'));print(json.dumps(d.get('dataset'),indent=2)[:4000])"

**3. The public MEPS archive.** MEPS is a standard MET product on
thredds.met.no, and the NetCDF headers carry the full projection, spacing and
extent. That pins the grid geometry even without the anemoi dataset.

**4. Ask MET.** Still worth doing, but for the question routes 1-3 cannot
answer: whether substituting ERA5 for the operational analysis is acceptable.
Not for the grid.

### Resolved: statistics come from the checkpoint

Traced through bris-inference on 2026-08-26. Normalisation runs entirely through
`self.model.pre_processors`, and `BrisPredictor.__init__` sets
`self.model = checkpoint.model` — so the processors, and the training statistics
inside them, come from the checkpoint. Outside the `legacy/` module there is not
a single reference to `statistics` in the package.

**Two-state statistics are therefore harmless.** anemoi-datasets computes and
stores them in the zarr, but nothing at inference reads them. This was the one
identified failure mode that would have completed successfully while writing
physically wrong fields; it does not apply.

It does still apply to training. Fine-tuning uses anemoi-training, which is a
different code path and does read dataset statistics — so a fine-tuning corpus
must be large enough for them to mean something.

## 5. Sanity checks on the output

Per `routing`, two NetCDF files are written per run — a `nordic_` file on the
MEPS domain and a `global_` file interpolated to 0.25°, each with `2t`, `msl`,
`tp` and derived `ws`.

- No NaNs anywhere.
- `2t` in a plausible Kelvin range; `msl` around 950–1050 hPa.
- `tp` **exactly zero** in dry areas, never slightly negative — the model applies
  `ReluBounding` to precipitation. Small negatives would mean the bounding did
  not load, i.e. something is wrong with the checkpoint or config.
- Cloud fractions `tcc/hcc/mcc/lcc` within [0, 1] — `HardtanhBounding`.
- The LAM/global seam should be continuous. A visible discontinuity is the first
  thing to look for if the cutout or `trim_edge` is misconfigured.

## 6. Storage

eX3 offers **no long-term storage**, and BeeGFS is hybrid spinning disk + SSD.

- Checkpoints (1.45 GB inference, 3.29 GB training) on BeeGFS: fine.
- Input zarr: stage to **node-local NVMe** (30 TB on the 4124GO-NART) before the
  run. Do not stream from BeeGFS.
- Treat anything on the cluster as ephemeral; keep the reproducible recipe here
  in git, not the data.
