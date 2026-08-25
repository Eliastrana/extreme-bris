# extreme-bris

Master's thesis work on extreme-weather performance in data-driven weather
forecasting, centred on MET Norway's [Bris](https://huggingface.co/met-no) model.

UiO (Institutt for informatikk) / Simula.

## Problem

Data-driven weather models systematically underestimate extremes, and Bris is no
exception. The CRPS-FFT paper reports, in its own verification:

- underestimation of the frequency of large precipitation events (Q-Q)
- underestimation of wind speed at point locations
- less ensemble spread than MEPS — the ensemble represents control-analysis
  uncertainty only, since it is initialised from a single analysis

MET's working hypothesis is that this needs more high-resolution training data.
This project tests a complementary hypothesis: that a meaningful part of the
deficit lives in the *training objective*, and can be recovered by fine-tuning.

## Approach

The core lever is a tail-aware training objective

    L = CRPS + gamma * twCRPS_tau

where the threshold-weighted CRPS is obtained by censoring both ensemble members
and target through the chaining function `v(x) = max(tau, x)`, so the existing
kernel-CRPS implementation can be reused unchanged.

Two tracks:

1. **Small model** — controlled experiments on a small regional model where full
   training runs are cheap enough to ablate loss, sampling and training-data
   volume independently. Built on `neural-lam` / CRPS-LAM over MEPS Nordic.
2. **Bris fine-tuning** — apply the best recipe to the operational CRPS-FFT
   checkpoint as a decoder-only fine-tune, and test whether it transfers.

Evaluation avoids the forecaster's dilemma by using proper weighted scoring
rules (twCRPS, threshold-weighted potential CRPS) rather than conditioning on
observed extremes, and verifies against station observations rather than only
the analysis used for training.

## Layout

    scripts/    setup and job scripts
    bris/       Bris-related code, configs and experiments

## Getting the models

    ./scripts/download_models.sh -c     # configs only, fast first look
    ./scripts/download_models.sh        # full checkpoints, several GB

Both HF repos are Apache 2.0. Checkpoints are gitignored — do not commit them.

Training requires the fork with the FFT-CRPS loss, since upstream Anemoi does
not ship it: `evenmn/anemoi-core`, branch `feat/crps-fft-loss`.

## Running Bris on eX3

    ./scripts/download_models.sh -c            # configs first, ~16 KB
    ./scripts/download_models.sh               # checkpoints, 4.7 GB total
    ./scripts/setup_env.sh                     # build the pinned env (on a GPU node)
    python scripts/inspect_checkpoint.py ~/bris-models/bris-forecaster/bris-crpsfft_inference.ckpt
    mkdir -p logs && sbatch bris/slurm/bris_inference.sbatch

See [docs/INPUTS.md](docs/INPUTS.md) for what the model needs as input and how we
plan to assemble it without MARS access.

### Findings that changed the plan

Read before spending cluster time — four things are not as the published
material suggests.

**The runner is `bris`, not `anemoi-inference`.** `config_inference.yaml`
targets `bris.model.brispredictor.BrisPredictor`, and `anemoi-inference` is not
in the dependency set at all. The `anemoi-inference run <config>` invocation
belongs to `met-no/bris-forecaster-pretrained`, which is a different model.

**`bris-forecaster-pretrained` ships no checkpoint.** The repo contains only
configs, `pyproject.toml`, `uv.lock` and a README; its inference config points
at a path on Leonardo. There is no lighter global model to smoke-test against —
the CRPS-FFT checkpoint is the only published weight set, so it is also the
first thing that has to work.

**Inference wants 8 GPUs, not 1.** `num_gpus_per_model: 4` with
`num_members: 2` — the model is sharded across four cards and two ensemble
members run alongside, which is exactly one 4124GO-NART node. Whether it fits on
fewer cards on 80 GB A100s is worth testing, but it is not a single-GPU job by
default.

**`uv sync --locked` will fail without a workaround.** The lockfile pins the
inference package over SSH:

    bris = { git = "ssh://git@github.com/metno/bris-inference.git", rev = "d1d27c1..." }

which needs a GitHub SSH key on the compute node. The repo is public, so
`scripts/setup_env.sh` adds a git `insteadOf` rewrite to HTTPS. That leaves the
URL string in `uv.lock` untouched, so `--locked` still validates.

### Open questions

- Slurm partition name for the 8x A100 80 GB node — `sinfo` on eX3.
- `multistep_input` is unset in both configs, so it defaults to 2 (t-6h and t0).
  Confirm from the checkpoint before building initial conditions.
- Node walltime limits, which is why the Slurm script resumes by date marker.

## References

- Nordhagen et al., *High-Resolution Probabilistic Data-Driven Weather Modeling
  with a Stretched Grid*, arXiv:2511.23043 — Bris CRPS-FFT
- Nipen et al., *Regional Data-Driven Weather Modeling with a Global Stretched
  Grid*, AIES 5(2), 2026 — arXiv:2409.02891
- Bakketun et al., *Enhancing a high resolution data-driven weather prediction
  model with surface descriptors*, arXiv:2607.02824 — decoder-only fine-tuning
- Lang et al., *AIFS-CRPS*, arXiv:2412.15832 — almost-fair CRPS
- Allen et al., *Improving probabilistic forecasts of extreme wind speeds by
  training statistical post-processing models with weighted scoring rules*,
  MWR 153(8), 2025 — arXiv:2407.15900
- Olivetti & Messori, *Do data-driven models beat numerical models in
  forecasting weather extremes?*, GMD 17, 7915, 2024
- Oskarsson et al., `neural-lam` — https://github.com/mllam/neural-lam
- CRPS-LAM, arXiv:2510.09484
