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

    git clone https://github.com/Eliastrana/extreme-bris.git
    cd extreme-bris
    ./run.sh

That is the whole thing. `run.sh` installs `uv` and the HF CLI, downloads the
checkpoints, builds the pinned environment, dumps the checkpoint metadata, and
submits the GPU smoke job — continuing past any failure so one broken step does
not mask the state of the rest. It writes `~/bris-runs/REPORT.md` and prints it.

    ./run.sh --no-gpu              # skip the GPU job
    ./run.sh --mail you@uio.no     # also mail the report

Network-dependent steps run on the login node with concurrency capped at 8
threads, which is what eX3 asks for. That is deliberate: compute nodes may have
no outbound internet, and finding that out inside a queued job wastes a
scheduling round-trip. GPU work always goes through `sbatch`, never an
interactive `srun`, so nothing can be left holding a card.

The individual jobs remain available if you want to drive them by hand:

    sbatch bris/slurm/setup_and_inspect.sbatch   # env + metadata, CPU only
    sbatch bris/slurm/bris_smoke.sbatch          # 1 GPU, reduced config
    sbatch bris/slurm/bris_inference.sbatch      # published config on hgx2q

**Expect the smoke step to report `blocked` today.** There is no input data yet;
that is the open work, not a misconfiguration.

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

**Inference asks for 8 GPUs by default.** `num_gpus_per_model: 4` with
`num_members: 2` — sharded across four cards, two ensemble members alongside,
which is exactly the `hgx2q` node. But those are the numbers MET ran on
Leonardo's 64 GB A100s, not a hard requirement of the checkpoint. For a
does-it-run check, override them downward and stay off the flagship node.

### GPU ladder

Start at the bottom. Only move up when something actually fails.

| Step | Partition | Allocation | Why |
|---|---|---|---|
| 1 | `a40q` | 1x A40 48 GB | idle, uncontended; `bris_smoke.sbatch` default |
| 2 | `a100q` | 2x A100 40 GB | if 48 GB is not enough |
| 3 | `dgx2q` | 4x V100 | first step where the published `num_gpus_per_model: 4` fits |
| 4 | `hgx2q` | 8x A100 80 GB | the published config unmodified; the real runs |

`a100q`/`milanq` have only 2 GPUs per node, so they cannot run
`num_gpus_per_model: 4` — they are for the env build, checkpoint inspection and
reduced-shard tests. `gh200q` is ARM and is excluded: the `uv.lock` is x86/CUDA.

    sbatch bris/slurm/bris_smoke.sbatch      # 1 GPU, 1 member, 12h forecast
    sbatch bris/slurm/bris_inference.sbatch  # published config on hgx2q

**`uv sync --locked` will fail without a workaround.** The lockfile pins the
inference package over SSH:

    bris = { git = "ssh://git@github.com/metno/bris-inference.git", rev = "d1d27c1..." }

which needs a GitHub SSH key on the compute node. The repo is public, so
`scripts/setup_env.sh` adds a git `insteadOf` rewrite to HTTPS. That leaves the
URL string in `uv.lock` untouched, so `--locked` still validates.

### eX3 does TLS interception

Outbound HTTPS is re-signed by a local CA held in the system trust store. `curl`
and `git` work; anything shipping its own roots does not:

    uv        bundled webpki roots  ->  invalid peer certificate: UnknownIssuer
    requests  bundled certifi       ->  SSLCertVerificationError

`scripts/tls_env.sh` locates the system bundle and exports `SSL_CERT_FILE`,
`REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE` and `UV_NATIVE_TLS`/`UV_SYSTEM_CERTS`.
Every script sources it. This trusts what the host already trusts — it does not
disable verification, and if no bundle is found it changes nothing rather than
weakening TLS.

Before running anything here by hand, source the shared paths — this sets
`BRIS_ENV_DIR` and friends, and the TLS trust settings:

    source ~/extreme-bris/scripts/env.sh

Without it, documented commands like `cd $BRIS_ENV_DIR` silently become `cd`
and run against the wrong environment.

Reading the grid definition out of the checkpoint (small, CPU-only — belongs on
the login node, not the queue):

    cd $BRIS_ENV_DIR && uv run python $BRIS_REPO_DIR/scripts/dump_grid.py \
        $BRIS_CKPT --npz $BRIS_RUN_DIR/grid.npz

### Open questions

- Whether the model runs unsharded on one card. Assumed plausible at inference
  (no optimizer state, no retained activations, single member), not established.
- `multistep_input` is unset in both configs, so it defaults to 2 (t-6h and t0).
  Confirm from the checkpoint before building initial conditions.
- Whether ERA5-initialised output is physical enough to be worth scoring.

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
