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
