"""Bris's spectral loss term, ported to anemoi-training 0.5.1.

WHAT IT IS. Bris CRPS-FFT was pretrained with two terms: the almost-fair
kernel CRPS on every grid point, and a tenth of the same score computed on the
2-D Fourier coefficients of the MEPS half of the grid. The first scores each
point on its own and cannot see whether neighbouring points hang together. The
second does: a member whose field is noisy has too much amplitude at high
wavenumbers, a blurred one too little, and either costs it. Nordhagen et al.
(2025, arXiv:2511.23043) needed it for spatially coherent fields. Every
fine-tuning in this project has run without it, because the class does not
exist in this anemoi, and scripts/spectra.py measured what that cost: the
control arm has three to four times the analysis' small-scale precipitation
variance, and A has 1.5 to 1.8 times it in wind.

WHERE THIS CODE COMES FROM. Not from the class name. The checkpoint's own
metadata records the loss it was trained with and the code it ran:

    anemoi.training.losses.AFCRPSFFTLoss  xdim 849, ydim 969, alpha 0.95,
        scalers [pressure_level, general_variable, nan_mask_weights],
        loss_weights [1.0, 0.1]
    anemoi.training at evenmn/anemoi-core f6f4a82e27f858f4683946173dd59ecb000b48de

Note AFCRPSFFTLoss, the almost-fair variant, not the CRPSFFTLoss that the
published config (and bris/train/smoke.yaml, copied from it) names. This file
is that commit's training/src/anemoi/training/losses/afcrps_fft.py, read line
by line, with three changes that do not touch the arithmetic:

  - The kernel CRPS is anemoi 0.5.1's AlmostFairKernelCRPS._kernel_crps, which
    is character for character the fork's at that commit (diffed 2026-10-23).
    torch.abs of a complex difference is its modulus, so the score on Fourier
    coefficients is the energy-score form of CRPS in the complex plane, as in
    the original.
  - Variables are scored a few at a time inside torch.utils.checkpoint. Peak
    training memory without this term was 142,987 of 143,771 MiB, and the
    term materialises a members-by-members tensor over 822,681 complex
    coefficients per variable. Chunking and recomputing in the backward pass
    changes only the order of a sum, and scripts/test_spectral_loss.py checks
    value and gradient against a transcription of the original.
  - Inputs are cast to float32 inside the disabled-autocast block. The
    original disables autocast for the same reason and leaves the dtype alone;
    torch.fft has no bfloat16, so on float32 inputs this is a no-op and on
    anything else the original would have raised.

ALPHA. The config's alpha 0.95 is overridden to 1.0 by the CombinedLoss
kwargs, here as in every other term and in Bris's own training; see the note
above training_loss in bris/train/finetune.yaml.

The low-pass mask is the one at that commit: a hard circle, radial frequency
below 0.5 cycles per grid length kept and the corners of the FFT square
dropped. Later commits on the branch tapered it linearly and then removed it;
Bris was not trained with either.

NODE ORDER. The first xdim * ydim = 822,681 grid nodes are the MEPS cutout,
row-major with y slow: the MEPS zarr's field_shape is 1069 x 949, trim_edge 50
makes it 969 x 849, and finetune_tail.yaml's lam_area_weight is defined on
nodes 0 to 822,680. The test checks the orientation with a wave along x.
"""

from __future__ import annotations

import contextlib
import logging

import einops
import torch
from anemoi.training.losses import AlmostFairKernelCRPS
from torch.utils.checkpoint import checkpoint

LOGGER = logging.getLogger(__name__)

_REPORT_FIRST = 5


def lowpass_mask_2d(nx: int, ny: int) -> torch.Tensor:
    """The fork's mask at f6f4a82: True where the radial frequency is below 0.5.

    Built on an (x, y) mesh and flattened as (y x), matching the coefficients'
    own flattening below.
    """
    fx = torch.fft.fftfreq(nx)
    fy = torch.fft.fftfreq(ny)
    kx, ky = torch.meshgrid(fx, fy, indexing="ij")
    k = torch.sqrt(kx * kx + ky * ky)
    return einops.rearrange(k < 0.5, "x y -> 1 1 (y x)")


class SpectralAlmostFairKernelCRPS(AlmostFairKernelCRPS):
    """Almost-fair kernel CRPS on the 2-D FFT of the regional fields (AFCRPSFFTLoss).

    Parameters
    ----------
    xdim, ydim:
        Shape of the regional grid, x fast. 849 and 969 for Bris.
    alpha:
        As in AlmostFairKernelCRPS. 0.95 for Bris.
    chunk_variables:
        How many output variables to transform and score at once. Memory only;
        the value does not depend on it.
    """

    def __init__(
        self,
        xdim: int,
        ydim: int,
        alpha: float = 1.0,
        no_autocast: bool = True,
        ignore_nans: bool = False,
        chunk_variables: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(alpha=alpha, no_autocast=no_autocast, ignore_nans=ignore_nans, **kwargs)
        self.xdim = int(xdim)
        self.ydim = int(ydim)
        self.len_reg = self.xdim * self.ydim
        self.chunk_variables = int(chunk_variables)
        if self.chunk_variables < 1:
            raise ValueError(f"chunk_variables must be at least 1, got {chunk_variables}")
        self.mask = lowpass_mask_2d(self.xdim, self.ydim)
        self.supports_sharding = False  # a Fourier transform needs the whole field
        self._seen = 0
        LOGGER.info(
            "spectral term: AFCRPS-FFT on the first %d nodes as %d x %d (y x), alpha %.3g, "
            "%d of %d coefficients inside the low-pass circle, %d variables per chunk",
            self.len_reg, self.ydim, self.xdim, self.alpha,
            int(self.mask.sum()), self.len_reg, self.chunk_variables,
        )

    def _spectral_kcrps(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Masked kernel CRPS of the Fourier coefficients.

        preds (bs, v, e, y, x), targets (bs, v, y, x) -> (bs, v, y*x).
        """
        if self.no_autocast:
            context = torch.amp.autocast(device_type=preds.device.type, enabled=False)
            preds, targets = preds.float(), targets.float()
        else:
            context = contextlib.nullcontext()
        with context:
            preds_spectral = torch.fft.fft2(preds)
            targets_spectral = torch.fft.fft2(targets)
            preds_spectral = einops.rearrange(preds_spectral, "bs v e y x -> bs v (y x) e")
            targets_spectral = einops.rearrange(targets_spectral, "bs v y x -> bs v (y x)")
            kcrps_ = self._kernel_crps(preds_spectral, targets_spectral, self.alpha)
        return kcrps_ * self.mask.to(kcrps_.device)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group=None,
    ) -> torch.Tensor:
        del squash, group  # the original reduces with a plain mean, whatever is asked
        if grid_shard_slice is not None:
            raise RuntimeError("the spectral term needs the whole regional grid; set model.keep_batch_sharded=False")

        preds = einops.rearrange(
            y_pred[:, :, : self.len_reg], "bs e (y x) v -> bs v e y x", x=self.xdim, y=self.ydim
        )
        targets = einops.rearrange(
            y_target[:, : self.len_reg], "bs (y x) v -> bs v y x", x=self.xdim, y=self.ydim
        )

        parts = []
        for start in range(0, preds.shape[1], self.chunk_variables):
            stop = start + self.chunk_variables
            p, t = preds[:, start:stop], targets[:, start:stop]
            if torch.is_grad_enabled() and p.requires_grad:
                parts.append(checkpoint(self._spectral_kcrps, p, t, use_reentrant=False))
            else:
                parts.append(self._spectral_kcrps(p, t))
        kcrps_ = torch.cat(parts, dim=1)

        kcrps_ = einops.rearrange(kcrps_, "bs v latlon -> bs 1 latlon v")
        result = self.scale(kcrps_, scaler_indices, without_scalers=without_scalers).mean()

        if self._seen < _REPORT_FIRST:
            self._seen += 1
            LOGGER.info(
                "spectral term, forward %d of %d reported: %.6g (before the CombinedLoss weight)",
                self._seen, _REPORT_FIRST, float(result.detach()),
            )
        return result

    @property
    def name(self) -> str:
        return "AFCRPS-FFT"
