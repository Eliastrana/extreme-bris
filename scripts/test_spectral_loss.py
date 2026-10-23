#!/usr/bin/env python3
"""Check xbris/spectral.py against the loss Bris was actually trained with, on the CPU.

    scripts/test_spectral_loss.py

The port claims to compute what evenmn/anemoi-core f6f4a82's AFCRPSFFTLoss
computed, only in variable chunks under activation checkpointing. This holds it
to that. The reference below is that commit's afcrps_fft.py transcribed as it
stands, on top of anemoi 0.5.1's AlmostFairKernelCRPS, whose kernel CRPS is the
fork's character for character.

  same value      port and reference agree, for several chunk sizes, with a
                  variable weight and the (1, 1) NaN mask attached as scalers
  same gradient   and so do their gradients with respect to the prediction
  regional only   nodes past xdim * ydim (the global grid) do not matter
  mask            the low-pass circle keeps the count it should at Bris's size
  full size       one forward and backward at 849 x 969 with three members
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _venv  # noqa: E402

_venv.ensure("anemoi", "torch")

import einops  # noqa: E402
import torch  # noqa: E402
from anemoi.training.losses import AlmostFairKernelCRPS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbris.spectral import SpectralAlmostFairKernelCRPS  # noqa: E402


class ReferenceAFCRPSFFTLoss(AlmostFairKernelCRPS):
    """evenmn/anemoi-core f6f4a82, training/src/anemoi/training/losses/afcrps_fft.py, transcribed."""

    def __init__(self, xdim, ydim, alpha=1.0, no_autocast=True, ignore_nans=False, **kwargs):
        super().__init__(alpha=alpha, ignore_nans=ignore_nans, **kwargs)
        self.xdim = xdim
        self.ydim = ydim
        self.len_reg = xdim * ydim
        self.transform = torch.fft.fft2
        self.mask = self.lowpass_mask_2d(xdim, ydim)
        self.no_autocast = no_autocast

    @staticmethod
    def lowpass_mask_2d(nx, ny):
        fx = torch.fft.fftfreq(nx)
        fy = torch.fft.fftfreq(ny)
        KX, KY = torch.meshgrid(fx, fy, indexing="ij")
        k = torch.sqrt(KX * KX + KY * KY)
        mask = k < 0.5
        return einops.rearrange(mask, "x y -> 1 1 (y x)")

    def _discrete_transform(self, preds, targets, batch_size):
        preds_spectral = self.transform(preds)
        targets_spectral = self.transform(targets)
        preds_spectral = einops.rearrange(preds_spectral, "(bs v) e y x -> bs v (y x) e", bs=batch_size)
        targets_spectral = einops.rearrange(targets_spectral, "(bs v) y x -> bs v (y x)", bs=batch_size)
        kcrps_ = self._kernel_crps(preds_spectral, targets_spectral, self.alpha)
        return kcrps_ * self.mask.to(preds.device)

    def forward(self, y_pred, y_target, squash=True, *, scaler_indices=None, without_scalers=None,
                grid_shard_slice=None, group=None):
        bs_ = y_pred.shape[0]
        y_pred_regional = y_pred[:, :, : self.len_reg]
        y_target_regional = y_target[:, : self.len_reg]
        y_pred_regional = einops.rearrange(y_pred_regional, "bs e (y x) v -> (bs v) e y x", x=self.xdim, y=self.ydim)
        y_target_regional = einops.rearrange(y_target_regional, "bs (y x) v -> (bs v) y x", x=self.xdim, y=self.ydim)
        kcrps_ = self._discrete_transform(y_pred_regional, y_target_regional, bs_)
        kcrps_ = einops.rearrange(kcrps_, "bs v latlon -> bs 1 latlon v")
        scaled = self.scale(kcrps_, scaler_indices, without_scalers=without_scalers)
        return scaled.mean()


XDIM, YDIM, NGLOBAL, NVARS, NENS, BS = 12, 10, 37, 5, 3, 2


def with_scalers(loss, nvars, seed=0):
    """Attach what Bris's spectral term carries: a weight per variable and the (1, 1) NaN mask."""
    g = torch.Generator().manual_seed(seed)
    loss.add_scaler(3, 0.5 + torch.rand(nvars, generator=g), name="general_variable")
    loss.add_scaler((2, 3), torch.ones(1, 1), name="nan_mask_weights")
    return loss


def sample(seed=1, xdim=XDIM, ydim=YDIM, nglobal=NGLOBAL, nvars=NVARS, nens=NENS, bs=BS):
    g = torch.Generator().manual_seed(seed)
    n = xdim * ydim + nglobal
    target = torch.randn(bs, n, nvars, generator=g)
    pred = target[:, None] + 0.7 * torch.randn(bs, nens, n, nvars, generator=g)
    return pred, target


def main() -> int:
    results = []

    def check(name, ok, detail):
        results.append((name, ok))
        print(f"  {'pass' if ok else 'FAIL'}  {name:15s} {detail}")

    ref = with_scalers(ReferenceAFCRPSFFTLoss(XDIM, YDIM, alpha=0.95), NVARS)
    pred, target = sample()

    pred_ref = pred.clone().requires_grad_(True)
    value_ref = ref(pred_ref, target)
    value_ref.backward()

    for chunk in (1, 2, 3, NVARS, 64):
        port = with_scalers(SpectralAlmostFairKernelCRPS(XDIM, YDIM, alpha=0.95, chunk_variables=chunk), NVARS)
        pred_port = pred.clone().requires_grad_(True)
        value_port = port(pred_port, target)
        value_port.backward()
        dv = abs(float(value_port) - float(value_ref))
        dg = float((pred_port.grad - pred_ref.grad).abs().max())
        gmax = float(pred_ref.grad.abs().max())
        check(f"same value/{chunk}", dv <= 1e-6 * abs(float(value_ref)),
              f"{float(value_port):.8f} vs reference {float(value_ref):.8f}")
        check(f"same grad/{chunk}", dg <= 1e-5 * gmax, f"largest difference {dg:.3g} of {gmax:.3g}")

    port = with_scalers(SpectralAlmostFairKernelCRPS(XDIM, YDIM, alpha=0.95, chunk_variables=2), NVARS)
    a = float(port(pred, target))
    pred2, target2 = pred.clone(), target.clone()
    pred2[:, :, XDIM * YDIM:] += 5.0
    target2[:, XDIM * YDIM:] -= 3.0
    b = float(port(pred2, target2))
    check("regional only", a == b, f"{a:.8f} with the global nodes moved, {b:.8f} without")

    big = SpectralAlmostFairKernelCRPS(849, 969, alpha=0.95)
    kept = int(big.mask.sum())
    check("mask", kept == int(ReferenceAFCRPSFFTLoss.lowpass_mask_2d(849, 969).sum()) and 0 < kept < 822_681,
          f"{kept} of 822,681 coefficients inside the circle ({kept / 822_681:.1%})")

    full = with_scalers(SpectralAlmostFairKernelCRPS(849, 969, alpha=0.95, chunk_variables=2), 4)
    fp, ft = sample(seed=3, xdim=849, ydim=969, nglobal=100, nvars=4, nens=3, bs=1)
    fp.requires_grad_(True)
    t0 = time.time()
    v = full(fp, ft)
    v.backward()
    check("full size", torch.isfinite(v).item() and torch.isfinite(fp.grad).all().item(),
          f"value {float(v):.6g}, forward and backward in {time.time() - t0:.1f}s on the CPU")

    bad = [n for n, ok in results if not ok]
    print()
    if bad:
        print(f"{len(bad)} of {len(results)} failed: {', '.join(bad)}", file=sys.stderr)
        return 1
    print(f"All {len(results)} checks hold: the port computes Bris's spectral term.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
