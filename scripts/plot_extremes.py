#!/usr/bin/env python3
"""Figures for the 2025-10-04 extreme-precipitation verification.

    ~/bris-env/.venv/bin/python scripts/plot_extremes.py \
        --pairs /tmp/meps_baseline.json -o results/extremes

Reads the gauge/Bris/MEPS triples written by the baseline comparison and draws
the two panels the result rests on: how the three distributions compare, and
how many gauges clear each warning-relevant threshold.

MEPS is the control, not a rival. It is on the same 2.5 km grid, scored at the
same gauges by the same nearest-neighbour rule, from the same cycle Bris was
initialised from. Whatever a grid-cell-against-a-point comparison costs, it
costs both equally - so the gap between them is the model, not the method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OBS_C = "#111111"
MEPS_C = "#2563eb"
BRIS_C = "#dc2626"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("results/extremes"))
    args = ap.parse_args()

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(args.pairs.read_text())
    o = np.array([x["obs"] for x in d])
    b = np.array([x["bris"] for x in d])
    m = np.array([x["meps"] for x in d])
    n = len(o)
    args.out.mkdir(parents=True, exist_ok=True)

    # --- figure 1: distributions + exceedance --------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    qs = np.array([50, 60, 70, 75, 80, 85, 90, 95, 98, 99, 99.5, 100])
    for arr, c, lbl in [(o, OBS_C, "Observert"), (m, MEPS_C, "MEPS"), (b, BRIS_C, "Bris")]:
        ax1.plot(qs, [np.percentile(arr, q) for q in qs], "o-", color=c,
                 lw=2, ms=4, label=lbl)
    ax1.set_xlabel("persentil over de 678 målerne")
    ax1.set_ylabel("nedbør 06–06 UTC (mm)")
    ax1.set_title("Fordelingen, ikke bare gjennomsnittet", fontsize=11, loc="left")
    ax1.grid(alpha=.3)
    ax1.legend(frameon=False)

    thr = [20, 30, 40, 50, 60, 75, 100]
    x = np.arange(len(thr))
    w = 0.27
    for k, (arr, c, lbl) in enumerate([(o, OBS_C, "Observert"), (m, MEPS_C, "MEPS"),
                                       (b, BRIS_C, "Bris")]):
        ax2.bar(x + (k - 1) * w, [int((arr >= t).sum()) for t in thr], w,
                color=c, label=lbl)
    for k, t in enumerate(thr):
        nb = int((b >= t).sum())
        if nb == 0:
            ax2.text(x[k] + w, 2, "0", ha="center", va="bottom",
                     color=BRIS_C, fontsize=9, fontweight="bold")
    ax2.set_xticks(x, [f"≥{t}" for t in thr])
    ax2.set_xlabel("terskel (mm/døgn)")
    ax2.set_ylabel("antall målere over terskel")
    ax2.set_title("Over 75 mm finnes Bris ikke", fontsize=11, loc="left")
    ax2.grid(alpha=.3, axis="y")
    ax2.legend(frameon=False)

    fig.suptitle(f"4. oktober 2025 — {n} målere i MEPS-domenet", fontsize=12, y=1.02)
    fig.tight_layout()
    p1 = args.out / "ekstrem-fordeling.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- figure 2: gauge by gauge --------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    top = max(o.max(), m.max(), b.max()) * 1.05
    ax.plot([0, top], [0, top], color="#999", lw=1, ls="--", zorder=1)
    ax.scatter(o, m, s=13, color=MEPS_C, alpha=.55, label="MEPS", zorder=2)
    ax.scatter(o, b, s=13, color=BRIS_C, alpha=.55, label="Bris", zorder=3)
    # The line no Bris point crosses.
    ax.axhline(b.max(), color=BRIS_C, lw=1, ls=":", zorder=1)
    ax.text(top * .98, b.max() + 2, f"Bris' høyeste punkt, {b.max():.0f} mm",
            ha="right", va="bottom", color=BRIS_C, fontsize=9)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("observert (mm/døgn)")
    ax.set_ylabel("prognose (mm/døgn)")
    ax.set_title("Hver måler, samme rutenett, samme metode", fontsize=11, loc="left")
    ax.grid(alpha=.3)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    p2 = args.out / "ekstrem-spredning.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"wrote {p1}\nwrote {p2}")
    for lbl, arr in [("observert", o), ("MEPS", m), ("Bris", b)]:
        print(f"  {lbl:9s} mean={arr.mean():5.1f}  p99={np.percentile(arr,99):5.1f}  "
              f"max={arr.max():5.1f}  >=75mm={int((arr>=75).sum()):3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
