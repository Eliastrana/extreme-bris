#!/usr/bin/env python3
"""Draw the stretched grid: where the coarse global points stop and the fine ones begin.

    ~/bris-env/.venv/bin/python scripts/plot_grid.py \
        --points /tmp/gridpts.npz --lam /tmp/lampts.npz -o results/extremes

This is geometry, not weather, and it is the part that is hard to see from the
numbers. The global half has a hole in it exactly the shape of the fine half,
because the cutout removed those points rather than letting the two overlap.

Points are sampled RANDOMLY, not by stride. The LAM is a curvilinear grid
flattened to one dimension, so taking every Nth point walks along its rows and
draws a set of arcs that look like structure in the data and are an artefact of
the sampling.

The right panel straddles the southern seam near 52 N rather than sitting
inside the hole. Inside it there is nothing coarse left to compare against -
that is the whole point of a cutout - so a zoom there shows one grid and
proves nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

GLOBAL_C = "#64748b"
LAM_C = "#2563eb"


def spaced(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", required=True, type=Path)
    ap.add_argument("--lam", required=True, type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("results/extremes"))
    args = ap.parse_args()

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    a = np.load(args.points)
    lat, lon = a["lat"], a["lon"]
    nlam = np.load(args.lam)["lat"].size
    lam_lat, lam_lon = lat[:nlam], lon[:nlam]
    g_lat, g_lon = lat[nlam:], lon[nlam:]

    def pick(x, y, k, mask=None):
        idx = np.arange(x.size) if mask is None else np.flatnonzero(mask)
        if idx.size > k:
            idx = rng.choice(idx, k, replace=False)
        return x[idx], y[idx]

    args.out.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.6))

    gx, gy = pick(g_lon, g_lat, 60000)
    lx, ly = pick(lam_lon, lam_lat, 30000)
    ax1.scatter(gx, gy, s=0.5, c=GLOBAL_C, linewidths=0,
                label=f"Globalt, 31 km — {spaced(g_lat.size)} punkter")
    ax1.scatter(lx, ly, s=0.5, c=LAM_C, linewidths=0,
                label=f"MEPS, 2,5 km — {spaced(nlam)} punkter")
    ax1.set_xlim(-35, 55)
    ax1.set_ylim(45, 82)
    ax1.set_xlabel("lengdegrad")
    ax1.set_ylabel("breddegrad")
    ax1.set_title("Det grove nettet slutter der det fine begynner",
                  fontsize=11, loc="left")
    leg = ax1.legend(markerscale=16, frameon=True, framealpha=.9,
                     loc="lower right", fontsize=9)
    for h in leg.legend_handles:
        h.set_alpha(1)
    ax1.grid(alpha=.25)

    # The southern seam sits near 52 N: coarse below it, fine above.
    W = dict(lo=(8, 14), la=(48.5, 56))
    mg = ((g_lon > W["lo"][0]) & (g_lon < W["lo"][1])
          & (g_lat > W["la"][0]) & (g_lat < W["la"][1]))
    ml = ((lam_lon > W["lo"][0]) & (lam_lon < W["lo"][1])
          & (lam_lat > W["la"][0]) & (lam_lat < W["la"][1]))
    gx, gy = pick(g_lon, g_lat, 4000, mg)
    lx, ly = pick(lam_lon, lam_lat, 12000, ml)
    ax2.scatter(gx, gy, s=17, c=GLOBAL_C, linewidths=0)
    ax2.scatter(lx, ly, s=1.4, c=LAM_C, linewidths=0)
    ax2.axhline(52.1, color="#dc2626", lw=1.1, ls="--")
    ax2.text(13.85, 52.35, "skjøten", ha="right", color="#dc2626", fontsize=9)
    ax2.set_xlim(*W["lo"])
    ax2.set_ylim(*W["la"])
    ax2.set_xlabel("lengdegrad")
    ax2.set_title("Sør for Danmark: samme utsnitt, to tettheter",
                  fontsize=11, loc="left")
    ax2.grid(alpha=.25)

    fig.suptitle(f"Ett rutenett, to oppløsninger — til sammen {spaced(lat.size)} punkter",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    p = args.out / "strukket-rutenett.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {p}")
    print(f"  global {g_lat.size}  lam {nlam}  total {lat.size}")
    print(f"  in zoom window: {int(mg.sum())} global, {int(ml.sum())} lam")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
