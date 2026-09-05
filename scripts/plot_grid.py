#!/usr/bin/env python3
"""Draw the stretched grid: where the coarse global points stop and the fine ones begin.

    ~/bris-env/.venv/bin/python scripts/plot_grid.py \
        --points /tmp/gridpts.npz --lam /tmp/lampts.npz -o results/extremes

The two halves are not a picture of weather, they are a picture of geometry -
which is the part that is hard to see from the numbers alone. The global half
has a hole in it exactly the shape of the fine half, because the cutout removed
those points rather than letting the two overlap.

Points are thinned for drawing. Every fourth global point and every three
hundredth LAM point is enough to show the density difference; drawing all
1.36 million would be a solid block of colour.
"""

from __future__ import annotations

import argparse
from pathlib import Path

GLOBAL_C = "#94a3b8"
LAM_C = "#2563eb"


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

    a = np.load(args.points)
    lat, lon = a["lat"], a["lon"]
    b = np.load(args.lam)
    nlam = b["lat"].size
    lam_lat, lam_lon = lat[:nlam], lon[:nlam]
    g_lat, g_lon = lat[nlam:], lon[nlam:]

    args.out.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # left: the whole picture, so the hole is visible
    ax1.scatter(g_lon[::4], g_lat[::4], s=0.6, c=GLOBAL_C, linewidths=0,
                label=f"Globalt, 31 km ({g_lat.size:,} punkter)".replace(",", " "))
    ax1.scatter(lam_lon[::300], lam_lat[::300], s=0.6, c=LAM_C, linewidths=0,
                label=f"MEPS, 2,5 km ({nlam:,} punkter)".replace(",", " "))
    ax1.set_xlim(-40, 55)
    ax1.set_ylim(45, 82)
    ax1.set_xlabel("lengdegrad")
    ax1.set_ylabel("breddegrad")
    ax1.set_title("Det grove rutenettet har et hull i seg", fontsize=11, loc="left")
    leg = ax1.legend(markerscale=14, frameon=False, loc="lower right", fontsize=9)
    for h in leg.legend_handles:
        h.set_alpha(1)
    ax1.grid(alpha=.25)

    # right: zoomed to the seam, where the density difference is the point
    m = (g_lon > 4) & (g_lon < 12) & (g_lat > 58) & (g_lat < 63)
    ml = (lam_lon > 4) & (lam_lon < 12) & (lam_lat > 58) & (lam_lat < 63)
    ax2.scatter(g_lon[m], g_lat[m], s=26, c=GLOBAL_C, linewidths=0)
    ax2.scatter(lam_lon[ml][::40], lam_lat[ml][::40], s=1.2, c=LAM_C, linewidths=0)
    ax2.set_xlim(4, 12)
    ax2.set_ylim(58, 63)
    ax2.set_xlabel("lengdegrad")
    ax2.set_title("Samme utsnitt, Sør-Norge: 12 ganger tettere", fontsize=11, loc="left")
    ax2.grid(alpha=.25)

    fig.suptitle("Ett rutenett, to oppløsninger — til sammen 1 359 281 punkter",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    p = args.out / "strukket-rutenett.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {p}")
    print(f"  global {g_lat.size}  lam {nlam}  total {lat.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
