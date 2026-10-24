"""Annulus-masked gridness, as in Sargolini et al. (2006) / Langston et al. (2010), implemented
following Banino et al. (2018): compute the spatial autocorrelogram, correlate it with its
rotations inside an annulus, score min(r60, r120) - max(r30, r90, r150), and take the maximum
over a range of annulus sizes. Used as a cross-check on NeuralPlayground's GridScorer.

    python gridscore.py runs/lam1_20k [runs/...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import rotate
from scipy.signal import correlate2d


def autocorrelogram(m):
    m = m - m.mean()
    ones = np.ones_like(m)
    num = correlate2d(m, m, mode="full")
    n = correlate2d(ones, ones, mode="full")
    s1 = correlate2d(m, ones, mode="full")
    s2 = correlate2d(ones, m, mode="full")
    sq1 = correlate2d(m ** 2, ones, mode="full")
    sq2 = correlate2d(ones, m ** 2, mode="full")
    cov = n * num - s1 * s2
    var = np.sqrt(np.clip(n * sq1 - s1 ** 2, 1e-12, None) * np.clip(n * sq2 - s2 ** 2, 1e-12, None))
    sac = cov / var
    sac[n < 20] = np.nan  # too few overlapping bins to trust
    return sac


def gridness(m, masks=None):
    sac = autocorrelogram(m)
    c = (np.array(sac.shape) - 1) / 2
    yy, xx = np.indices(sac.shape)
    r = np.hypot(yy - c[0], xx - c[1])
    G = m.shape[0]
    if masks is None:  # inner radius 0.2*G fixed-ish, outer radius swept, as in Banino et al.
        masks = [(0.2 * G, k * G / 10) for k in np.arange(4, 12) / 1.0]
    best = -np.inf
    for r_in, r_out in masks:
        ring = (r >= r_in) & (r <= r_out) & np.isfinite(sac)
        base = np.where(np.isfinite(sac), sac, 0.0)
        corr = {}
        for ang in (30, 60, 90, 120, 150):
            rot = rotate(base, ang, reshape=False, order=1)
            a, b = base[ring], rot[ring]
            corr[ang] = np.corrcoef(a, b)[0, 1]
        g = min(corr[60], corr[120]) - max(corr[30], corr[90], corr[150])
        best = max(best, g)
    return best


if __name__ == "__main__":
    for d in sys.argv[1:]:
        ck = torch.load(Path(d) / "checkpoint.pt", map_location="cpu") if (Path(d) / "checkpoint.pt").exists() else None
        maps = ck["state_dict"]["v"].numpy() if ck else None
        if maps is None:
            print(d, "no checkpoint"); continue
        g = np.array([gridness(m.T) for m in maps[:24]])
        s = json.loads((Path(d) / "summary.json").read_text())
        s["gridness_annulus"] = g.tolist()
        s["gridness_annulus_median"] = float(np.median(g))
        (Path(d) / "summary.json").write_text(json.dumps(s, indent=1))
        print(f"{Path(d).name:26s} annulus gridness median {np.median(g):.2f}  (NPG {s.get('gridness_median', float('nan')):.2f})")
