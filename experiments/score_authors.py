"""Score checkpoints from the authors' released code with the same gridness scorer used for all
our runs (gridscore.py), plus a simple grid-spacing estimate, and save rate-map images.

Run inside the authors' environment (needs torch, numpy, scipy, matplotlib):
    python /path/to/xu2025_mine/experiments/score_authors.py ../authors_runs

It finds every checkpoint-step*.pth below the given folder, takes the latest per run, and writes
authors_scores.md (one row per run) plus <run>/rate_maps_scored.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gridscore import autocorrelogram, gridness  # noqa: E402


def find_v(obj, prefix=""):
    """Return (name, array) for the first 3-D tensor shaped (n, G, G) whose name ends in 'v'."""
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 3 and obj.shape[1] == obj.shape[2] and prefix.split(".")[-1] == "v":
            return prefix, obj.detach().cpu().float().numpy()
        return None
    if isinstance(obj, dict):
        items = obj.items()
    elif hasattr(obj, "state_dict"):
        items = obj.state_dict().items()
    else:
        return None
    for k, val in items:
        hit = find_v(val, f"{prefix}.{k}" if prefix else str(k))
        if hit:
            return hit
    return None


def spacing(m):
    """Median distance (in lattice cells) from the autocorrelogram centre to its 6 nearest peaks."""
    sac = np.nan_to_num(autocorrelogram(m), nan=-1.0)
    c = (np.array(sac.shape) - 1) / 2
    peaks = (sac == maximum_filter(sac, size=5)) & (sac > 0.1)
    yy, xx = np.nonzero(peaks)
    d = np.hypot(yy - c[0], xx - c[1])
    d = np.sort(d[d > 2])[:6]
    return float(np.median(d)) if len(d) else float("nan")


def main(root):
    root = Path(root)
    ckpts = {}
    for p in sorted(root.rglob("checkpoint-step*.pth")):
        step = int(p.stem.split("step")[-1])
        run = p.relative_to(root).parts[0]
        if step >= ckpts.get(run, (None, -1))[1]:
            ckpts[run] = (p, step)
    rows = []
    for run, (p, step) in sorted(ckpts.items()):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        hit = find_v(ck)
        if hit is None:
            print(f"{run}: no (n, G, G) tensor named 'v' found in {p}; keys: {list(ck)[:10] if isinstance(ck, dict) else type(ck)}")
            continue
        name, v = hit
        G = v.shape[-1]
        g = np.array([gridness(m.T) for m in v])
        sp = np.array([spacing(m.T) for m in v]) / G          # fraction of arena width
        rows.append(dict(run=run, step=step, tensor=name, n=len(g),
                         gridness_median=float(np.median(g)), valid_pct=float(100 * np.mean(g > 0.37)),
                         spacing_median=float(np.nanmedian(sp)), gridness=g.tolist()))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n = len(v)
            cols = 6
            fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(10, 1.8 * np.ceil(n / cols)))
            for k, ax in enumerate(np.ravel(axes)):
                ax.axis("off")
                if k < n:
                    ax.imshow(v[k], cmap="jet", origin="lower")
                    ax.set_title(f"{g[k]:.2f}", fontsize=8)
            fig.suptitle(f"{run} (step {step}), annulus gridness per cell")
            fig.tight_layout()
            fig.savefig(root / run / "rate_maps_scored.png", dpi=100)
            plt.close(fig)
        except Exception as e:  # images are a convenience only
            print(f"{run}: could not save image ({e})")
    lines = ["| Run | Step | Hexagonal (median gridness > 0.37) | Median annulus gridness | Valid % | Median spacing (fraction of arena width) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['step']} | {'yes' if r['gridness_median'] > 0.37 else 'no'} | "
                     f"{r['gridness_median']:.2f} | {r['valid_pct']:.0f} | {r['spacing_median']:.3f} |")
    (root / "authors_scores.md").write_text("\n".join(lines) + "\n")
    (root / "authors_scores.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
