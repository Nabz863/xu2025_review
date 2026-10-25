"""Appendix figures for the review, built entirely from your own run folders.

    python experiments/make_figures.py --authors ../authors_runs

Reads (relative to the project folder):
  runs/A_recon_s0..5            reconstruction seeds                       (Fig. A1, A3)
  runs/base_seed0..1            paper-text model, lambda = 1               (Fig. A2)
  runs/lam*_s0, lam*_s1         Karabelo's lambda sweep (copy his folders into runs/)   (Fig. A2)
  archive/runs_dev/ctrl_lam075, rescue_lam075_s1   lambda = 0.75, seeds 0 and 1        (Fig. A2)
  runs/ctrl_all_theirs, theirs_seed1..3            authors' configuration, rebuilt     (Fig. A3)
  runs/D_*                      ablations in our code                       (Fig. A3)
  <authors>/authors_scores.json authors' released code, from score_authors.py          (Fig. A3)
Gridness is the annulus score stored by gridscore.py / score_authors.py. Missing runs are
reported and skipped. Writes figures/*.pdf and *.png.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gridscore import autocorrelogram  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MAPS = "jet"  # rate maps and autocorrelograms: the convention in the paper and the grid-cell literature
THRESH = 0.37
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
})
MISSING = []


def gscore(folder):
    """Median annulus gridness of a run folder in our format, or None."""
    f = Path(folder) / "summary.json"
    if not f.exists():
        MISSING.append(str(folder)); return None
    s = json.loads(f.read_text())
    if "gridness_annulus_median" not in s:
        MISSING.append(f"{folder} (not scored: run python gridscore.py {folder})"); return None
    return s["gridness_annulus_median"]


def scores(folders):
    return [g for g in (gscore(f) for f in folders) if g is not None]


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", dpi=600, bbox_inches="tight")  # dpi sets the raster resolution of the maps inside the PDF
    fig.savefig(out / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def smooth(m, f=10):
    """Upsample a 40x40 map bilinearly, which is exactly how the model reads v between lattice points.
    Without this, PDF viewers stretch a 40x40 image across the page and blur it."""
    from scipy.ndimage import zoom
    return zoom(m, f, order=1)


def fig_a1(runs, out):
    """One run per outcome: hexagons, stripes and a square lattice in the linear model, and the
    nonlinear model (ReLU) built the same way as the reconstruction."""
    rows = [("A_recon_s5", "(a) Reconstruction, seed 5: hexagonal"),
            ("A_recon_s4", "(b) Reconstruction, seed 4: parallel stripes"),
            ("base_seed1", "(c) Paper text only, seed 1: square lattice"),
            ("E_recon_nl1_s1", "(d) Nonlinear model 1 (ReLU), seed 1"),
            ("E_recon_nl1_s0", "(e) Nonlinear model 1 (ReLU), seed 0")]
    rows = [(n, label, gscore(runs / n)) for n, label in rows]
    rows = [(n, label, g) for n, label, g in rows if g is not None and (runs / n / "checkpoint.pt").exists()]
    if not rows:
        print("Fig. A1 skipped: no run folders with checkpoints found."); return
    fig, axes = plt.subplots(len(rows), 4, figsize=(6.6, 1.75 * len(rows)), squeeze=False,
                             gridspec_kw=dict(wspace=0.08, hspace=0.4))
    for row, (name, label, g) in enumerate(rows):
        v = torch.load(runs / name / "checkpoint.pt", map_location="cpu")["state_dict"]["v"].numpy()
        for k in range(3):
            ax = axes[row, k]
            ax.imshow(smooth(v[k].T), origin="lower", cmap=MAPS, interpolation="none")
            ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        ax = axes[row, 3]
        ax.imshow(smooth(np.nan_to_num(autocorrelogram(v[0].T), nan=-0.5), 5), origin="lower", cmap=MAPS, vmin=-0.5, vmax=1, interpolation="none")
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        if name.startswith("E_"):   # nonlinear rows: outcome read from the score
            label += ": hexagonal" if g > THRESH else ": not hexagonal"
        axes[row, 0].set_title(f"{label} (median gridness {g:.2f})", loc="left", color=INK)
    axes[0, 3].set_title("Autocorrelogram, cell 1", color=INK2, fontsize=7)
    save(fig, out, "figA1_success_vs_failure")
    print("Fig. A1: " + ", ".join(n for n, _, _ in rows))


def fig_a2(runs, archive, out):
    pts = {}  # lam -> {seed: gridness}
    for d in runs.glob("lam*_s*"):
        m = re.fullmatch(r"lam([0-9.]+)_s(\d+)", d.name)
        if m:
            g = gscore(d)
            if g is not None:
                pts.setdefault(float(m.group(1)), {})[int(m.group(2))] = g
    extra = {(1.0, 0): runs / "base_seed0", (1.0, 1): runs / "base_seed1",
             (0.75, 0): archive / "ctrl_lam075", (0.75, 1): archive / "rescue_lam075_s1"}
    for (lam, sd), d in extra.items():
        g = gscore(d)
        if g is not None:
            pts.setdefault(lam, {})[sd] = g
    if not pts:
        print("Fig. A2 skipped: no lambda runs found."); return
    lams = sorted(pts)
    x = np.arange(len(lams))
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    if 0.75 in lams:
        i = lams.index(0.75)
        ax.axvspan(i - 0.4, i + 0.4, color="#efeeeb", zorder=0)
        ax.text(i, -1.12, "authors' code", color=INK2, fontsize=6.5, ha="center")
    ax.axhline(THRESH, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(-0.35, THRESH + 0.07, "valid-cell threshold (0.37)", color=INK2, fontsize=6.5, ha="left")
    for sd, c, mk in [(0, BLUE, "o"), (1, ORANGE, "s")]:
        y = np.array([pts[l].get(sd, np.nan) for l in lams])
        ax.plot(x, y, mk + "-", color=c, lw=1.2, ms=5, mec="white", mew=1, label=f"Seed {sd}", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"{l:g}" for l in lams]); ax.set_xlim(-0.4, len(lams) - 0.6)
    ax.set_xlabel("λ (weight on the transformation loss; values not to scale)")
    ax.set_ylabel("Median gridness (annulus score)")
    ax.set_ylim(-1.2, 2.0); ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    save(fig, out, "figA2_lambda_sweep")
    print("Fig. A2: lambda values", lams)


def fig_a3(runs, authors, out):
    keys = ["full", "noL1", "noL2", "nonorm", "nononneg"]
    groups = ["None\n(full model)", "No isometry\nloss L₁", "No transformation\nloss L₂", "No\nnormalisation", "No non-\nnegativity"]
    claims = ["hexagonal", "not hexagonal", "disrupted", "disrupted", "hexagonal"]
    data = {
        "Our reconstruction": [scores([runs / f"A_recon_s{i}" for i in range(6)])]
        + [scores([runs / f"D_recon_{k}_s{s}" for s in (0, 1)]) for k in keys[1:]],
        "Authors' configuration, rebuilt": [scores([runs / n for n in ("ctrl_all_theirs", "theirs_seed1", "theirs_seed2", "theirs_seed3")])]
        + [scores([runs / f"D_theirs_{k}_s{s}" for s in (0, 1)]) for k in keys[1:]],
    }
    af = Path(authors) / "authors_scores.json" if authors else None
    if af and af.exists():
        rows = {r["run"]: r["gridness_median"] for r in json.loads(af.read_text())}
        data["Authors' released code"] = [[g for n, g in rows.items() if n.startswith("linear_s10_seed")]] + [
            [g for n, g in rows.items() if n.startswith(f"abl_{k}_seed")] for k in keys[1:]]
    else:
        MISSING.append(f"{af} (authors' released code; pass --authors)")
    style = {"Our reconstruction": (BLUE, "o"), "Authors' configuration, rebuilt": (ORANGE, "s"),
             "Authors' released code": (AQUA, "D")}
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.axhline(THRESH, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(-0.45, THRESH + 0.07, "valid-cell threshold (0.37)", color=INK2, fontsize=6.5, ha="left")
    rng = np.random.default_rng(0)
    n = len(data)
    for j, (name, vals) in enumerate(data.items()):
        c, m = style[name]
        for g, ys in enumerate(vals):
            xs = g + (j - (n - 1) / 2) * 0.24 + rng.uniform(-0.08, 0.08, len(ys))
            ax.scatter(xs, ys, s=22, color=c, marker=m, edgecolor="white", linewidth=0.8, zorder=3,
                       label=name if g == 0 else None)
    for g, claim in enumerate(claims):
        ax.text(g, -1.05, f"Paper: {claim}", ha="center", color=INK2, fontsize=6.5, style="italic")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups); ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(-1.15, 1.95); ax.set_ylabel("Median gridness (annulus score)")
    ax.grid(axis="y", color=GRID, lw=0.5); ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save(fig, out, "figA3_ablations")
    print("Fig. A3: runs per group", {k: [len(v) for v in vals] for k, vals in data.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(ROOT / "runs"))
    p.add_argument("--archive", default=str(ROOT / "archive" / "runs_dev"))
    p.add_argument("--authors", default=str(ROOT.parent / "authors_runs"), help="folder holding authors_scores.json")
    p.add_argument("--out", default=str(ROOT / "figures"))
    a = p.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    fig_a1(Path(a.runs), out)
    fig_a2(Path(a.runs), Path(a.archive), out)
    fig_a3(Path(a.runs), a.authors, out)
    if MISSING:
        print("\nNot found or not scored (skipped):\n  " + "\n  ".join(dict.fromkeys(MISSING)))
    print("\nWritten to", out)