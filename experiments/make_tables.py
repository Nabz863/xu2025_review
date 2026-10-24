"""Collect run summaries into the three result tables (markdown) used in RESULTS.md.

    python experiments/make_tables.py > experiments/tables.md

A run is "hexagonal" if its median annulus gridness exceeds 0.37 (Banino et al. 2018 threshold,
as used by Gao et al. 2021 and Xu et al. 2022). Valid % = cells with annulus gridness > 0.37.
"""
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent.parent / "runs"


def load(name):
    f = RUNS / name / "summary.json"
    if not f.exists():
        return None
    s = json.loads(f.read_text())
    ann = np.array(s.get("gridness_annulus", [np.nan]), dtype=float)
    npg = np.array(s.get("gridness", [np.nan]), dtype=float)
    return dict(L1=s["final_L1"], ann=np.nanmedian(ann), npg=np.nanmedian(npg),
                valid=100 * np.mean(ann > 0.37) if np.isfinite(ann).any() else np.nan,
                sscale=s.get("s_times_scale", np.nan), s=s["args"]["s"])


def fmt(x, nd=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def group_row(label, names):
    rs = [load(n) for n in names]
    done = [r for r in rs if r]
    if not done:
        return f"| {label} | not run | | | | |"
    hexn = sum(r["ann"] > 0.37 for r in done)
    per_seed = " / ".join(fmt(r["ann"]) if r else "…" for r in rs)
    return (f"| {label} | **{hexn} / {len(done)}** | {per_seed} | "
            f"{fmt(np.mean([r['valid'] for r in done]), 0)} | {fmt(np.mean([r['L1'] for r in done]) * 1e3)} | "
            f"{fmt(np.mean([r['sscale'] for r in done]))} |")


HEAD = "| Condition | Hexagonal runs | Annulus gridness per seed | Mean valid % | Mean final L1 (×10⁻³) | s × scale |\n|---|---|---|---|---|---|"
THEIRS = ["ctrl_all_theirs", "theirs_seed1", "theirs_seed2", "theirs_seed3"]

print("## Part 1. Reconstruction from the paper and its references\n")
print(HEAD)
print(group_row("Reconstruction (seeds 0–5)", [f"A_recon_s{i}" for i in range(6)]))
print(group_row("Paper text only, no reference conventions (seeds 0–5)", ["base_seed0", "base_seed1", "base_seed2", "base_seed3", "base_seed4", "base_seed5"]))
print(group_row("Authors' configuration (seeds 0–3)", THEIRS))

print("\n## Part 2. Authors' configuration, one silent choice changed at a time (seeds 0, 1)\n")
print(HEAD)
print(group_row("Authors' configuration, unchanged", THEIRS[:2]))
for key, label in [("headings", "Headings: 144 bins, interpolated, continuous (vs 18 bins, nearest)"),
                   ("vinit", "v init: uniform, projected (vs N(0, 0.001))"),
                   ("Binit", "B init: He (vs N(0, 0.001))"),
                   ("lattice", "Lattice points on the walls (vs cell centres)"),
                   ("dxlength", "Δx uniform in length (vs uniform over the disc)"),
                   ("boundary", "x + Δx may leave the square (vs kept inside)"),
                   ("lam1", "λ = 1 (vs 0.75)"),
                   ("noclip", "No gradient clipping (vs max-norm 10)")]:
    print(group_row(label, [f"B_{key}_s0", f"B_{key}_s1"]))

print("\n## Part 3. Where the code departs from what the paper states (authors' configuration, seeds 0–2)\n")
print(HEAD)
print(group_row("Authors' configuration, as released", THEIRS[:3]))
print(group_row("Learning rate constant 0.003, as the paper states (vs warmup + decay)", [f"C_lrconst_s{i}" for i in range(3)]))
print(group_row("Losses as plain expectations, as eqs. 3–4 state (vs L1 summed, L2 ×30000)", [f"C_lossmean_s{i}" for i in range(3)]))
print(group_row("s = 15 with the code's range s‖Δx‖ ≤ 1.125", ["C_s15code_s0", "C_s15code_s1"]))
print(group_row("s = 15 with the paper's range s‖Δx‖ ≤ 1.25", ["C_s15paper_s0", "C_s15paper_s1"]))
print(group_row("s = 5 (Table 2 check)", ["C_s5_s0"]))

print("\n## Part 4. The paper's ablations (Fig. 3), on the reconstruction and the authors' configuration (seeds 0, 1)\n")
print("Paper's claims: removing L1 gives non-hexagonal patterns (Fig. 3h); removing L2 or normalisation disrupts the patterns (3f, 3g); removing non-negativity leaves hexagons (3a, 3e).\n")
print(HEAD)
for base, label in [("recon", "Reconstruction"), ("theirs", "Authors' configuration")]:
    for key, what in [("noL1", "no conformal isometry loss (L1)"), ("noL2", "no transformation loss (L2)"),
                      ("nonorm", "no normalisation"), ("nononneg", "no non-negativity")]:
        print(group_row(f"{label}, {what}", [f"D_{base}_{key}_s0", f"D_{base}_{key}_s1"]))
