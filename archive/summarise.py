"""Print one row per run: final losses, median gridness, fraction > 0, s x scale."""
import json, sys
from pathlib import Path
rows = []
for f in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "runs").glob("*/summary.json")):
    s = json.loads(f.read_text())
    rows.append((f.parent.name, s["final_L1"], s["final_L2"], s.get("gridness_median"), s.get("frac_gridness_gt_0"), s.get("s_times_scale")))
print(f"{'run':26s} {'L1':>8s} {'L2':>8s} {'grid':>6s} {'>0':>5s} {'s*sc':>5s}")
for r in rows:
    print(f"{r[0]:26s} {r[1]:8.4f} {r[2]:8.4f} {r[3]:6.2f} {r[4] if r[4] is not None else float('nan'):5.2f} {r[5]:5.2f}")
