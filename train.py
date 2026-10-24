"""Train the conformal-isometry model with the paper's stated settings (§3.2, §3.4, App. A).

    python train.py --out runs/lam1                      # defaults: linear, s=10, lambda=1
    python train.py --lam 0.1 --out runs/lam0.1          # lambda sweep: one value per run
    python train.py --no-nonneg --out runs/abl_nonneg    # ablations: see flags below
"""
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import ConformalIsometry, sample_dx, sample_x

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--transform", default="linear", choices=["linear", "nonlinear1", "nonlinear2"])
p.add_argument("--d", type=int, default=None, help="neurons; default 24 (1000 for nonlinear2), §3.4")
p.add_argument("--K", type=int, default=16, help="heading bins (PAPER-SILENT)")
p.add_argument("--s", type=float, default=10.0, help="scaling factor, §3.4 uses 5, 10, 15")
p.add_argument("--lam", type=float, default=1.0, help="lambda in L = L1 + lambda*L2 (PAPER-SILENT)")
p.add_argument("--w_iso", type=float, default=1.0, help="weight on L1; 0 = 'remove conformal isometry' ablation")
p.add_argument("--D", type=float, default=1.25, help="L1 range s*||dx|| <= D, §3.2")
p.add_argument("--r_trans", type=float, default=0.075, help="L2 range ||dx|| <= 0.075, §3.2")
p.add_argument("--steps", type=int, default=200_000, help="App. A: 200,000 iterations")
p.add_argument("--batch", type=int, default=4000, help="App. A")
p.add_argument("--lr", type=float, default=0.003, help="App. A")
p.add_argument("--no-nonneg", dest="nonneg", action="store_false", help="ablation: skip Assumption 4")
p.add_argument("--no-normalize", dest="normalize", action="store_false", help="ablation: skip Assumption 3")
p.add_argument("--seed", type=int, default=0)
# Alternatives to the three sampling choices that match the authors' code
p.add_argument("--radius_uniform", action="store_true", help="||dx|| uniform in length, not over the disc")
p.add_argument("--continuous_heading", action="store_true", help="L2 headings continuous, rounded to nearest bin")
p.add_argument("--v_init", default="uniform", choices=["uniform", "small"])
p.add_argument("--B_init", default="he", choices=["he", "small"])
p.add_argument("--lattice", default="corners", choices=["corners", "centres"])
# The authors' training choices (from their released code), for the controlled comparison
p.add_argument("--lr_schedule", default="constant", choices=["constant", "warmup_decay", "gao_step"],
               help="warmup_decay: linear warmup over 3k steps, constant to 6k, linear decay to 0")
p.add_argument("--grad_clip", type=float, default=None, help="max gradient norm (authors: 10); 0 = none")
p.add_argument("--heading_interp", action="store_true", help="interpolate B between neighbouring heading bins (predecessor convention)")
p.add_argument("--loss_scale", type=float, default=1.0,
               help="multiplies the whole loss; only matters with grad_clip (authors' effective scale: 40000)")
p.add_argument("--leave_square", action="store_true", help="x uniform over the square; x+dx may leave it (border padding)")
p.add_argument("--log_every", type=int, default=500)
args = p.parse_args()

torch.manual_seed(args.seed)
d = args.d or (1000 if args.transform == "nonlinear2" else 24)
model = ConformalIsometry(d=d, K=args.K, transform=args.transform,
                          padding="border" if args.leave_square else "zeros",
                          v_init=args.v_init, B_init=args.B_init, lattice=args.lattice,
                          heading_interp=args.heading_interp)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)          # §3.4: Adam; App. A: lr 0.003, constant
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
history, t0 = [], time.time()

for step in range(1, args.steps + 1):
    n = args.batch
    # L1 batch: continuous heading, ||dx|| up to D/s (non-infinitesimal range)
    _, _, dx1 = sample_dx(n, args.D / args.s, radius_uniform=args.radius_uniform)
    x1 = sample_x(dx1, keep_inside=not args.leave_square)
    # L2 batch: heading on the K bin centres (PAPER-SILENT; B is only defined there), ||dx|| <= 0.075
    if args.continuous_heading:
        theta2 = torch.rand(n) * 2 * math.pi
    else:
        theta2 = torch.randint(0, args.K, (n,)).float() * (2 * math.pi / args.K)
    theta2, dr2, dx2 = sample_dx(n, args.r_trans, theta=theta2, radius_uniform=args.radius_uniform)
    x2 = sample_x(dx2, keep_inside=not args.leave_square)

    L1 = model.loss_isometry(x1, dx1, args.s) if args.w_iso else torch.zeros(())
    L2 = model.loss_transform(x2, theta2, dr2) if args.lam else torch.zeros(())
    loss = args.w_iso * L1 + args.lam * L2

    if args.lr_schedule == "warmup_decay":
        if step < 3000:
            lr = args.lr * step / 3000
        elif step > 6000:
            lr = args.lr * max(0.0, 1 - (step - 6000) / (args.steps - 6000))
        else:
            lr = args.lr
        for g in opt.param_groups:
            g["lr"] = lr
    elif args.lr_schedule == "gao_step":   # Gao et al. (2021): 0.003, halved every 500 iterations after 8,000
        lr = args.lr * 0.5 ** max(0, (step - 8000) // 500 + 1) if step > 8000 else args.lr
        for g in opt.param_groups:
            g["lr"] = lr

    opt.zero_grad()
    (args.loss_scale * loss).backward()
    if args.grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    opt.step()
    model.project(args.nonneg, args.normalize)                  # §3.2: projections after each iteration

    if step % args.log_every == 0 or step == 1:
        rec = dict(step=step, L1=L1.item(), L2=L2.item(), t=round(time.time() - t0))
        history.append(rec)
        print(f"step {step:7d}  L1 {rec['L1']:.3e}  L2 {rec['L2']:.3e}  {rec['t']}s", flush=True)

# ------------------------------------------------------------------- outputs
maps = model.v.detach().numpy()                                 # (d, G, G)
torch.save({"args": vars(args), "state_dict": model.state_dict()}, out / "checkpoint.pt")
(out / "history.json").write_text(json.dumps(history, indent=1))

summary = {"args": vars(args), "final_L1": history[-1]["L1"], "final_L2": history[-1]["L2"]}
try:   # gridness and scale via NeuralPlayground, if installed
    from neuralplayground.comparison.metrics import GridScorer
    scorer, grid, scale = GridScorer(model.G), [], []
    for m in maps[:24]:
        try:
            _, props = scorer.get_scores(m.T.astype(float))
            grid.append(float(props["gridscore"]))
            scale.append(float(props["scale"]) / (model.G - 1 if args.lattice == "corners" else model.G))    # bins -> metres
        except Exception:
            grid.append(float("nan")); scale.append(float("nan"))
    summary.update(gridness_median=float(np.nanmedian(grid)), frac_gridness_gt_0=float(np.mean(np.array(grid) > 0)),
                   scale_median_m=float(np.nanmedian(scale)), s_times_scale=float(args.s * np.nanmedian(scale)),
                   gridness=grid, scale_m=scale)
except ImportError:
    grid = None
(out / "summary.json").write_text(json.dumps(summary, indent=1))

fig, axes = plt.subplots(4, 6, figsize=(10, 7))
for k, ax in enumerate(axes.ravel()):
    ax.imshow(maps[k].T, origin="lower", cmap="jet")           # x horizontal, y vertical
    ax.axis("off")
    if grid:
        ax.set_title(f"{grid[k]:.2f}", fontsize=8)
fig.suptitle(f"{args.transform}, s={args.s}, lambda={args.lam}, {args.steps} steps", fontsize=10)
fig.tight_layout()
fig.savefig(out / "rate_maps.png", dpi=120)
print(json.dumps({k: v for k, v in summary.items() if k not in ("gridness", "scale_m", "args")}, indent=1))
