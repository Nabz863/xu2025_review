"""Conformal-isometry grid-cell model, implemented from the paper:
Xu, Gao, Zhang, Wei & Wu (2025), "On Conformal Isometry of Grid Cells", ICLR 2025.

Positions are in metres in the 1 m x 1 m square [0, 1]^2.
v[:, i, j] is the embedding at lattice point x-index i, y-index j.
Choices the paper does not specify are marked PAPER-SILENT and logged in decisions.md.
"""
import math

import torch
import torch.nn.functional as F


class ConformalIsometry(torch.nn.Module):
    def __init__(self, d=24, G=40, K=16, transform="linear", padding="zeros",
                 v_init="uniform", B_init="he", lattice="corners", heading_interp=False):
        super().__init__()
        self.d, self.G, self.K, self.kind = d, G, K, transform
        self.padding = padding  # only matters if positions leave the square (see sample_x)
        self.heading_interp = heading_interp  # True: B(theta) linearly interpolated between neighbouring bins
        self.lattice = lattice  # "corners": points on the walls; "centres": points at cell centres
        he = math.sqrt(2 / d) if B_init == "he" else 0.001

        # v: one d-vector per lattice point (§3.2). PAPER-SILENT init: uniform positive, projected below.
        if v_init == "uniform":
            self.v = torch.nn.Parameter(torch.rand(d, G, G))
        else:  # alternative: tiny zero-centred, not projected before training
            self.v = torch.nn.Parameter(torch.randn(d, G, G) * 0.001)

        # B(theta): one parameter set per heading bin. PAPER-SILENT: K bins, He init.
        if transform in ("linear", "nonlinear1"):
            self.B = torch.nn.Parameter(torch.randn(K, d, d) * he)   # d x d (§3.1)
        else:
            self.B = torch.nn.Parameter(torch.randn(K, d) * he)      # d x 1 for nonlinear 2 (§3.1)
        if transform != "linear":
            self.A = torch.nn.Parameter(torch.randn(d, d) * he)      # d x d (§3.1)
            self.b = torch.nn.Parameter(torch.zeros(d))              # bias vector (§3.1)

        if v_init == "uniform":
            self.project()  # start inside Assumptions 3 and 4

    # ---------------------------------------------------------- Assumptions 3, 4
    @torch.no_grad()
    def project(self, nonneg=True, normalize=True):
        """§3.2: after each iteration, zero negatives, then normalise ||v(x)|| = 1."""
        if nonneg:
            self.v.clamp_(min=0)
        if normalize:
            # norm over dim 0 = across the d neurons at each lattice point
            self.v.div_(self.v.norm(dim=0, keepdim=True).clamp(min=1e-8))

    # ------------------------------------------------------- bilinear lookup
    def lookup(self, pos):
        """v(x) for continuous positions. pos: (N, 2) in metres. Returns (N, d).
        PAPER-SILENT: lattice corners sit on the walls (spacing 1/(G-1)), so align_corners=True."""
        g = (pos * 2 - 1).flip(-1).view(1, 1, -1, 2)   # metres -> [-1,1]; (x,y) -> (y,x) for grid_sample
        # corners: -1/+1 = first/last lattice point. centres: -1/+1 = outer edges of the first/last cell.
        out = F.grid_sample(self.v.unsqueeze(0), g, mode="bilinear", align_corners=(self.lattice == "corners"),
                            padding_mode=self.padding)
        return out[0, :, 0, :].T

    # ------------------------------------------------- Assumption 2: F(v, dx)
    def heading_bin(self, theta):
        """Nearest of the K heading bins; % K wraps theta near 2*pi back to bin 0."""
        return torch.round(theta / (2 * math.pi / self.K)).long() % self.K

    def transform(self, v_x, theta, dr):
        """v(x + dx) predicted from v(x). v_x: (N, d), theta: (N,), dr: (N,) metres."""
        if self.heading_interp:
            # Gao et al. (2021), Xu et al. (2022): discretise directions, interpolate linearly between neighbours
            pos = theta / (2 * math.pi / self.K)
            k0 = torch.floor(pos).long() % self.K
            w = (pos - torch.floor(pos)).view(-1, *([1] * (self.B.dim() - 1)))
            Bt = (1 - w) * self.B[k0] + w * self.B[(k0 + 1) % self.K]
        else:
            Bt = self.B[self.heading_bin(theta)]
        dr = dr[:, None]
        if self.kind == "linear":       # v + B(theta) v dr
            return v_x + torch.einsum("nij,nj->ni", Bt, v_x) * dr
        if self.kind == "nonlinear1":   # ReLU(A v + B(theta) v dr + b)
            return F.relu(v_x @ self.A.T + torch.einsum("nij,nj->ni", Bt, v_x) * dr + self.b)
        return F.relu(v_x @ self.A.T + Bt * dr + self.b)   # nonlinear2: ReLU(A v + B(theta) dr + b)

    # ---------------------------------------------------------------- losses
    def loss_isometry(self, x, dx, s):
        """L1 (eq. 3) = E[(||v(x+dx) - v(x)|| - s||dx||)^2], as a Monte Carlo average."""
        dv = (self.lookup(x + dx) - self.lookup(x)).norm(dim=1)
        return ((dv - s * dx.norm(dim=1)) ** 2).mean()

    def loss_transform(self, x, theta, dr):
        """L2 (eq. 4) = E[||v(x+dx) - F(v(x), dx)||^2], as a Monte Carlo average."""
        dx = dr[:, None] * torch.stack([theta.cos(), theta.sin()], dim=1)
        pred = self.transform(self.lookup(x), theta, dr)
        return ((self.lookup(x + dx) - pred) ** 2).sum(dim=1).mean()


# ------------------------------------------------------------------ sampling
def sample_dx(n, r_max, theta=None, radius_uniform=False):
    """dx uniform over the disc of radius r_max (PAPER-SILENT: 'uniform' read as uniform in area,
    so ||dx|| = r_max * sqrt(U)). radius_uniform=True instead draws ||dx|| = r_max * U.
    Returns theta, dr, dx."""
    if theta is None:
        theta = torch.rand(n) * 2 * math.pi
    u = torch.rand(n)
    dr = r_max * (u if radius_uniform else u.sqrt())
    dx = dr[:, None] * torch.stack([theta.cos(), theta.sin()], dim=1)
    return theta, dr, dx


def sample_x(dx, keep_inside=True):
    """x uniform over positions where both x and x+dx lie inside [0,1]^2 (PAPER-SILENT).
    keep_inside=False draws x uniformly over the square and lets x+dx leave it; the lookup
    then uses the model's padding mode ("border" = value at the nearest wall)."""
    if not keep_inside:
        return torch.rand_like(dx)
    lo = (-dx).clamp(min=0)
    hi = (1 - dx).clamp(max=1)
    return lo + torch.rand_like(dx) * (hi - lo)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = ConformalIsometry()
    v = m.v.detach()
    h = 1 / 39
    print("on-lattice:", torch.allclose(m.lookup(torch.tensor([[3 * h, 7 * h]]))[0], v[:, 3, 7], atol=1e-6))
    print("midpoint:  ", torch.allclose(m.lookup(torch.tensor([[3.5 * h, 7 * h]]))[0],
                                        0.5 * (v[:, 3, 7] + v[:, 4, 7]), atol=1e-6))
    n = m.v.norm(dim=0)
    print("norms all 1:", torch.allclose(n, torch.ones_like(n)), "| min v >= 0:", bool((m.v >= 0).all()))
    _, _, dx = sample_dx(5000, 0.125)
    x = sample_x(dx)
    inside = ((x >= 0) & (x <= 1) & (x + dx >= -1e-6) & (x + dx <= 1 + 1e-6)).all()
    print("samples stay inside:", bool(inside))
    print("heading wrap:", m.heading_bin(torch.tensor([0.0, 2 * math.pi - 1e-4])).tolist(), "(expect [0, 0])")
