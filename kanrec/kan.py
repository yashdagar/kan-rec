import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BASES = ("bspline", "rbf", "cheby", "rational")


def bspline_basis(x, grid, order):
    xv = x.unsqueeze(-1)
    b = ((xv >= grid[:-1]) & (xv < grid[1:])).to(x.dtype)
    for k in range(1, order + 1):
        left = (xv - grid[: -(k + 1)]) / (grid[k:-1] - grid[: -(k + 1)])
        right = (grid[k + 1:] - xv) / (grid[k + 1:] - grid[1:-k])
        b = left * b[..., :-1] + right * b[..., 1:]
    return b


class KANLinear(nn.Module):
    """y = W_base silu(x) + sum_i phi_{o,i}(x_i), with phi from one of four bases."""

    def __init__(self, in_features, out_features, basis="bspline", grid_size=8, spline_order=3,
                 grid_range=(-2.0, 2.0), rational_groups=8):
        super().__init__()
        if basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}")
        self.in_features, self.out_features = in_features, out_features
        self.basis, self.grid_size, self.spline_order = basis, grid_size, spline_order
        self.base = nn.Linear(in_features, out_features)

        if basis == "bspline":
            lo, hi = grid_range
            h = (hi - lo) / grid_size
            grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + lo
            self.register_buffer("grid", grid)
            self.n_basis = grid_size + spline_order
        elif basis == "rbf":
            lo, hi = grid_range
            self.register_buffer("centers", torch.linspace(lo, hi, grid_size))
            self.inv_width = (grid_size - 1) / (hi - lo)
            self.n_basis = grid_size
        elif basis == "cheby":
            self.n_basis = grid_size + 1
        else:
            if in_features % rational_groups:
                raise ValueError("in_features must be divisible by rational_groups")
            self.groups = rational_groups
            num = torch.zeros(rational_groups, 6)
            num[:, 1] = 1.0
            self.rat_num = nn.Parameter(num)
            self.rat_den = nn.Parameter(torch.zeros(rational_groups, 4))
            self.n_basis = 0

        if self.n_basis:
            self.coef = nn.Parameter(torch.empty(out_features, in_features, self.n_basis))
            nn.init.normal_(self.coef, std=0.1 / math.sqrt(in_features * self.n_basis))
        else:
            self.rat_linear = nn.Linear(in_features, out_features, bias=False)

    def expand(self, x):
        if self.basis == "bspline":
            return bspline_basis(x, self.grid, self.spline_order)
        if self.basis == "rbf":
            return torch.exp(-((x.unsqueeze(-1) - self.centers) * self.inv_width) ** 2)
        if self.basis == "cheby":
            t = torch.tanh(x)
            polys = [torch.ones_like(t), t]
            for _ in range(2, self.n_basis):
                polys.append(2 * t * polys[-1] - polys[-2])
            return torch.stack(polys[: self.n_basis], dim=-1)
        raise RuntimeError("rational basis has no expansion")

    def rational(self, x):
        shape = x.shape
        xg = x.reshape(-1, self.groups, self.in_features // self.groups)
        powers = torch.stack([xg ** p for p in range(6)], dim=-1)
        num = (powers * self.rat_num[None, :, None, :]).sum(-1)
        den = 1.0 + (powers[..., 1:5] * self.rat_den[None, :, None, :]).sum(-1).abs()
        return (num / den).reshape(shape)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_features)
        out = self.base(F.silu(x))
        if self.basis == "rational":
            out = out + self.rat_linear(self.rational(x))
        else:
            phi = self.expand(x)
            out = out + phi.reshape(x.shape[0], -1) @ self.coef.reshape(self.out_features, -1).T
        return out.reshape(*shape[:-1], self.out_features)

    @torch.no_grad()
    def edge_function(self, out_idx, in_idx, xs):
        if self.basis == "rational":
            raise ValueError("rational activations are per group, not per edge")
        phi = self.expand(xs.unsqueeze(-1).expand(-1, self.in_features))[:, in_idx]
        spline = phi @ self.coef[out_idx, in_idx]
        return spline + self.base.weight[out_idx, in_idx] * F.silu(xs)


def n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.SiLU(), nn.Linear(hidden, out_features))

    def forward(self, x):
        return self.net(x)


def matched_mlp(in_features, out_features, target_params):
    hidden = max(1, round((target_params - out_features) / (in_features + out_features + 1)))
    return MLP(in_features, out_features, hidden)


class ResidualBlock(nn.Module):
    """x + f(LayerNorm(x)); f is a KANLinear or a parameter-matched MLP. With zero_init the output
    layer of f starts at zero, so the block starts as the identity."""

    def __init__(self, dim, kind="kan", basis="rbf", grid_size=8, match_params=None, zero_init=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if kind == "kan":
            self.f = KANLinear(dim, dim, basis=basis, grid_size=grid_size)
        elif kind == "mlp":
            target = match_params if match_params is not None else n_params(KANLinear(dim, dim, basis=basis, grid_size=grid_size))
            self.f = matched_mlp(dim, dim, target)
        else:
            raise ValueError(kind)
        if zero_init:
            with torch.no_grad():
                last = self.f if kind == "kan" else self.f.net[-1]
                for name, p in last.named_parameters():
                    if name in ("coef", "base.weight", "base.bias", "rat_linear.weight", "weight", "bias"):
                        p.zero_()

    def forward(self, x):
        return x + self.f(self.norm(x))
