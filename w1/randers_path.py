import math

import torch
from euclidian_path import optimize, random_endpoints


def randers_cost(field):
    def cost(y):
        v = y[1:] - y[:-1]
        mid = 0.5 * (y[1:] + y[:-1])
        return (v.norm(dim=1) + (field(mid) * v).sum(dim=1)).sum()

    return cost


def constant_field(dim, max_norm=0.5):
    w = torch.randn(dim)
    w = max_norm * w / w.norm()
    return lambda x: w.expand_as(x)


def random_field(dim, n_modes=4, freq=1.0, max_norm=0.8):
    W = freq * torch.randn(n_modes, dim)
    phi = 2 * math.pi * torch.rand(n_modes)
    A = torch.randn(n_modes, dim)
    A = max_norm * A / A.norm(dim=1).sum()
    return lambda x: torch.sin(x @ W.T + phi) @ A


def sample_field(field, lo=-3.0, hi=3.0, n=20):
    g = torch.linspace(lo, hi, n)
    X, Y = torch.meshgrid(g, g, indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
    return pts, field(pts)


def main(seed=0):
    torch.manual_seed(seed)
    ya, yb = random_endpoints(2)
    fields = {"constant": constant_field(2), "random": random_field(2)}
    results = {}
    grids = {}
    for name, field in fields.items():
        cost = randers_cost(field)
        forward = optimize(ya, yb, cost)
        backward = optimize(yb, ya, cost)
        results[name] = {"forward": forward, "backward": backward}
        grids[name] = sample_field(field)
        print(
            name,
            "forward:",
            cost(forward[-1]).item(),
            "backward:",
            cost(backward[-1]).item(),
        )
    torch.save(
        {"ya": ya, "yb": yb, "results": results, "grids": grids}, "randers_paths.pt"
    )


if __name__ == "__main__":
    main()
