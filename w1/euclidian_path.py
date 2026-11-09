import torch


def random_endpoints(dim):
    return torch.randn(dim), torch.randn(dim)


def straight_path(ya, yb, n_points):
    t = torch.linspace(0, 1, n_points)[:, None]  # n,1
    return ya + t * (yb - ya)


def init_arbitrary_path(ya, yb, n_points, noise):
    inner = straight_path(ya, yb, n_points)[1:-1]
    return (inner + noise * torch.randn_like(inner)).requires_grad_()


def discrete_length(y):
    return (y[1:] - y[:-1]).norm(dim=1).sum()  # shifted by 1 -- i & i+1


def optimize(ya, yb, cost=discrete_length, n_points=50, noise=0.3, steps=2000, lr=1e-2):
    inner = init_arbitrary_path(ya, yb, n_points, noise)
    opt = torch.optim.Adam([inner], lr=lr)
    history = []
    for _ in range(steps):
        y = torch.cat([ya[None], inner, yb[None]])  # -- (n, d) turn 1,2 (leading axis)
        loss = cost(y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(y.detach().clone())
    return torch.stack(history)
