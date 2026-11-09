import argparse

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from euclidian_path import optimize, random_endpoints
from randers_path import constant_field, random_field, randers_cost

FORWARD, BACKWARD = "tab:blue", "tab:orange"


def parse_args():
    p = argparse.ArgumentParser(description="Animate forward/backward Randers paths in R^2.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-points", type=int, default=50)
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--frames", type=int, default=150, help="number of animation frames (history is subsampled)")
    p.add_argument("--save", type=str, default=None, help="save to .gif or .mp4 instead of showing")
    return p.parse_args()


def setup_panel(ax, title, field, ya, yb, histories, lo, hi):
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    # Field: |beta| as background, arrows for direction.
    gx, gy = torch.linspace(lo[0], hi[0], 200), torch.linspace(lo[1], hi[1], 200)
    X, Y = torch.meshgrid(gx, gy, indexing="xy")
    B = field(torch.stack([X.flatten(), Y.flatten()], dim=1)).norm(dim=1).reshape(X.shape)
    im = ax.imshow(B, origin="lower", extent=(lo[0], hi[0], lo[1], hi[1]), cmap="Greys", vmin=0, vmax=1, alpha=0.35)
    qx, qy = torch.linspace(lo[0], hi[0], 16), torch.linspace(lo[1], hi[1], 16)
    QX, QY = torch.meshgrid(qx, qy, indexing="xy")
    q = torch.stack([QX.flatten(), QY.flatten()], dim=1)
    V = field(q)
    ax.quiver(q[:, 0], q[:, 1], V[:, 0], V[:, 1], color="gray", alpha=0.6, width=0.003)

    ax.scatter(*ya.numpy(), s=80, color="black", zorder=4)
    ax.scatter(*yb.numpy(), s=80, color="black", marker="*", zorder=4)
    ax.annotate("$y_a$", ya.numpy(), xytext=(6, 6), textcoords="offset points")
    ax.annotate("$y_b$", yb.numpy(), xytext=(6, 6), textcoords="offset points")

    lines = {}
    for key, color, label in [("forward", FORWARD, r"$y_a \to y_b$"), ("backward", BACKWARD, r"$y_b \to y_a$")]:
        ax.plot(*histories[key == "backward"][0].T, color=color, alpha=0.25, lw=1)
        (line,) = ax.plot([], [], color=color, lw=2, label=label)
        dots = ax.scatter([], [], s=8, color=color, zorder=3)
        lines[key] = (line, dots)
    ax.legend(loc="lower right", fontsize=9)
    text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", family="monospace",
                   bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    return lines, text, im


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    ya, yb = random_endpoints(2)
    fields = {"constant": constant_field(2), "random": random_field(2)}
    kw = dict(n_points=args.n_points, noise=args.noise, steps=args.steps, lr=args.lr)

    runs = {}
    for name, field in fields.items():
        cost = randers_cost(field)
        fw, bw = optimize(ya, yb, cost, **kw), optimize(yb, ya, cost, **kw)
        runs[name] = dict(
            field=field,
            histories=(fw, bw),
            costs=([cost(y).item() for y in fw], [cost(y).item() for y in bw]),
        )

    n_steps = args.steps
    frame_idx = torch.logspace(0, torch.log10(torch.tensor(n_steps)), args.frames).long() - 1
    frame_idx = frame_idx.unique().tolist()

    # Shared bounds cover every iterate of every run so the view never jumps.
    pts = torch.cat([h.reshape(-1, 2) for run in runs.values() for h in run["histories"]])
    lo, hi = pts.min(0).values, pts.max(0).values
    pad = 0.1 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    fig.suptitle("Randers path", fontsize=14)

    panels = {}
    for ax, (name, run) in zip(axes, runs.items()):
        lines, text, im = setup_panel(ax, f"{name} field", run["field"], ya, yb, run["histories"], lo, hi)
        panels[name] = (lines, text)
    fig.colorbar(im, ax=axes, shrink=0.7, label=r"$\|\beta(x)\|$")

    def update(k):
        i = frame_idx[k]
        artists = []
        for name, run in runs.items():
            lines, text = panels[name]
            for key, h in zip(("forward", "backward"), run["histories"]):
                y = h[i].numpy()
                line, dots = lines[key]
                line.set_data(y[:, 0], y[:, 1])
                dots.set_offsets(y)
                artists += [line, dots]
            cf, cb = run["costs"][0][i], run["costs"][1][i]
            text.set_text(f"step {i:5d}\nF(a→b) = {cf:.4f}\nF(b→a) = {cb:.4f}")
            artists.append(text)
        return artists

    anim = FuncAnimation(fig, update, frames=len(frame_idx), interval=40, blit=False)

    if args.save:
        writer = PillowWriter(fps=25) if args.save.endswith(".gif") else None
        anim.save(args.save, writer=writer, dpi=100)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
