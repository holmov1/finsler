import argparse

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from euclidian_path import discrete_length, optimize, random_endpoints


def parse_args():
    p = argparse.ArgumentParser(description="Animate a discrete path shrinking to a straight line in R^2.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-points", type=int, default=50)
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--frames", type=int, default=150, help="number of animation frames (history is subsampled)")
    p.add_argument("--save", type=str, default=None, help="save to .gif or .mp4 instead of showing")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    ya, yb = random_endpoints(2)
    history = optimize(ya, yb, n_points=args.n_points, noise=args.noise, steps=args.steps, lr=args.lr)
    lengths = [discrete_length(y).item() for y in history]
    history = history.numpy()
    target = (yb - ya).norm().item()

    # Geometric spacing: most of the motion happens in the first few hundred steps.
    frame_idx = torch.logspace(0, torch.log10(torch.tensor(len(history))), args.frames).long() - 1
    frame_idx = frame_idx.unique().tolist()

    fig, ax_plane = plt.subplots(figsize=(8, 6))
    ax_plane.set_title("Euclidean path by length minimisation", fontsize=13)

    # Plane: bounds cover every iterate so the view never jumps.
    pts = history.reshape(-1, 2)
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.1 * (hi - lo).max()
    ax_plane.set_xlim(lo[0] - pad, hi[0] + pad)
    ax_plane.set_ylim(lo[1] - pad, hi[1] + pad)
    ax_plane.set_aspect("equal")
    ax_plane.grid(alpha=0.25)
    ax_plane.set_xlabel("$x_1$")
    ax_plane.set_ylabel("$x_2$")

    ax_plane.plot(*zip(ya.numpy(), yb.numpy()), "--", color="gray", lw=1, label="straight line")
    ax_plane.plot(*history[0].T, color="tab:red", alpha=0.35, lw=1, label="initial path")
    trail = LineCollection([], cmap="Blues", lw=0.8)
    ax_plane.add_collection(trail)
    (path_line,) = ax_plane.plot([], [], color="tab:blue", lw=2, label="current path")
    path_pts = ax_plane.scatter([], [], s=10, color="tab:blue", zorder=3)
    ax_plane.scatter(*ya.numpy(), s=80, color="black", zorder=4)
    ax_plane.scatter(*yb.numpy(), s=80, color="black", marker="*", zorder=4)
    ax_plane.annotate("$y_a$", ya.numpy(), xytext=(6, 6), textcoords="offset points")
    ax_plane.annotate("$y_b$", yb.numpy(), xytext=(6, 6), textcoords="offset points")
    ax_plane.legend(loc="best", fontsize=9)
    step_text = ax_plane.text(0.02, 0.97, "", transform=ax_plane.transAxes, va="top", family="monospace")

    def update(k):
        i = frame_idx[k]
        y = history[i]
        path_line.set_data(y[:, 0], y[:, 1])
        path_pts.set_offsets(y)

        past = [history[j] for j in frame_idx[: k + 1 : 10]]
        trail.set_segments(past)
        trail.set_array(torch.linspace(0.2, 0.6, len(past)).numpy())
        trail.set_alpha(0.4)

        step_text.set_text(f"step {i:5d}\nL = {lengths[i]:.4f}\nL* = {target:.4f}")
        return path_line, path_pts, trail, step_text

    anim = FuncAnimation(fig, update, frames=len(frame_idx), interval=40, blit=False)
    fig.tight_layout()

    if args.save:
        writer = PillowWriter(fps=25) if args.save.endswith(".gif") else None
        anim.save(args.save, writer=writer, dpi=110)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
