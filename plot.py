from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .env import make_env
except ImportError:  # Direct script execution from the repository root.
    from env import make_env


def read_metrics(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if value == "":
                    parsed[key] = np.nan
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            rows.append(parsed)
        return rows


def series(rows: list[dict[str, float]], key: str) -> tuple[np.ndarray, np.ndarray]:
    return np.array([r["global_step"] for r in rows]), np.array([r[key] for r in rows])


def style(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#dddddd", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_metrics(run_dir: Path) -> None:
    rows = read_metrics(run_dir / "metrics.csv")
    out = run_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    colors = {"easy": "#009E73", "medium": "#0072B2", "hard": "#D55E00", "mean": "#222222"}
    x, y = series(rows, "eval_safe_success_mean")
    axes[0, 0].plot(x, y, lw=3, color=colors["mean"], label="mean")
    for difficulty in ("easy", "medium", "hard"):
        x, y = series(rows, f"eval_{difficulty}_safe_success")
        axes[0, 0].plot(x, y, lw=2, color=colors[difficulty], label=difficulty)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Safe Success")
    style(axes[0, 0], "safe success rate")
    axes[0, 0].legend(frameon=False)

    x, y = series(rows, "eval_collision_mean")
    axes[0, 1].plot(x, y, lw=3, color=colors["mean"], label="mean")
    for difficulty in ("easy", "medium", "hard"):
        x, y = series(rows, f"eval_{difficulty}_collision")
        axes[0, 1].plot(x, y, lw=2, color=colors[difficulty], label=difficulty)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("Collision Episodes")
    style(axes[0, 1], "collision episode rate")

    x, y = series(rows, "eval_min_distance_mean")
    axes[1, 0].plot(x, y, lw=3, color=colors["mean"], label="mean")
    for difficulty in ("easy", "medium", "hard"):
        x, y = series(rows, f"eval_{difficulty}_min_distance")
        axes[1, 0].plot(x, y, lw=2, color=colors[difficulty], label=difficulty)
    axes[1, 0].set_title("Closest Distance To Target")
    style(axes[1, 0], "mean min distance")

    x, train_safe = series(rows, "train_safe_success_rate")
    _, train_collision = series(rows, "train_collision_episode_rate")
    axes[1, 1].plot(x, train_safe, lw=2.5, color="#009E73", label="train safe success")
    axes[1, 1].plot(x, train_collision, lw=2.5, color="#D55E00", label="train collision")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Recent Training Episodes")
    style(axes[1, 1], "rate")
    axes[1, 1].legend(frameon=False)

    fig.suptitle("Randomized Reacher PPO Baseline")
    fig.tight_layout()
    fig.savefig(out / "baseline_metrics.png", dpi=220)
    plt.close(fig)


def plot_geometry(run_dir: Path, seed: int = 123) -> None:
    out = run_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    for ax, difficulty in zip(axes, ("easy", "medium", "hard")):
        env = make_env(difficulty=difficulty)
        for idx in range(18):
            env.reset(seed=seed + idx)
            base, elbow, hand = env._points()
            ax.plot([base[0], elbow[0], hand[0]], [base[1], elbow[1], hand[1]], color="#999999", lw=1, alpha=0.35)
            ax.scatter([env.target[0]], [env.target[1]], color="#009E73", s=18, alpha=0.8)
            ax.add_patch(plt.Circle(env.obstacle, env.obstacle_radius, color="#D55E00", alpha=0.18))
        ax.set_title(f"{difficulty.title()} Samples")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#e5e5e5")
        ax.set_xlabel("x")
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(-0.25, 1.25)
    axes[0].set_ylabel("y")
    fig.suptitle("Randomized Task Geometry")
    fig.tight_layout()
    fig.savefig(out / "task_geometry_samples.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    plot_metrics(run_dir)
    plot_geometry(run_dir)


if __name__ == "__main__":
    main()
