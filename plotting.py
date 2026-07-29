"""
Bar + per-participant-dot figures, matching the PDF's style. Every figure
is displayed (plt.show()) and then saved to config.DIAGRAMS_DIR.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def _save_and_show(fig, filename: str):
    path = config.DIAGRAMS_DIR / filename
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"saved: {path}")


def bar_with_dots(
    values_by_level: dict[str, pd.Series],
    title: str,
    ylabel: str = "Pearson R",
    filename: str | None = None,
    dot_label: str | None = None,
):
    """
    values_by_level: {level_label: Series of per-participant R values
    (or a length-1 Series for a single AVG-vector R)}.
    """
    levels = list(values_by_level.keys())
    means = [values_by_level[lv].mean() for lv in levels]

    fig, ax = plt.subplots(figsize=(1.6 * len(levels) + 1, 4.5))
    ax.bar(levels, means, color="lightgray", edgecolor="black", zorder=2)

    rng = np.random.default_rng(0)
    for x, lv in enumerate(levels):
        vals = values_by_level[lv].values
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, color="black", s=18, zorder=3, alpha=0.8)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.set_title(title)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, rotation=30, ha="right")
    fig.tight_layout()

    if filename:
        _save_and_show(fig, filename)
    else:
        plt.show()
        plt.close(fig)


def bar_with_noise_ceiling(
    values_by_level: dict[str, pd.Series],
    noise_ceiling_by_level: dict[str, pd.Series],
    title: str,
    ylabel: str = "Pearson R",
    filename: str | None = None,
):
    """Same as bar_with_dots, plus a faint noise-ceiling bar/dots per level."""
    levels = list(values_by_level.keys())
    model_means = [values_by_level[lv].mean() for lv in levels]
    ceiling_means = [noise_ceiling_by_level[lv].mean() for lv in levels]

    x = np.arange(len(levels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(1.8 * len(levels) + 1, 4.5))
    ax.bar(x - width / 2, model_means, width, color="lightgray", edgecolor="black", label="Model", zorder=2)
    ax.bar(x + width / 2, ceiling_means, width, color="lightblue", edgecolor="black", label="Noise ceiling", zorder=2)

    rng = np.random.default_rng(0)
    for i, lv in enumerate(levels):
        mv = values_by_level[lv].values
        cv = noise_ceiling_by_level[lv].values
        ax.scatter(np.full(len(mv), i - width / 2) + rng.uniform(-0.06, 0.06, len(mv)), mv, color="black", s=14, zorder=3, alpha=0.8)
        ax.scatter(np.full(len(cv), i + width / 2) + rng.uniform(-0.06, 0.06, len(cv)), cv, color="navy", s=14, zorder=3, alpha=0.8)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()

    if filename:
        _save_and_show(fig, filename)
    else:
        plt.show()
        plt.close(fig)
