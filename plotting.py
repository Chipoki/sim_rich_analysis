"""
Bar + per-participant-dot figures, matching the PDF's style. Every figure
is displayed (plt.show()) and then saved to config.DIAGRAMS_DIR.

Every figure always shows: a numeric value label on each bar, a reference
line (zero by default, overridable per call), the per-level sample size
baked directly into its x-tick label, and a legend spelling out what a bar
vs. a dot means. The legend is always placed *below* the axes (never inside
the plotting area), and the figure is saved with bbox_inches="tight" (see
_save_and_show), so the saved PNG -- the artifact actually reviewed -- never
clips the legend regardless of how many entries it has.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import config

_DEFAULT_REFERENCE_LINES: dict[float, str | None] = {0.0: None}


def _save_and_show(fig, filename: str):
    path = config.DIAGRAMS_DIR / filename
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"saved: {path}")


def _draw_reference_lines(ax, reference_lines: dict[float, str | None] | None) -> list[Line2D]:
    """Draw each reference line; return a legend handle for each *labeled* one
    (an unlabeled line, e.g. the default plain zero-line, needs no legend
    entry -- it's self-explanatory against a Pearson-R axis)."""
    handles = []
    for value, label in (reference_lines or _DEFAULT_REFERENCE_LINES).items():
        ax.axhline(value, color="gray", linewidth=0.9, linestyle="--", zorder=1)
        if label:
            handles.append(Line2D([0], [0], color="gray", linestyle="--", label=label))
    return handles


def _level_ticklabel(level: str, n: int) -> str:
    return f"{level}\n(N={n})"


def _legend_below(ax, handles: list) -> None:
    """Place the legend below the axes so it can never sit on top of a bar or
    dot, however many entries it has."""
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=min(len(handles), 3),
        frameon=True,
        fontsize=8,
    )


def bar_with_dots(
    values_by_level: dict[str, pd.Series],
    title: str,
    ylabel: str = "Pearson R",
    xlabel: str = "Level",
    filename: str | None = None,
    dot_label: str = "Individual participant",
    dot_categories_by_level: dict[str, pd.Series] | None = None,
    category_colors: dict[str, str] | None = None,
    category_labels: dict[str, str] | None = None,
    reference_lines: dict[float, str | None] | None = None,
):
    """
    values_by_level: {level_label: Series of per-participant values (or a
    length-1 Series for a single AVG-vector value)}.

    dot_categories_by_level (optional): {level_label: Series keyed the same
    way as values_by_level[level_label]}, giving each sample a category (e.g.
    which of the 3 examined layers produced its best-layer score). When
    given, dots are colored by category_colors[category] instead of a flat
    color, and the legend gains one entry per category (named via
    category_labels) so the color code is never left unexplained.
    """
    levels = list(values_by_level.keys())
    means = [values_by_level[lv].mean() for lv in levels]

    fig, ax = plt.subplots(figsize=(1.7 * len(levels) + 1, 4.8))
    bars = ax.bar(levels, means, color="lightgray", edgecolor="black", zorder=2)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    rng = np.random.default_rng(0)
    for x, lv in enumerate(levels):
        # dropna: an undefined (NaN) correlation -- e.g. a participant with
        # zero-variance ratings, or a degenerate all-excluded group -- has no
        # sensible y-position and, left in, crashes matplotlib's categorical
        # x-axis unit conversion; skipping it just omits that one dot.
        series = values_by_level[lv].dropna()
        vals = series.values
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        if dot_categories_by_level is not None:
            cats = dot_categories_by_level[lv].reindex(series.index)
            colors = [category_colors.get(c, "black") for c in cats]
        else:
            colors = "black"
        ax.scatter(
            np.full(len(vals), x) + jitter, vals, color=colors, s=22, zorder=3, alpha=0.85,
            edgecolor="white", linewidth=0.4,
        )

    legend_handles = [Patch(facecolor="lightgray", edgecolor="black", label="Mean across participants")]
    if dot_categories_by_level is not None and category_colors:
        present = set()
        for lv in levels:
            present.update(dot_categories_by_level[lv].dropna().unique())
        for cat in sorted(present):
            label = (category_labels or {}).get(cat, cat)
            legend_handles.append(
                Line2D(
                    [0], [0], marker="o", linestyle="none", markeredgecolor="white",
                    color=category_colors.get(cat, "black"), label=f"{label} (won for this dot)",
                )
            )
    else:
        legend_handles.append(Line2D([0], [0], marker="o", linestyle="none", color="black", label=dot_label))
    legend_handles.extend(_draw_reference_lines(ax, reference_lines))

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(
        [_level_ticklabel(lv, len(values_by_level[lv])) for lv in levels], rotation=30, ha="right"
    )
    _legend_below(ax, legend_handles)

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
    xlabel: str = "Level",
    filename: str | None = None,
    reference_lines: dict[float, str | None] | None = None,
):
    """Same as bar_with_dots, plus a faint noise-ceiling bar/dots per level."""
    levels = list(values_by_level.keys())
    model_means = [values_by_level[lv].mean() for lv in levels]
    ceiling_means = [noise_ceiling_by_level[lv].mean() for lv in levels]

    x = np.arange(len(levels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(1.9 * len(levels) + 1, 4.8))
    model_bars = ax.bar(x - width / 2, model_means, width, color="lightgray", edgecolor="black", zorder=2)
    ceiling_bars = ax.bar(x + width / 2, ceiling_means, width, color="lightblue", edgecolor="black", zorder=2)
    ax.bar_label(model_bars, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(ceiling_bars, fmt="%.2f", padding=3, fontsize=8)

    rng = np.random.default_rng(0)
    for i, lv in enumerate(levels):
        # dropna: see bar_with_dots -- an undefined (NaN) correlation has no
        # sensible y-position and crashes the categorical x-axis otherwise.
        mv = values_by_level[lv].dropna().values
        cv = noise_ceiling_by_level[lv].dropna().values
        ax.scatter(
            np.full(len(mv), i - width / 2) + rng.uniform(-0.06, 0.06, len(mv)), mv,
            color="black", s=16, zorder=3, alpha=0.85, edgecolor="white", linewidth=0.3,
        )
        ax.scatter(
            np.full(len(cv), i + width / 2) + rng.uniform(-0.06, 0.06, len(cv)), cv,
            color="navy", s=16, zorder=3, alpha=0.85, edgecolor="white", linewidth=0.3,
        )

    legend_handles = [
        Patch(facecolor="lightgray", edgecolor="black", label="Model -- bar: mean, dots: participants"),
        Patch(facecolor="lightblue", edgecolor="black", label="Noise ceiling -- bar: mean, dots: participants"),
    ]
    legend_handles.extend(_draw_reference_lines(ax, reference_lines))

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_level_ticklabel(lv, len(values_by_level[lv])) for lv in levels], rotation=30, ha="right"
    )
    _legend_below(ax, legend_handles)

    if filename:
        _save_and_show(fig, filename)
    else:
        plt.show()
        plt.close(fig)
