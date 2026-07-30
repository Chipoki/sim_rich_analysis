"""
Bar + per-participant-dot figures, matching the PDF's style, plus cross-
participant scatter relationships (individual-differences analyses) and
condition line plots tracking a ratio metric across level transitions. Every
figure is saved to config.DIAGRAMS_DIR (filenames may include a category
sub-directory, e.g. "degradation/foo.png") -- never displayed interactively
(no plt.show()), so a full run.py run completes unattended without popping
up GUI windows.

Every bar figure always shows: a numeric value label on each bar, a
reference line (zero by default, overridable per call), the per-level
sample size baked directly into its x-tick label, and a legend spelling out
what a bar vs. a dot means. The legend is always placed *below* the axes
(never inside the plotting area), and the figure is saved with
bbox_inches="tight" (see _save), so the saved PNG -- the artifact actually
reviewed -- never clips the legend regardless of how many entries it has.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import pearsonr

import config

_DEFAULT_REFERENCE_LINES: dict[float, str | None] = {0.0: None}
_DEFAULT_CONDITION_COLORS = {"letters": "steelblue", "no_letters": "salmon"}


def _save(fig, filename: str):
    """filename may include a category sub-directory (e.g. "degradation/foo.png");
    that sub-directory is created on demand."""
    path = config.DIAGRAMS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
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
        _save(fig, filename)
    else:
        plt.close(fig)


def bar_with_noise_ceiling(
    values_by_level: dict[str, pd.Series],
    noise_ceiling_by_level: dict[str, pd.Series],
    title: str,
    ylabel: str = "Pearson R",
    xlabel: str = "Level",
    filename: str | None = None,
    reference_lines: dict[float, str | None] | None = None,
    series_a_label: str = "Model",
    series_b_label: str = "Noise ceiling",
    series_a_color: str = "lightgray",
    series_b_color: str = "lightblue",
    series_a_dot_color: str = "black",
    series_b_dot_color: str = "navy",
    footnote: str | None = None,
):
    """
    Two side-by-side bar+dot series per level. Originally written for
    model-vs-noise-ceiling specifically (hence the default labels/colors),
    but the shape -- two named per-level series compared side by side -- is
    general, e.g. also reused for a Letters-vs-No-Letters comparison.

    footnote (optional): a short symbolic note rendered below the legend --
    used by the model-vs-noise-ceiling figures to name the reliability
    methodology behind the plotted r values (LOO / split-half + iteration
    count), so that provenance isn't left implicit.
    """
    levels = list(values_by_level.keys())
    model_means = [values_by_level[lv].mean() for lv in levels]
    ceiling_means = [noise_ceiling_by_level[lv].mean() for lv in levels]

    x = np.arange(len(levels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(1.9 * len(levels) + 1, 4.8))
    model_bars = ax.bar(x - width / 2, model_means, width, color=series_a_color, edgecolor="black", zorder=2)
    ceiling_bars = ax.bar(x + width / 2, ceiling_means, width, color=series_b_color, edgecolor="black", zorder=2)
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
            color=series_a_dot_color, s=16, zorder=3, alpha=0.85, edgecolor="white", linewidth=0.3,
        )
        ax.scatter(
            np.full(len(cv), i + width / 2) + rng.uniform(-0.06, 0.06, len(cv)), cv,
            color=series_b_dot_color, s=16, zorder=3, alpha=0.85, edgecolor="white", linewidth=0.3,
        )

    legend_handles = [
        Patch(facecolor=series_a_color, edgecolor="black", label=f"{series_a_label} -- bar: mean, dots: participants"),
        Patch(facecolor=series_b_color, edgecolor="black", label=f"{series_b_label} -- bar: mean, dots: participants"),
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
    if footnote:
        ax.text(0.5, -0.38, footnote, transform=ax.transAxes, ha="center", fontsize=7, style="italic", wrap=True)

    if filename:
        _save(fig, filename)
    else:
        plt.close(fig)


def scatter_relationship(
    x_by_condition: dict[str, pd.Series],
    y_by_condition: dict[str, pd.Series],
    title: str,
    xlabel: str,
    ylabel: str = "Pearson R",
    condition_colors: dict[str, str] | None = None,
    reference_lines: dict[float, str | None] | None = None,
    filename: str | None = None,
):
    """
    Cross-participant scatter relating two per-participant quantities (e.g.
    response time vs model-agreement) -- an individual-differences analysis,
    distinct from the bar-per-level figures above. One color per condition;
    each condition's own Pearson r (and N) is computed and shown directly in
    the legend, so the relationship's strength is never left for the reader
    to eyeball unassisted.

    x_by_condition / y_by_condition: {condition: Series}, indexed by
    participant_id. Only participants present in *both* series (per
    condition) are plotted/correlated.
    """
    condition_colors = condition_colors or _DEFAULT_CONDITION_COLORS

    fig, ax = plt.subplots(figsize=(6.5, 5))
    legend_handles = []
    for condition, x in x_by_condition.items():
        y = y_by_condition[condition]
        common = x.index.intersection(y.index)
        xv, yv = x.loc[common].dropna(), y.loc[common].dropna()
        common = xv.index.intersection(yv.index)
        xv, yv = xv.loc[common], yv.loc[common]
        color = condition_colors.get(condition, "black")
        ax.scatter(xv, yv, color=color, s=32, alpha=0.8, edgecolor="white", linewidth=0.4, zorder=3)

        if len(common) >= 2 and xv.std() > 0 and yv.std() > 0:
            r, _p = pearsonr(xv, yv)
            label = f"{condition} (r={r:.2f}, N={len(common)})"
        else:
            label = f"{condition} (N={len(common)})"
        legend_handles.append(Line2D([0], [0], marker="o", linestyle="none", color=color, label=label))

    legend_handles.extend(_draw_reference_lines(ax, reference_lines))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _legend_below(ax, legend_handles)

    if filename:
        _save(fig, filename)
    else:
        plt.close(fig)


_DEFAULT_RATIO_REFERENCE_LINES: dict[float, str | None] = {1.0: "no change (ratio = 1)"}


def ratio_line_plot(
    ratios_by_condition: dict[str, list[float]],
    transition_labels: list[str],
    title: str,
    ylabel: str = "Correlation ratio: r(later level) / r(earlier level)",
    xlabel: str = "Degradation transition",
    condition_colors: dict[str, str] | None = None,
    reference_lines: dict[float, str | None] | None = None,
    filename: str | None = None,
):
    """
    One line per condition, tracking a ratio metric across a sequence of
    adjacent-level transitions (e.g. "Full:Deg05", "Deg05:Deg3",
    "Deg3:Tunnel") -- a companion view to the per-level bar figures, showing
    how correlation changes relative to the *previous* level rather than each
    level's absolute value. Direction: ratio_i = value(transition_labels[i]'s
    later level) / value(its earlier level), so ratio < 1 means correlation
    dropped moving to the more-degraded level.

    ratios_by_condition: {condition: [ratio_per_transition, ...]}, each list
    the same length/order as transition_labels. A ratio may be NaN/inf if its
    denominator level's underlying value was ~0 -- matplotlib simply leaves a
    gap at that point rather than raising.
    """
    condition_colors = condition_colors or _DEFAULT_CONDITION_COLORS
    reference_lines = _DEFAULT_RATIO_REFERENCE_LINES if reference_lines is None else reference_lines

    fig, ax = plt.subplots(figsize=(1.6 * len(transition_labels) + 2, 4.8))
    legend_handles = []
    for condition, ratios in ratios_by_condition.items():
        color = condition_colors.get(condition, "black")
        ax.plot(transition_labels, ratios, marker="o", color=color, linewidth=1.8, markersize=6, zorder=3)
        legend_handles.append(Line2D([0], [0], marker="o", color=color, label=condition))

    legend_handles.extend(_draw_reference_lines(ax, reference_lines))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _legend_below(ax, legend_handles)

    if filename:
        _save(fig, filename)
    else:
        plt.close(fig)
