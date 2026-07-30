"""
Scoring, ranking, and plotting helpers for the DINOv2 layer/pooling grid
search. Kept separate from run_grid_search.py so the scoring logic can be
unit-tested against small hand-built tables without any real embeddings.
"""

import re

import matplotlib.pyplot as plt
import pandas as pd

import analysis
import config

_CONFIG_KEY_RE = re.compile(r"^block(\d{2})_(cls|mean)$")


def parse_config_key(config_key: str) -> tuple[int, str]:
    """"block06_mean" -> (6, "mean")."""
    m = _CONFIG_KEY_RE.match(config_key)
    if not m:
        raise ValueError(f"Not a grid config key: {config_key!r}")
    return int(m.group(1)), m.group(2)


def score_configs(
    wide: pd.DataFrame,
    condition: str,
    level: str,
    similarity_by_config: dict[str, pd.Series],
    excluded: set,
) -> pd.DataFrame:
    """
    One row per config_key: avg_vector_r (group-mean human vector vs model),
    mean_participant_r (mean of per-participant Pearson r), n_participants.
    """
    rows = []
    for config_key, sim in similarity_by_config.items():
        per_participant = analysis.per_participant_correlations(wide, condition, sim, excluded)
        avg_r = analysis.avg_vector_correlation(wide, condition, sim, excluded)
        # Baseline configs (e.g. "resnet50_places365_avgpool") don't match the
        # grid's "blockNN_pooling" naming -- keep them in the same table with
        # block=None so pooled_ranking/heatmap plotting (which filter on
        # block.notna()) skip them automatically.
        is_grid_config = bool(_CONFIG_KEY_RE.match(config_key))
        block, pooling = parse_config_key(config_key) if is_grid_config else (None, config_key)
        rows.append({
            "level": level,
            "condition": condition,
            "config_key": config_key,
            "block": block,
            "pooling": pooling,
            "avg_vector_r": avg_r,
            "mean_participant_r": per_participant.mean(),
            "n_participants": len(per_participant),
        })
    return pd.DataFrame(rows)


def best_config_per_participant(
    wide: pd.DataFrame,
    condition: str,
    level: str,
    similarity_by_config: dict[str, pd.Series],
    excluded: set,
) -> pd.DataFrame:
    """Per participant: which config_key wins, and their r using it."""
    best = analysis.best_layer_per_participant(wide, condition, similarity_by_config, excluded)
    out = best.rename(columns={"pearson_r": "r", "best_layer": "best_config"}).reset_index(names="participant_id")
    out.insert(0, "condition", condition)
    out.insert(0, "level", level)
    return out[["level", "condition", "participant_id", "best_config", "r"]]


def pooled_ranking(grid_results: pd.DataFrame, metric: str, top_n: int = 10) -> pd.DataFrame:
    """
    Average `metric` for each (block, pooling) config across every level and
    condition present in `grid_results`, ranked descending. Only meaningful
    for the 24 grid configs (block is not None), not baseline rows.
    """
    grid_only = grid_results[grid_results["block"].notna()]
    grouped = (
        grid_only.groupby(["block", "pooling", "config_key"], as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)
    )
    return grouped.head(top_n).reset_index(drop=True)


def top_configs(grid_results: pd.DataFrame, metric: str, level: str, condition: str, top_n: int = 10) -> pd.DataFrame:
    """Top configs for one specific (level, condition), not averaged."""
    subset = grid_results[(grid_results["level"] == level) & (grid_results["condition"] == condition)]
    return subset.sort_values(metric, ascending=False).head(top_n).reset_index(drop=True)


def most_preferred_configs(best_config_df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """How often each config_key wins across every (level, condition, participant) row."""
    return best_config_df["best_config"].value_counts().head(top_n)


def build_summary_text(
    grid_results: pd.DataFrame,
    best_config_df: pd.DataFrame,
    baseline_results: pd.DataFrame,
) -> str:
    lines = ["# DINOv2 layer x pooling grid search -- summary", ""]

    lines.append("## Top 10 configs overall (averaged across all levels & conditions)")
    for metric, label in [("avg_vector_r", "AVG-vector r"), ("mean_participant_r", "mean per-participant r")]:
        lines.append(f"\n### By {label}")
        ranked = pooled_ranking(grid_results, metric, top_n=10)
        for _, row in ranked.iterrows():
            lines.append(f"- {row['config_key']}: {row[metric]:.4f}")

    lines.append("\n## Most frequently participant-preferred configs (across all levels/conditions)")
    for config_key, count in most_preferred_configs(best_config_df, top_n=10).items():
        lines.append(f"- {config_key}: won for {count} (level, condition, participant) combinations")

    lines.append("\n## Baseline comparison (existing run.py defaults)")
    for _, row in baseline_results.iterrows():
        lines.append(
            f"- {row['config_key']} @ {row['level']}/{row['condition']}: "
            f"avg_vector_r={row['avg_vector_r']:.4f}, mean_participant_r={row['mean_participant_r']:.4f}"
        )

    lines.append("\n## Per level/condition top-3 by AVG-vector r")
    for level in grid_results["level"].unique():
        for condition in grid_results["condition"].unique():
            top3 = top_configs(grid_results, "avg_vector_r", level, condition, top_n=3)
            if top3.empty:
                continue
            lines.append(f"\n### {level} / {condition}")
            for _, row in top3.iterrows():
                lines.append(f"- {row['config_key']}: avg_vector_r={row['avg_vector_r']:.4f}")

    return "\n".join(lines)


def plot_heatmaps(grid_results: pd.DataFrame, diagrams_dir) -> None:
    """
    One heatmap per (level, condition, metric): block (x-axis, 1-12) x
    pooling (y-axis, cls/mean), cell = metric value.
    """
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    grid_only = grid_results[grid_results["block"].notna()]

    for metric in ("avg_vector_r", "mean_participant_r"):
        for (level, condition), subset in grid_only.groupby(["level", "condition"]):
            pivot = subset.pivot(index="pooling", columns="block", values=metric).reindex(["cls", "mean"])

            fig, ax = plt.subplots(figsize=(10, 2.5))
            im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("DINOv2 block")
            ax.set_title(f"{metric} -- {level} / {condition}")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(diagrams_dir / f"{level}_{condition}_{metric}.png", dpi=config.FIGURE_DPI)
            plt.close(fig)
