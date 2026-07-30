import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: savefig must never require a display

# See test_extractor.py's comment: no shared conftest.py in this dir to avoid
# colliding with tests/conftest.py's bare "conftest" module name.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dinov2_layer_grid_search"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import analysis  # noqa: E402
import reporting  # noqa: E402


def _make_wide():
    data = {
        ("letters", "p1"): [1, 2, 3, 4],
        ("letters", "p2"): [2, 2, 3, 5],
        ("letters", "p3"): [1, 3, 2, 4],
        ("letters", "p4"): [2, 3, 4, 5],
        ("letters", "p5"): [1, 1, 2, 3],
    }
    wide = pd.DataFrame(data, index=["a", "b", "c", "d"])
    wide.columns = pd.MultiIndex.from_tuples(wide.columns)
    return wide


def _make_similarity_by_config():
    idx = ["a", "b", "c", "d"]
    return {
        "block01_cls": pd.Series([1, 2, 3, 4], index=idx),
        "block01_mean": pd.Series([4, 3, 2, 1], index=idx),
        "block06_mean": pd.Series([1, 1, 2, 3], index=idx),
        "resnet50_places365_avgpool": pd.Series([2, 2, 3, 5], index=idx),
    }


def test_parse_config_key():
    assert reporting.parse_config_key("block06_mean") == (6, "mean")
    assert reporting.parse_config_key("block12_cls") == (12, "cls")
    with pytest.raises(ValueError):
        reporting.parse_config_key("resnet50_places365_avgpool")


def test_score_configs_matches_direct_analysis_calls():
    wide = _make_wide()
    similarity_by_config = _make_similarity_by_config()

    result = reporting.score_configs(wide, "letters", "Full", similarity_by_config, excluded=set())

    assert set(result["config_key"]) == set(similarity_by_config)
    for _, row in result.iterrows():
        sim = similarity_by_config[row["config_key"]]
        expected_avg = analysis.avg_vector_correlation(wide, "letters", sim)
        expected_mean = analysis.per_participant_correlations(wide, "letters", sim).mean()
        assert row["avg_vector_r"] == pytest.approx(expected_avg)
        assert row["mean_participant_r"] == pytest.approx(expected_mean)
        assert row["n_participants"] == 5
        assert row["level"] == "Full"
        assert row["condition"] == "letters"


def test_score_configs_parses_block_and_pooling_for_grid_keys_only():
    wide = _make_wide()
    result = reporting.score_configs(wide, "letters", "Full", _make_similarity_by_config(), excluded=set())
    by_key = result.set_index("config_key")

    assert by_key.loc["block01_cls", "block"] == 1
    assert by_key.loc["block01_cls", "pooling"] == "cls"
    assert by_key.loc["block06_mean", "block"] == 6
    assert by_key.loc["block06_mean", "pooling"] == "mean"
    assert pd.isna(by_key.loc["resnet50_places365_avgpool", "block"])


def test_best_config_per_participant_matches_analysis_best_layer_per_participant():
    wide = _make_wide()
    similarity_by_config = _make_similarity_by_config()

    result = reporting.best_config_per_participant(wide, "letters", "Full", similarity_by_config, excluded=set())
    expected = analysis.best_layer_per_participant(wide, "letters", similarity_by_config)

    assert list(result.columns) == ["level", "condition", "participant_id", "best_config", "r"]
    assert len(result) == 5
    for _, row in result.iterrows():
        assert row["best_config"] == expected.loc[row["participant_id"], "best_layer"]
        assert row["r"] == pytest.approx(expected.loc[row["participant_id"], "pearson_r"])


def test_pooled_ranking_averages_metric_across_levels_and_excludes_baselines():
    grid_results = pd.DataFrame([
        {"level": "Full", "condition": "letters", "config_key": "block01_cls", "block": 1, "pooling": "cls", "avg_vector_r": 0.8, "mean_participant_r": 0.5, "n_participants": 5},
        {"level": "Deg05", "condition": "letters", "config_key": "block01_cls", "block": 1, "pooling": "cls", "avg_vector_r": 0.4, "mean_participant_r": 0.3, "n_participants": 5},
        {"level": "Full", "condition": "letters", "config_key": "block02_mean", "block": 2, "pooling": "mean", "avg_vector_r": 0.9, "mean_participant_r": 0.6, "n_participants": 5},
        {"level": "Full", "condition": "letters", "config_key": "resnet50_places365_avgpool", "block": None, "pooling": "resnet50_places365_avgpool", "avg_vector_r": 0.99, "mean_participant_r": 0.99, "n_participants": 5},
    ])

    ranked = reporting.pooled_ranking(grid_results, "avg_vector_r", top_n=10)

    assert "resnet50_places365_avgpool" not in set(ranked["config_key"])
    assert list(ranked["config_key"]) == ["block02_mean", "block01_cls"]
    assert ranked.loc[ranked["config_key"] == "block01_cls", "avg_vector_r"].iloc[0] == pytest.approx((0.8 + 0.4) / 2)


def test_most_preferred_configs_counts_wins():
    best_config_df = pd.DataFrame({
        "best_config": ["block01_cls", "block01_cls", "block02_mean", "block01_cls"],
    })

    counts = reporting.most_preferred_configs(best_config_df, top_n=10)

    assert counts["block01_cls"] == 3
    assert counts["block02_mean"] == 1


def test_build_summary_text_contains_expected_sections():
    wide = _make_wide()
    similarity_by_config = _make_similarity_by_config()
    grid_results = reporting.score_configs(wide, "letters", "Full", similarity_by_config, excluded=set())
    best_config_df = reporting.best_config_per_participant(wide, "letters", "Full", similarity_by_config, excluded=set())
    baseline_results = grid_results[grid_results["block"].isna()]

    text = reporting.build_summary_text(grid_results, best_config_df, baseline_results)

    assert "Top 10 configs overall" in text
    assert "Most frequently participant-preferred configs" in text
    assert "Baseline comparison" in text
    assert "block01_cls" in text


def test_plot_heatmaps_writes_one_png_per_level_condition_metric(tmp_path):
    wide = _make_wide()
    similarity_by_config = _make_similarity_by_config()
    grid_results = reporting.score_configs(wide, "letters", "Full", similarity_by_config, excluded=set())

    reporting.plot_heatmaps(grid_results, tmp_path)

    assert (tmp_path / "Full_letters_avg_vector_r.png").exists()
    assert (tmp_path / "Full_letters_mean_participant_r.png").exists()
