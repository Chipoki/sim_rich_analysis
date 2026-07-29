import pandas as pd

import config
import plotting


def _series_by_level():
    return {
        "Full": pd.Series({"p1": 0.8, "p2": 0.6}),
        "Deg05": pd.Series({"p1": 0.5, "p2": 0.4}),
    }


def test_bar_with_dots_saves_file_when_filename_given(patch_result_dirs):
    plotting.bar_with_dots(_series_by_level(), "title", filename="degradation.png")

    assert (config.DIAGRAMS_DIR / "degradation.png").exists()


def test_bar_with_dots_no_file_written_without_filename(patch_result_dirs):
    plotting.bar_with_dots(_series_by_level(), "title", filename=None)

    assert list(config.DIAGRAMS_DIR.iterdir()) == []


def test_bar_with_noise_ceiling_saves_file(patch_result_dirs):
    values = _series_by_level()
    ceiling = {"Full": pd.Series({"p1": 0.9, "p2": 0.7}), "Deg05": pd.Series({"p1": 0.9, "p2": 0.7})}

    plotting.bar_with_noise_ceiling(values, ceiling, "title", filename="vs_ceiling.png")

    assert (config.DIAGRAMS_DIR / "vs_ceiling.png").exists()


def test_bar_with_dots_handles_nan_and_empty_series(patch_result_dirs):
    """A NaN entry (e.g. a participant with zero-variance ratings, or a
    degenerate all-excluded group) must not crash rendering: matplotlib's
    categorical x-axis (from the string level labels) previously raised a
    ConversionError when a scatter point's y-value was NaN."""
    values = {"Full": pd.Series({"p1": float("nan"), "p2": 0.5}), "Deg05": pd.Series([], dtype=float)}

    plotting.bar_with_dots(values, "title", filename="nan_and_empty.png")

    assert (config.DIAGRAMS_DIR / "nan_and_empty.png").exists()


def test_bar_with_noise_ceiling_handles_nan_series(patch_result_dirs):
    values = {"Full": pd.Series({"p1": float("nan"), "p2": 0.5}), "Deg05": pd.Series({"p1": 0.4, "p2": 0.3})}
    ceiling = {"Full": pd.Series({"p1": 0.9, "p2": float("nan")}), "Deg05": pd.Series({"p1": 0.9, "p2": 0.7})}

    plotting.bar_with_noise_ceiling(values, ceiling, "title", filename="nan_ceiling.png")

    assert (config.DIAGRAMS_DIR / "nan_ceiling.png").exists()


def _keep_figure_open(monkeypatch):
    """Keep the last-drawn figure open (instead of plotting.py's normal
    plt.close) so its legend/axes can be inspected via plotting.plt.gcf()."""
    monkeypatch.setattr(plotting.plt, "close", lambda *a, **k: None)


def test_bar_with_dots_default_legend_explains_bar_and_dot(monkeypatch, patch_result_dirs):
    _keep_figure_open(monkeypatch)

    plotting.bar_with_dots(_series_by_level(), "title", filename=None)

    fig = plotting.plt.gcf()
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "Mean across participants" in labels
    assert "Individual participant" in labels
    plotting.plt.close(fig)


def test_bar_with_dots_colors_by_category_and_legends_each_one(monkeypatch, patch_result_dirs):
    _keep_figure_open(monkeypatch)
    categories = {
        "Full": pd.Series({"p1": "avgpool", "p2": "layer4"}),
        "Deg05": pd.Series({"p1": "layer4", "p2": "avgpool"}),
    }

    plotting.bar_with_dots(
        _series_by_level(), "title", filename=None,
        dot_categories_by_level=categories, category_colors=config.LAYER_COLORS, category_labels=config.LAYERS,
    )

    fig = plotting.plt.gcf()
    labels = " ".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
    assert config.LAYERS["avgpool"] in labels
    assert config.LAYERS["layer4"] in labels
    plotting.plt.close(fig)


def test_bar_with_dots_reference_line_adds_labeled_legend_entry(monkeypatch, patch_result_dirs):
    _keep_figure_open(monkeypatch)

    plotting.bar_with_dots(_series_by_level(), "title", filename=None, reference_lines={0.5: "chance level"})

    fig = plotting.plt.gcf()
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "chance level" in labels
    plotting.plt.close(fig)


def test_bar_with_dots_xlabel_default_and_override(monkeypatch, patch_result_dirs):
    _keep_figure_open(monkeypatch)

    plotting.bar_with_dots(_series_by_level(), "title", filename=None)
    fig1 = plotting.plt.gcf()
    assert fig1.axes[0].get_xlabel() == "Level"
    plotting.plt.close(fig1)

    plotting.bar_with_dots(_series_by_level(), "title", xlabel="Experimental block", filename=None)
    fig2 = plotting.plt.gcf()
    assert fig2.axes[0].get_xlabel() == "Experimental block"
    plotting.plt.close(fig2)


def test_bar_with_dots_xtick_labels_include_sample_size(monkeypatch, patch_result_dirs):
    _keep_figure_open(monkeypatch)

    plotting.bar_with_dots(_series_by_level(), "title", filename=None)

    fig = plotting.plt.gcf()
    ticklabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert any("N=2" in t for t in ticklabels)
    plotting.plt.close(fig)


def test_bar_with_noise_ceiling_legend_explains_both_series(monkeypatch, patch_result_dirs):
    _keep_figure_open(monkeypatch)
    values = _series_by_level()
    ceiling = {"Full": pd.Series({"p1": 0.9, "p2": 0.7}), "Deg05": pd.Series({"p1": 0.9, "p2": 0.7})}

    plotting.bar_with_noise_ceiling(values, ceiling, "title", filename=None)

    fig = plotting.plt.gcf()
    labels = " ".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
    assert "Model" in labels and "mean" in labels
    assert "Noise ceiling" in labels
    plotting.plt.close(fig)


def test_bar_with_noise_ceiling_custom_labels_and_colors(monkeypatch, patch_result_dirs):
    """bar_with_noise_ceiling is reused (not just for model-vs-ceiling) for
    the Letters-vs-No-Letters comparison, so its labels/colors must be
    overridable rather than hardcoded."""
    _keep_figure_open(monkeypatch)
    letters = _series_by_level()
    no_letters = {"Full": pd.Series({"p1": 0.3, "p2": 0.2}), "Deg05": pd.Series({"p1": 0.3, "p2": 0.2})}

    plotting.bar_with_noise_ceiling(
        letters, no_letters, "title", filename=None,
        series_a_label="Letters", series_b_label="No-Letters",
        series_a_color="steelblue", series_b_color="salmon",
    )

    fig = plotting.plt.gcf()
    labels = " ".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
    assert "Letters" in labels
    assert "No-Letters" in labels
    assert "Model" not in labels
    plotting.plt.close(fig)


def test_bar_with_noise_ceiling_footnote_rendered(monkeypatch, patch_result_dirs):
    """The reliability-provenance footnote (LOO / split-half + iteration
    count) must actually be drawn on the figure when passed, not just
    accepted and discarded."""
    _keep_figure_open(monkeypatch)
    values = _series_by_level()
    ceiling = {"Full": pd.Series({"p1": 0.9, "p2": 0.7}), "Deg05": pd.Series({"p1": 0.9, "p2": 0.7})}

    plotting.bar_with_noise_ceiling(
        values, ceiling, "title", filename=None,
        footnote="Noise ceiling = LOO. Split-half (SB, 1000 iterations): 0.73.",
    )

    fig = plotting.plt.gcf()
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("1000 iterations" in t for t in texts)
    plotting.plt.close(fig)


def test_bar_with_noise_ceiling_no_footnote_by_default(monkeypatch, patch_result_dirs):
    """No footnote= given -> no italic footnote text; bar_label's own value
    labels are unaffected (they're normal-style, not italic)."""
    _keep_figure_open(monkeypatch)
    values = _series_by_level()
    ceiling = {"Full": pd.Series({"p1": 0.9, "p2": 0.7}), "Deg05": pd.Series({"p1": 0.9, "p2": 0.7})}

    plotting.bar_with_noise_ceiling(values, ceiling, "title", filename=None)

    fig = plotting.plt.gcf()
    italic_texts = [t.get_text() for t in fig.axes[0].texts if t.get_style() == "italic"]
    assert italic_texts == []
    plotting.plt.close(fig)


def test_save_and_show_creates_category_subdirectory(patch_result_dirs):
    plotting.bar_with_dots(_series_by_level(), "title", filename="degradation/nested.png")

    assert (config.DIAGRAMS_DIR / "degradation" / "nested.png").exists()


def test_scatter_relationship_saves_file(patch_result_dirs):
    x = {"letters": pd.Series({"p1": 100.0, "p2": 200.0, "p3": 300.0})}
    y = {"letters": pd.Series({"p1": 0.1, "p2": 0.5, "p3": 0.9})}

    plotting.scatter_relationship(x, y, "title", xlabel="RT (ms)", filename="individual_differences/rt.png")

    assert (config.DIAGRAMS_DIR / "individual_differences" / "rt.png").exists()


def test_scatter_relationship_legend_reports_r_and_n(monkeypatch, patch_result_dirs):
    _keep_figure_open(monkeypatch)
    x = {"letters": pd.Series({"p1": 1.0, "p2": 2.0, "p3": 3.0})}
    y = {"letters": pd.Series({"p1": 1.0, "p2": 2.0, "p3": 3.0})}  # perfectly correlated

    plotting.scatter_relationship(x, y, "title", xlabel="x", filename=None)

    fig = plotting.plt.gcf()
    labels = " ".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
    assert "r=1.00" in labels
    assert "N=3" in labels
    plotting.plt.close(fig)


def test_scatter_relationship_handles_nan_and_mismatched_index(patch_result_dirs):
    """NaN values and participants present in only one of x/y must not crash
    -- only fully-paired, non-NaN participants get plotted/correlated."""
    x = {"letters": pd.Series({"p1": 1.0, "p2": float("nan"), "p3": 3.0, "p4": 4.0})}
    y = {"letters": pd.Series({"p1": 1.0, "p2": 2.0, "p3": float("nan")})}  # p4 missing entirely

    plotting.scatter_relationship(x, y, "title", xlabel="x", filename="individual_differences/nan_case.png")

    assert (config.DIAGRAMS_DIR / "individual_differences" / "nan_case.png").exists()
