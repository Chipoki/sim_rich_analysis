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
