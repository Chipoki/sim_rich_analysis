import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import analysis
import config
import parsing


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


def test_get_excluded_participants_computed_from_letter_accuracy(write_result_file, pair_ids):
    letters_path = write_result_file(
        "letters.txt",
        "letters",
        [
            {"participant_id": "p_good", "pair_ids": pair_ids, "letter_correct": {pid: True for pid in pair_ids}},
            {"participant_id": "p_bad", "pair_ids": pair_ids, "letter_correct": {pid: False for pid in pair_ids}},
        ],
    )
    no_letters_path = write_result_file("no_letters.txt", "no_letters", [{"participant_id": "n1", "pair_ids": pair_ids}])
    df = parsing.load_all({"letters": letters_path, "no_letters": no_letters_path})

    excluded = analysis.get_excluded_participants(df)

    assert excluded["letters"] == {"p_bad"}
    assert excluded["no_letters"] == set()


def test_get_excluded_participants_manual_no_letters_exclusion(write_result_file, pair_ids, monkeypatch):
    letters_path = write_result_file("letters.txt", "letters", [{"participant_id": "p1", "pair_ids": pair_ids}])
    no_letters_path = write_result_file("no_letters.txt", "no_letters", [{"participant_id": "n1", "pair_ids": pair_ids}])
    df = parsing.load_all({"letters": letters_path, "no_letters": no_letters_path})
    monkeypatch.setattr(config, "NO_LETTERS_MANUAL_EXCLUDE_PARTICIPANT_ID", "n1")

    excluded = analysis.get_excluded_participants(df)

    assert excluded["no_letters"] == {"n1"}


def test_per_participant_correlations_matches_pearsonr():
    wide = _make_wide()
    model = pd.Series([1, 2, 3, 4], index=wide.index)

    result = analysis.per_participant_correlations(wide, "letters", model)

    for pid in ["p1", "p2", "p3", "p4", "p5"]:
        expected, _ = pearsonr(wide[("letters", pid)], model)
        assert result[pid] == pytest.approx(expected)


def test_per_participant_correlations_excludes_participants():
    wide = _make_wide()
    model = pd.Series([1, 2, 3, 4], index=wide.index)

    result = analysis.per_participant_correlations(wide, "letters", model, excluded={"p2", "p5"})

    assert set(result.index) == {"p1", "p3", "p4"}


def test_best_layer_per_participant_picks_max_r():
    wide = _make_wide()
    layer_a = pd.Series([1, 2, 3, 4], index=wide.index)
    layer_b = pd.Series([4, 3, 2, 1], index=wide.index)

    result = analysis.best_layer_per_participant(wide, "letters", {"a": layer_a, "b": layer_b})

    per_layer_a = analysis.per_participant_correlations(wide, "letters", layer_a)
    per_layer_b = analysis.per_participant_correlations(wide, "letters", layer_b)
    for pid in ["p1", "p2", "p3", "p4", "p5"]:
        assert result[pid] == pytest.approx(max(per_layer_a[pid], per_layer_b[pid]))


def test_avg_vector_correlation_matches_group_mean():
    wide = _make_wide()
    model = pd.Series([1, 2, 3, 4], index=wide.index)

    r = analysis.avg_vector_correlation(wide, "letters", model)

    group_mean = wide["letters"].mean(axis=1)
    expected, _ = pearsonr(group_mean, model)
    assert r == pytest.approx(expected)


def test_avg_vector_correlation_respects_exclusion():
    wide = _make_wide()
    model = pd.Series([1, 2, 3, 4], index=wide.index)

    r = analysis.avg_vector_correlation(wide, "letters", model, excluded={"p2", "p5"})

    group_mean = wide["letters"][["p1", "p3", "p4"]].mean(axis=1)
    expected, _ = pearsonr(group_mean, model)
    assert r == pytest.approx(expected)


def test_split_half_reliability_reproducible_for_fixed_seed():
    wide = _make_wide()

    r1 = analysis.split_half_reliability(wide, "letters", n_iter=20, seed=0)
    r2 = analysis.split_half_reliability(wide, "letters", n_iter=20, seed=0)

    assert r1 == r2
    assert -1.5 <= r1 <= 1.5  # Spearman-Brown correction can slightly exceed +/-1


def test_leave_one_out_reliability_matches_definition():
    wide = _make_wide()

    loo = analysis.leave_one_out_reliability(wide, "letters")

    assert set(loo.index) == {"p1", "p2", "p3", "p4", "p5"}
    group = wide["letters"]
    for pid in loo.index:
        rest_mean = group.drop(columns=pid).mean(axis=1)
        expected, _ = pearsonr(group[pid], rest_mean)
        assert loo[pid] == pytest.approx(expected)


def test_leave_one_out_reliability_respects_exclusion():
    wide = _make_wide()

    loo = analysis.leave_one_out_reliability(wide, "letters", excluded={"p5"})

    assert set(loo.index) == {"p1", "p2", "p3", "p4"}


def test_paired_level_test_identical_series_is_nan():
    r_by_level = {
        "Full": pd.Series({"p1": 0.5, "p2": 0.6}),
        "Deg05": pd.Series({"p1": 0.5, "p2": 0.6}),
    }

    note = analysis.paired_level_test(r_by_level, "Full", "Deg05")

    assert note["n"] == 2
    assert np.isnan(note["p_value"])


def test_paired_level_test_real_difference_gives_valid_p():
    r_by_level = {
        "Full": pd.Series({"p1": 0.9, "p2": 0.8, "p3": 0.85, "p4": 0.7, "p5": 0.6, "p6": 0.75}),
        "Deg05": pd.Series({"p1": 0.5, "p2": 0.4, "p3": 0.3, "p4": 0.2, "p5": 0.1, "p6": 0.35}),
    }

    note = analysis.paired_level_test(r_by_level, "Full", "Deg05")

    assert note["n"] == 6
    assert 0.0 <= note["p_value"] <= 1.0
