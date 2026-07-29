import pandas as pd
import pytest

import parsing
from conftest import make_participant_trials


def test_parse_participant_letters_shape(pair_ids):
    ratings = {pid: i + 1 for i, pid in enumerate(pair_ids)}
    letter_correct = {pid: (i % 2 == 0) for i, pid in enumerate(pair_ids)}
    trials = make_participant_trials("p1", "letters", pair_ids, ratings, letter_correct)

    meta, rows = parsing._parse_participant(trials, "letters")

    assert meta["participant_id"] == "p1"
    assert meta["condition"] == "letters"
    assert len(rows) == len(pair_ids)
    for row in rows:
        pid = row["pair_id"]
        assert row["participant_similarity_rating"] == ratings[pid]
        assert row["letter_discrimination_correct"] == letter_correct[pid]
        assert row["letter1_shown"] == "L"


def test_parse_participant_no_letters_shape(pair_ids):
    trials = make_participant_trials("p1", "no_letters", pair_ids)

    _meta, rows = parsing._parse_participant(trials, "no_letters")

    assert len(rows) == len(pair_ids)
    for row in rows:
        assert row["letter_discrimination_correct"] is None
        assert row["letter1_shown"] is None


def test_pair_id_is_the_join_key_not_trial_order(pair_ids):
    """The module's core invariant: trial_index/stream position must never
    be relied on -- only pair_id. Shuffling the pair order within the stream
    must not change which rating ends up attached to which pair_id."""
    ratings = {pid: i + 1 for i, pid in enumerate(pair_ids)}

    trials_in_order = make_participant_trials("p1", "no_letters", pair_ids, ratings, shuffle_order=False)
    trials_shuffled = make_participant_trials("p1", "no_letters", pair_ids, ratings, shuffle_order=True)

    _m1, rows_in_order = parsing._parse_participant(trials_in_order, "no_letters")
    _m2, rows_shuffled = parsing._parse_participant(trials_shuffled, "no_letters")

    by_pair_in_order = {r["pair_id"]: r["participant_similarity_rating"] for r in rows_in_order}
    by_pair_shuffled = {r["pair_id"]: r["participant_similarity_rating"] for r in rows_shuffled}

    assert by_pair_in_order == by_pair_shuffled == ratings


def test_missing_similarity_question_raises(pair_ids):
    trials = make_participant_trials("p1", "no_letters", pair_ids[:1])
    assert trials[-1]["trial_type"] == "similarity_question"
    malformed = trials[:-1]  # drop the trailing similarity_question

    with pytest.raises(ValueError, match="Expected similarity_question"):
        parsing._parse_participant(malformed, "no_letters")


def test_letter_accuracy_by_participant(pair_ids):
    letter_correct = {pair_ids[0]: True, pair_ids[1]: True, pair_ids[2]: False}
    trials = make_participant_trials("p1", "letters", pair_ids, letter_correct=letter_correct)
    _meta, rows = parsing._parse_participant(trials, "letters")
    df = pd.DataFrame(rows)
    df["participant_id"] = "p1"
    df["condition"] = "letters"
    df["letter_discrimination_correct"] = df["letter_discrimination_correct"].astype("boolean")

    acc = parsing.letter_accuracy_by_participant(df)

    assert acc["p1"] == pytest.approx(2 / 3)


def test_letter_accuracy_by_participant_and_block(pair_ids):
    letter_correct = {pair_ids[0]: True, pair_ids[1]: False, pair_ids[2]: True}
    trials = make_participant_trials("p1", "letters", pair_ids, letter_correct=letter_correct)
    _meta, rows = parsing._parse_participant(trials, "letters")
    df = pd.DataFrame(rows)
    df["participant_id"] = "p1"
    df["condition"] = "letters"
    df["letter_discrimination_correct"] = df["letter_discrimination_correct"].astype("boolean")
    # one pair per block, in row order (the fixture itself always uses
    # block_index=0, so assign distinct blocks by hand to exercise grouping)
    df["block_index"] = [0, 1, 2]

    acc = parsing.letter_accuracy_by_participant_and_block(df)

    assert acc.loc["p1", 0] == 1.0
    assert acc.loc["p1", 1] == 0.0
    assert acc.loc["p1", 2] == 1.0


def test_load_all_merges_conditions_and_casts_dtype(write_result_file, pair_ids):
    letters_path = write_result_file(
        "letters.txt", "letters", [{"participant_id": "L1", "pair_ids": pair_ids}]
    )
    no_letters_path = write_result_file(
        "no_letters.txt", "no_letters", [{"participant_id": "N1", "pair_ids": pair_ids}]
    )

    df = parsing.load_all({"letters": letters_path, "no_letters": no_letters_path})

    assert set(df["condition"].unique()) == {"letters", "no_letters"}
    assert str(df["letter_discrimination_correct"].dtype) == "boolean"
    no_letters_rows = df[df["condition"] == "no_letters"]
    assert no_letters_rows["letter_discrimination_correct"].isna().all()
