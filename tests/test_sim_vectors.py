import pandas as pd

import sim_vectors


def _tidy_df():
    rows = [
        {"pair_id": "003_004", "condition": "letters", "participant_id": "p1", "participant_similarity_rating": 5},
        {"pair_id": "001_002", "condition": "letters", "participant_id": "p1", "participant_similarity_rating": 2},
        {"pair_id": "001_002", "condition": "letters", "participant_id": "p2", "participant_similarity_rating": 3},
        {"pair_id": "003_004", "condition": "letters", "participant_id": "p2", "participant_similarity_rating": 4},
        {"pair_id": "001_002", "condition": "no_letters", "participant_id": "p3", "participant_similarity_rating": 1},
        {"pair_id": "003_004", "condition": "no_letters", "participant_id": "p3", "participant_similarity_rating": 6},
    ]
    return pd.DataFrame(rows)


def test_build_human_similarity_vectors_shape_and_order():
    wide = sim_vectors.build_human_similarity_vectors(_tidy_df())

    # index sorted by pair_id, not by first-seen order in the tidy df
    assert list(wide.index) == ["001_002", "003_004"]
    assert set(wide.columns) == {("letters", "p1"), ("letters", "p2"), ("no_letters", "p3")}
    assert wide.loc["001_002", ("letters", "p1")] == 2
    assert wide.loc["003_004", ("letters", "p1")] == 5
    assert wide.loc["001_002", ("no_letters", "p3")] == 1


def test_get_vector_and_get_group():
    wide = sim_vectors.build_human_similarity_vectors(_tidy_df())

    vec = sim_vectors.get_vector(wide, "letters", "p2")
    assert vec.loc["001_002"] == 3
    assert vec.loc["003_004"] == 4

    group = sim_vectors.get_group(wide, "letters")
    assert list(group.columns) == ["p1", "p2"]


def test_save_load_round_trip(tmp_path):
    wide = sim_vectors.build_human_similarity_vectors(_tidy_df())
    path = tmp_path / "sim_vectors.pkl"

    sim_vectors.save(wide, path)
    loaded = sim_vectors.load(path)

    pd.testing.assert_frame_equal(wide, loaded)


def test_build_block_assignment_shape_and_per_participant_lookup():
    rows = [
        {"pair_id": "003_004", "condition": "letters", "participant_id": "p1", "block_index": 1},
        {"pair_id": "001_002", "condition": "letters", "participant_id": "p1", "block_index": 0},
        {"pair_id": "001_002", "condition": "letters", "participant_id": "p2", "block_index": 1},
        {"pair_id": "003_004", "condition": "letters", "participant_id": "p2", "block_index": 0},
    ]
    df = pd.DataFrame(rows)

    block_assignment = sim_vectors.build_block_assignment(df)

    assert list(block_assignment.index) == ["001_002", "003_004"]
    # block membership differs per participant for the same pair_id -- this
    # is the whole point (independently randomized presentation order)
    assert block_assignment.loc["001_002", ("letters", "p1")] == 0
    assert block_assignment.loc["001_002", ("letters", "p2")] == 1
    assert block_assignment.loc["003_004", ("letters", "p1")] == 1
    assert block_assignment.loc["003_004", ("letters", "p2")] == 0
