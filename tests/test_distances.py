import numpy as np
import pandas as pd
import pytest

import distances


def test_cosine_distance_identical_vectors_is_zero():
    v = np.array([1.0, 2.0, 3.0])
    assert distances.cosine_distance(v, v) == pytest.approx(0.0, abs=1e-9)


def test_cosine_distance_orthogonal_vectors_is_one():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert distances.cosine_distance(a, b) == pytest.approx(1.0)


def test_cosine_distance_zero_vector_guard():
    a = np.zeros(3)
    b = np.array([1.0, 2.0, 3.0])
    assert distances.cosine_distance(a, b) == 1.0


def test_pair_distances_splits_pair_id_and_sorts():
    embs = {"001": np.array([1.0, 0.0]), "002": np.array([0.0, 1.0]), "003": np.array([1.0, 0.0])}
    pair_ids = ["003_001", "001_002"]

    result = distances.pair_distances(embs, pair_ids)

    assert list(result.index) == sorted(pair_ids)
    assert result["001_002"] == pytest.approx(1.0)
    assert result["003_001"] == pytest.approx(0.0, abs=1e-9)


def test_build_dist_vectors_columns_and_similarity_relationship():
    pair_ids = ["001_002"]
    embeddings_by_level_layer = {
        ("Full", "avgpool"): {"001": np.array([1.0, 0.0]), "002": np.array([0.0, 1.0])},
        ("Deg05", "avgpool"): {"001": np.array([1.0, 0.0]), "002": np.array([1.0, 0.0])},
    }

    df = distances.build_dist_vectors(embeddings_by_level_layer, pair_ids)

    assert df.columns.names == ["level", "layer", "metric"]
    assert set(df.columns.get_level_values("level")) == {"Full", "Deg05"}
    dist = df[("Full", "avgpool", "distance")]
    sim = df[("Full", "avgpool", "similarity")]
    pd.testing.assert_series_equal(sim, 1.0 - dist, check_names=False)


def test_get_similarity_accessor():
    pair_ids = ["001_002"]
    embeddings_by_level_layer = {
        ("Full", "avgpool"): {"001": np.array([1.0, 0.0]), "002": np.array([1.0, 0.0])},
    }
    df = distances.build_dist_vectors(embeddings_by_level_layer, pair_ids)

    sim = distances.get_similarity(df, "Full", "avgpool")

    assert sim["001_002"] == pytest.approx(1.0)


def test_save_load_round_trip(tmp_path):
    pair_ids = ["001_002"]
    embeddings_by_level_layer = {
        ("Full", "avgpool"): {"001": np.array([1.0, 0.0]), "002": np.array([0.0, 1.0])},
    }
    df = distances.build_dist_vectors(embeddings_by_level_layer, pair_ids)
    path = tmp_path / "dist_vectors.pkl"

    distances.save(df, path)
    loaded = distances.load(path)

    pd.testing.assert_frame_equal(df, loaded)
