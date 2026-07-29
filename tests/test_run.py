"""
Pipeline-level smoke test: runs run.run() end-to-end against a fully
synthetic, reduced-scale mini pipeline (few pairs, tiny images, one layer,
a stub extractor instead of real ResNet compute) to confirm the stages wire
together correctly. Not a substitute for tests/test_slow_e2e.py, which
exercises the real data/weights when available.
"""

import pickle

import pytest

import config
import embeddings
import run
from conftest import StubExtractor


@pytest.fixture
def synthetic_pipeline_env(monkeypatch, tmp_path, patch_result_dirs, write_result_file, tiny_image_dir, pair_ids):
    image_ids = sorted({iid for pid in pair_ids for iid in pid.split("_")})

    letters_specs = [
        {"participant_id": "L1", "pair_ids": pair_ids, "ratings": {"001_002": 1, "003_004": 3, "005_006": 6}},
        {"participant_id": "L2", "pair_ids": pair_ids, "ratings": {"001_002": 2, "003_004": 4, "005_006": 5}},
        {"participant_id": "L3", "pair_ids": pair_ids, "ratings": {"001_002": 1, "003_004": 2, "005_006": 6}},
        {"participant_id": "L4", "pair_ids": pair_ids, "ratings": {"001_002": 3, "003_004": 3, "005_006": 4}},
    ]
    no_letters_specs = [
        {"participant_id": "N1", "pair_ids": pair_ids, "ratings": {"001_002": 2, "003_004": 5, "005_006": 1}},
        {"participant_id": "N2", "pair_ids": pair_ids, "ratings": {"001_002": 3, "003_004": 4, "005_006": 2}},
        {"participant_id": "N3", "pair_ids": pair_ids, "ratings": {"001_002": 1, "003_004": 6, "005_006": 3}},
        {"participant_id": "N4", "pair_ids": pair_ids, "ratings": {"001_002": 4, "003_004": 4, "005_006": 1}},
    ]
    letters_path = write_result_file("letters.txt", "letters", letters_specs)
    no_letters_path = write_result_file("no_letters.txt", "no_letters", no_letters_specs)

    tiny_image_dir(image_ids, subdir="full")
    tiny_image_dir(image_ids, subdir="deg05")
    tiny_image_dir(image_ids, subdir="desat")

    monkeypatch.setattr(config, "RESULT_FILES", {"letters": letters_path, "no_letters": no_letters_path})
    monkeypatch.setattr(config, "SRC_DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "LEVEL_FOLDERS", {"Full": "full", "Deg05": "deg05", "Desaturated": "desat"})
    monkeypatch.setattr(config, "DEGRADATION_LEVELS", ["Full", "Deg05"])
    monkeypatch.setattr(config, "DESATURATION_LEVELS", ["Full", "Desaturated"])
    monkeypatch.setattr(config, "ALL_LEVELS", ["Full", "Deg05", "Desaturated"])
    monkeypatch.setattr(config, "LAYERS", {"avgpool": "AvgPool"})
    monkeypatch.setattr(embeddings, "get_extractor", lambda backend=config.BACKEND: StubExtractor())


def test_run_end_to_end_smoke(synthetic_pipeline_env):
    run.run()

    results_path = config.DATA_DIR / "analysis_results.pkl"
    assert results_path.exists()

    for condition in ["letters", "no_letters"]:
        for suffix in (
            "degradation_levels", "best_layer", "avg_vector", "desaturation", "degradation_vs_noise_ceiling",
        ):
            assert (config.DIAGRAMS_DIR / f"{condition}_{suffix}.png").exists()

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    assert set(results.keys()) == {"letters", "no_letters"}
    for condition_results in results.values():
        assert set(condition_results.keys()) == {
            "r_by_level", "best_layer_by_level", "avg_r_by_level", "r_by_desat", "loo", "split_half",
        }
