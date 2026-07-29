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
        for suffix in ("degradation_levels", "best_layer", "avg_vector", "degradation_vs_noise_ceiling"):
            assert (config.DIAGRAMS_DIR / "degradation" / f"{condition}_{suffix}.png").exists()
        assert (config.DIAGRAMS_DIR / "desaturation" / f"{condition}_desaturation.png").exists()
        assert (config.DIAGRAMS_DIR / "experimental_blocks" / f"{condition}_by_block.png").exists()
    assert (config.DIAGRAMS_DIR / "experimental_blocks" / "letters_accuracy_by_block.png").exists()

    # cross-condition figures: one per examined layer (config.LAYERS is
    # monkeypatched to a single "avgpool" entry in this fixture) plus the two
    # individual-differences scatter plots
    assert (config.DIAGRAMS_DIR / "degradation" / "letters_vs_no_letters_avgpool.png").exists()
    assert (config.DIAGRAMS_DIR / "individual_differences" / "r_vs_response_time.png").exists()
    assert (config.DIAGRAMS_DIR / "individual_differences" / "letters_r_vs_letter_accuracy.png").exists()

    # config.EXTRA_BACKENDS defaults to ["dinov2"] (not monkeypatched here), so
    # the same figure set is expected again under diagrams/dinov2/ -- except
    # letters_accuracy_by_block, which is purely behavioral and only ever
    # generated once (from the primary backend's run, see run._run_condition).
    for condition in ["letters", "no_letters"]:
        for suffix in ("degradation_levels", "best_layer", "avg_vector", "degradation_vs_noise_ceiling"):
            assert (config.DIAGRAMS_DIR / "dinov2" / "degradation" / f"{condition}_{suffix}.png").exists()
        assert (config.DIAGRAMS_DIR / "dinov2" / "desaturation" / f"{condition}_desaturation.png").exists()
        assert (config.DIAGRAMS_DIR / "dinov2" / "experimental_blocks" / f"{condition}_by_block.png").exists()
    assert not (config.DIAGRAMS_DIR / "dinov2" / "experimental_blocks" / "letters_accuracy_by_block.png").exists()
    assert (config.DIAGRAMS_DIR / "dinov2" / "degradation" / "letters_vs_no_letters_dinov2.png").exists()
    assert (config.DIAGRAMS_DIR / "dinov2" / "individual_differences" / "r_vs_response_time.png").exists()
    assert (config.DIAGRAMS_DIR / "dinov2" / "individual_differences" / "letters_r_vs_letter_accuracy.png").exists()

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    assert set(results.keys()) == {"resnet50_places365", "dinov2"}
    common_keys = {
        "r_by_level", "r_by_level_per_layer", "best_layer_by_level", "avg_r_by_level",
        "r_by_desat", "r_by_block", "loo", "split_half",
    }
    for backend_results in results.values():
        assert set(backend_results.keys()) == {"letters", "no_letters"}
        assert set(backend_results["no_letters"].keys()) == common_keys
        assert set(backend_results["letters"].keys()) == common_keys | {"letter_accuracy_by_block"}


def test_run_extra_backends_disabled_produces_single_backend_results(monkeypatch, synthetic_pipeline_env):
    """config.EXTRA_BACKENDS is opt-in-by-default (["dinov2"]); disabling it
    must fall back to exactly the pre-extra-backend behavior -- one backend,
    no diagrams/dinov2/ sub-tree at all."""
    monkeypatch.setattr(config, "EXTRA_BACKENDS", [])

    run.run()

    with open(config.DATA_DIR / "analysis_results.pkl", "rb") as f:
        results = pickle.load(f)
    assert set(results.keys()) == {"resnet50_places365"}
    assert not (config.DIAGRAMS_DIR / "dinov2").exists()


def test_layers_for_backend_resnet_returns_config_layers():
    assert run._layers_for_backend("resnet50_places365") == config.LAYERS


def test_layers_for_backend_single_layer_backend_keyed_by_backend_name():
    assert run._layers_for_backend("dinov2") == {"dinov2": "DinoV2"}


def test_default_layer_for_backend():
    assert run._default_layer_for_backend("resnet50_places365") == config.DEFAULT_LAYER
    assert run._default_layer_for_backend("dinov2") == "dinov2"


def test_layer_colors_for_backend():
    assert run._layer_colors_for_backend("resnet50_places365") == config.LAYER_COLORS
    assert run._layer_colors_for_backend("dinov2") == {"dinov2": config.SINGLE_LAYER_BACKEND_COLORS["dinov2"]}
