"""
Pipeline-level smoke test: runs run.run() end-to-end against a fully
synthetic, reduced-scale mini pipeline (few pairs, tiny images, one layer,
a stub extractor instead of real ResNet compute) to confirm the stages wire
together correctly. Not a substitute for tests/test_slow_e2e.py, which
exercises the real data/weights when available.
"""

import pickle

import pandas as pd
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

    # Every backend -- config.BACKEND ("resnet50_places365" -> "resnet50")
    # included -- gets its own diagrams/<name>/ sub-tree; no more special-
    # cased unprefixed path for the primary backend (see config.BACKEND_DIR_NAMES).
    for backend_dir in ("resnet50", "dinov2"):
        for condition in ["letters", "no_letters"]:
            for suffix in ("degradation_levels", "best_layer", "avg_vector", "degradation_vs_noise_ceiling"):
                assert (config.DIAGRAMS_DIR / backend_dir / "degradation" / f"{condition}_{suffix}.png").exists()
            assert (config.DIAGRAMS_DIR / backend_dir / "desaturation" / f"{condition}_desaturation.png").exists()
            assert (config.DIAGRAMS_DIR / backend_dir / "experimental_blocks" / f"{condition}_by_block.png").exists()
        assert (config.DIAGRAMS_DIR / backend_dir / "individual_differences" / "r_vs_response_time.png").exists()
        assert (
            config.DIAGRAMS_DIR / backend_dir / "individual_differences" / "letters_r_vs_letter_accuracy.png"
        ).exists()

    # letters_accuracy_by_block is purely behavioral (no embeddings involved)
    # and only ever generated once, from the primary backend's run -- stays
    # at the top level, in neither backend's sub-tree.
    assert (config.DIAGRAMS_DIR / "experimental_blocks" / "letters_accuracy_by_block.png").exists()
    assert not (config.DIAGRAMS_DIR / "dinov2" / "experimental_blocks" / "letters_accuracy_by_block.png").exists()
    assert not (config.DIAGRAMS_DIR / "resnet50" / "experimental_blocks" / "letters_accuracy_by_block.png").exists()

    # Cross-condition figures: one bar + one ratio-line-plot per examined
    # layer (config.LAYERS is monkeypatched to a single "avgpool" entry for
    # resnet50_places365 in this fixture; dinov2 has its own single-layer key).
    assert (config.DIAGRAMS_DIR / "resnet50" / "degradation" / "letters_vs_no_letters_avgpool.png").exists()
    assert (config.DIAGRAMS_DIR / "resnet50" / "degradation" / "letters_vs_no_letters_ratio_avgpool.png").exists()
    assert (config.DIAGRAMS_DIR / "dinov2" / "degradation" / "letters_vs_no_letters_dinov2.png").exists()
    assert (config.DIAGRAMS_DIR / "dinov2" / "degradation" / "letters_vs_no_letters_ratio_dinov2.png").exists()

    # Integrated (cross-model) comparison: resnet50 vs dinov2's best-layer r,
    # one figure per condition, built once after both backends have run.
    for condition in ["letters", "no_letters"]:
        assert (config.DIAGRAMS_DIR / "integrated" / f"{condition}_resnet50_vs_dinov2_best_layer.png").exists()

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
    must fall back to exactly one backend -- resnet50's own sub-tree still
    gets built, but no dinov2/ sub-tree and no integrated/ comparison (needs
    >1 backend) at all."""
    monkeypatch.setattr(config, "EXTRA_BACKENDS", [])

    run.run()

    with open(config.DATA_DIR / "analysis_results.pkl", "rb") as f:
        results = pickle.load(f)
    assert set(results.keys()) == {"resnet50_places365"}
    assert (config.DIAGRAMS_DIR / "resnet50" / "degradation" / "letters_degradation_levels.png").exists()
    assert not (config.DIAGRAMS_DIR / "dinov2").exists()
    assert not (config.DIAGRAMS_DIR / "integrated").exists()


def test_layers_for_backend_resnet_returns_config_layers():
    assert run._layers_for_backend("resnet50_places365") == config.LAYERS


def test_layers_for_backend_single_layer_backend_keyed_by_backend_name(patch_result_dirs):
    # patch_result_dirs redirects config.DINOV2_GRID_SEARCH_RESULTS_PATH to a
    # nonexistent tmp path, so this deterministically exercises the
    # no-grid-search fallback regardless of the real machine's state.
    assert run._layers_for_backend("dinov2") == {"dinov2": "DinoV2"}


def test_default_layer_for_backend(patch_result_dirs):
    assert run._default_layer_for_backend("resnet50_places365") == config.DEFAULT_LAYER
    assert run._default_layer_for_backend("dinov2") == "dinov2"


def test_layer_colors_for_backend(patch_result_dirs):
    assert run._layer_colors_for_backend("resnet50_places365") == config.LAYER_COLORS
    assert run._layer_colors_for_backend("dinov2") == {"dinov2": config.SINGLE_LAYER_BACKEND_COLORS["dinov2"]}


def test_layers_for_backend_dinov2_uses_grid_search_results_when_present(patch_result_dirs, monkeypatch):
    """When supplemental/dinov2_layer_grid_search has already dumped
    grid_results.pkl, dinov2's "layers" for the main pipeline become the top
    config.DINOV2_GRID_TOP_N ranked (block, pooling) configs instead of the
    single default pooled embedding -- this is the feature under test."""
    config.DINOV2_GRID_SEARCH_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"config_key": "block11_cls", "block": 11.0, "pooling": "cls", "avg_vector_r": 0.5},
        {"config_key": "block12_mean", "block": 12.0, "pooling": "mean", "avg_vector_r": 0.4},
        {"config_key": "block01_cls", "block": 1.0, "pooling": "cls", "avg_vector_r": 0.1},
        {"config_key": "dinov2_default", "block": None, "pooling": "dinov2_default", "avg_vector_r": 0.9},
    ]).to_pickle(config.DINOV2_GRID_SEARCH_RESULTS_PATH)
    monkeypatch.setattr(config, "DINOV2_GRID_TOP_N", 2)

    layers = run._layers_for_backend("dinov2")

    assert list(layers.keys()) == ["block11_cls", "block12_mean"]
    assert run._default_layer_for_backend("dinov2") == "block11_cls"
    colors = run._layer_colors_for_backend("dinov2")
    assert set(colors.keys()) == {"block11_cls", "block12_mean"}
    assert len(set(colors.values())) == 2  # distinct colors per config, like config.LAYER_COLORS


def test_degradation_transitions_use_current_config_not_a_frozen_constant(monkeypatch):
    """These must be functions, not module-level constants computed once at
    import time -- otherwise monkeypatching config.DEGRADATION_LEVELS (as the
    synthetic_pipeline_env fixture does, down to 2 levels) would silently
    keep using the real 4-level list and KeyError inside _transition_ratios."""
    monkeypatch.setattr(config, "DEGRADATION_LEVELS", ["Full", "Deg05"])

    assert run._degradation_transitions() == [("Full", "Deg05")]
    assert run._degradation_transition_labels() == ["Full:Deg05"]


def test_transition_ratios_divides_later_by_earlier_mean(monkeypatch):
    monkeypatch.setattr(config, "DEGRADATION_LEVELS", ["Full", "Deg05", "Deg3"])
    by_level = {
        "Full": pd.Series({"p1": 0.8, "p2": 0.4}),  # mean 0.6
        "Deg05": pd.Series({"p1": 0.3, "p2": 0.3}),  # mean 0.3
        "Deg3": pd.Series({"p1": 0.6, "p2": 0.6}),  # mean 0.6
    }

    ratios = run._transition_ratios(by_level)

    assert ratios == pytest.approx([0.3 / 0.6, 0.6 / 0.3])


def test_run_integrated_figures_builds_one_comparison_per_condition(patch_result_dirs, monkeypatch):
    monkeypatch.setattr(config, "DEGRADATION_LEVELS", ["Full", "Deg05"])

    def fake_condition_results(r_full, r_deg05):
        best = {
            "Full": pd.DataFrame({"pearson_r": [r_full], "best_layer": ["x"]}, index=["p1"]),
            "Deg05": pd.DataFrame({"pearson_r": [r_deg05], "best_layer": ["x"]}, index=["p1"]),
        }
        return {"best_layer_by_level": best}

    results = {
        "resnet50_places365": {
            "letters": fake_condition_results(0.5, 0.4),
            "no_letters": fake_condition_results(0.6, 0.3),
        },
        "dinov2": {
            "letters": fake_condition_results(0.2, 0.1),
            "no_letters": fake_condition_results(0.3, 0.2),
        },
    }

    run._run_integrated_figures(results, ["resnet50_places365", "dinov2"])

    for condition in ["letters", "no_letters"]:
        assert (config.DIAGRAMS_DIR / "integrated" / f"{condition}_resnet50_vs_dinov2_best_layer.png").exists()


def test_run_integrated_figures_noop_without_extra_backend(patch_result_dirs):
    run._run_integrated_figures({"resnet50_places365": {"letters": {}}}, ["resnet50_places365"])

    assert not (config.DIAGRAMS_DIR / "integrated").exists()
