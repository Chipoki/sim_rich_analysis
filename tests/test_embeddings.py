import pickle

import numpy as np
import pandas as pd
import pytest
import torch

import config
import embeddings
from conftest import StubExtractor


def test_extract_level_computes_and_caches_all_layers(patch_result_dirs, tiny_image_dir):
    image_dir = tiny_image_dir(["001", "002", "003"])
    extractor = StubExtractor()

    result = extractor.extract_level("Full", image_dir, ["avgpool", "layer4", "classifier"])

    assert set(result.keys()) == {"avgpool", "layer4", "classifier"}
    for layer in ("avgpool", "layer4", "classifier"):
        dim = StubExtractor.LAYER_DIMS[layer]
        assert set(result[layer].keys()) == {"001", "002", "003"}
        assert result[layer]["001"].shape == (dim,)
    assert extractor.forward_calls == 1  # 3 images < BATCH_SIZE -> single batch
    for layer in ("avgpool", "layer4", "classifier"):
        # Nested under the extractor's own BACKEND_KEY dir name (StubExtractor -> "stub"),
        # matching every real extractor's namespaced cache -- see config.BACKEND_DIR_NAMES.
        assert (config.EMBEDDINGS_DIR / "stub" / f"Full_{layer}.pkl").exists()


def test_extract_level_second_call_reads_from_cache(patch_result_dirs, tiny_image_dir):
    image_dir = tiny_image_dir(["001", "002"])
    StubExtractor().extract_level("Full", image_dir, ["avgpool"])

    second = StubExtractor()
    result = second.extract_level("Full", image_dir, ["avgpool"])

    assert second.load_calls == 0
    assert second.forward_calls == 0
    assert set(result["avgpool"].keys()) == {"001", "002"}


def test_extract_level_partial_cache_recomputes_only_missing_layer(patch_result_dirs, tiny_image_dir):
    image_dir = tiny_image_dir(["001", "002"])
    fake_cached = {"001": np.ones(4), "002": np.zeros(4)}
    cache_path = config.EMBEDDINGS_DIR / "stub" / "Full_classifier.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(fake_cached, f)

    extractor = StubExtractor()
    result = extractor.extract_level("Full", image_dir, ["avgpool", "classifier"])

    assert extractor.forward_calls == 1  # only "avgpool" was missing
    assert np.array_equal(result["classifier"]["001"], fake_cached["001"])
    assert np.array_equal(result["classifier"]["002"], fake_cached["002"])
    assert set(result["avgpool"].keys()) == {"001", "002"}


def test_extract_level_takes_named_files_and_a_cache_root_of_its_own(patch_result_dirs, tiny_image_dir, tmp_path):
    """A second stimulus set (the lab's) names its own files and must not share the default cache."""
    image_dir = tiny_image_dir(["001", "002"])
    files = {"scene_a": image_dir / "image_001.jpg", "scene_b": image_dir / "image_002.jpg"}
    extractor = StubExtractor()

    result = extractor.extract_level("Full", image_dir, ["avgpool"], cache_root=tmp_path / "other_set", images=files)

    assert set(result["avgpool"]) == {"scene_a", "scene_b"}
    assert (tmp_path / "other_set" / "stub" / "Full_avgpool.pkl").exists()
    assert not (config.EMBEDDINGS_DIR / "stub" / "Full_avgpool.pkl").exists()
    # the default cache then holds nothing for this level, so the default call computes its own (ids 001, 002)
    assert set(extractor.extract_level("Full", image_dir, ["avgpool"])["avgpool"]) == {"001", "002"}


def test_extract_level_raises_when_no_images(patch_result_dirs, tiny_image_dir):
    empty_dir = tiny_image_dir([])

    with pytest.raises(FileNotFoundError):
        StubExtractor().extract_level("Full", empty_dir, ["avgpool"])


def test_get_extractor_dispatch(patch_result_dirs, monkeypatch):
    # patch_result_dirs redirects config.DINOV2_GRID_SEARCH_RESULTS_PATH to a
    # nonexistent tmp path, and pinning is cleared here, so "dinov2"
    # deterministically falls back to the single pooled extractor regardless
    # of what's on the real machine or in config.
    monkeypatch.setattr(config, "DINOV2_PINNED_CONFIGS", ())
    assert isinstance(embeddings.get_extractor("resnet50_places365"), embeddings.ResNetPlacesExtractor)
    assert isinstance(embeddings.get_extractor("dinov2"), embeddings.DinoV2Extractor)
    with pytest.raises(ValueError):
        embeddings.get_extractor("not_a_real_backend")


def test_backend_key_maps_to_config_backend_dir_names():
    """Every concrete extractor's BACKEND_KEY must be a valid
    config.BACKEND_DIR_NAMES entry -- that's what namespaces its on-disk
    cache (and, via run.py, its diagrams sub-tree)."""
    assert embeddings.ResNetPlacesExtractor.BACKEND_KEY == "resnet50_places365"
    assert embeddings.DinoV2Extractor.BACKEND_KEY == "dinov2"
    assert embeddings.DinoV2GridExtractor.BACKEND_KEY == "dinov2"
    for backend_key in (
        embeddings.ResNetPlacesExtractor.BACKEND_KEY,
        embeddings.DinoV2Extractor.BACKEND_KEY,
        embeddings.DinoV2GridExtractor.BACKEND_KEY,
    ):
        assert backend_key in config.BACKEND_DIR_NAMES


def test_extract_level_nests_cache_under_backend_dir_name(patch_result_dirs, tiny_image_dir):
    """DinoV2Extractor and DinoV2GridExtractor share BACKEND_KEY == "dinov2",
    so they must also share the same on-disk cache sub-dir -- that's what
    lets the main pipeline reuse the supplemental grid search's already-
    computed embeddings for free (see embeddings.dinov2_grid_layers)."""
    image_dir = tiny_image_dir(["001", "002"])
    StubExtractor().extract_level("Full", image_dir, ["avgpool"])

    assert (config.EMBEDDINGS_DIR / "stub" / "Full_avgpool.pkl").exists()
    assert not (config.EMBEDDINGS_DIR / "Full_avgpool.pkl").exists()


def test_get_extractor_uses_grid_extractor_when_grid_search_present(patch_result_dirs):
    grid_results = pd.DataFrame([
        {"config_key": "block11_cls", "block": 11.0, "pooling": "cls", "avg_vector_r": 0.5},
    ])
    config.DINOV2_GRID_SEARCH_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid_results.to_pickle(config.DINOV2_GRID_SEARCH_RESULTS_PATH)

    assert isinstance(embeddings.get_extractor("dinov2"), embeddings.DinoV2GridExtractor)


# -- DinoV2GridExtractor: every transformer block x pooling combination in
# one forward pass (moved here from supplemental/tests/test_extractor.py
# alongside the class itself) --------------------------------------------


class _FakeDinoV2Model:
    """Mimics dinov2_vits14's get_intermediate_layers signature/return shape
    (verified against the real model: a tuple of (patch_tokens, cls_token)
    pairs, one per requested block) without any real weights or network."""

    embed_dim = 384
    n_patches = 16  # arbitrary small patch-token count for a fast test

    def get_intermediate_layers(self, x, n, return_class_token, norm):
        assert return_class_token is True
        assert norm is True
        batch_size = x.shape[0]
        return tuple(
            (torch.rand(batch_size, self.n_patches, self.embed_dim), torch.rand(batch_size, self.embed_dim))
            for _ in n
        )


def test_dinov2_grid_extractor_all_keys_covers_every_block_pooling_combination():
    assert len(embeddings.DinoV2GridExtractor.ALL_KEYS) == config.DINOV2_GRID_N_BLOCKS * len(config.DINOV2_GRID_POOLINGS)
    assert len(set(embeddings.DinoV2GridExtractor.ALL_KEYS)) == len(embeddings.DinoV2GridExtractor.ALL_KEYS)
    assert "block01_cls" in embeddings.DinoV2GridExtractor.ALL_KEYS
    assert "block12_mean" in embeddings.DinoV2GridExtractor.ALL_KEYS


def test_dinov2_grid_extractor_forward_batch_returns_all_block_pooling_keys_with_correct_shape():
    extractor = embeddings.DinoV2GridExtractor()
    model = _FakeDinoV2Model()
    batch = [torch.rand(3, 224, 224) for _ in range(4)]

    out = extractor._forward_batch(model, batch)

    assert set(out.keys()) == set(embeddings.DinoV2GridExtractor.ALL_KEYS)
    for vec in out.values():
        assert vec.shape == (4, model.embed_dim)


def test_dinov2_grid_extractor_forward_batch_mean_pooling_averages_patch_tokens():
    extractor = embeddings.DinoV2GridExtractor()

    class _FixedModel:
        embed_dim = 2

        def get_intermediate_layers(self, x, n, return_class_token, norm):
            batch_size = x.shape[0]
            patch_tokens = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]] * batch_size)  # mean over dim=1 -> [2, 3]
            cls_token = torch.zeros(batch_size, 2)
            return tuple((patch_tokens, cls_token) for _ in n)

    out = extractor._forward_batch(_FixedModel(), [torch.rand(3, 224, 224)])

    assert out["block01_mean"].tolist() == [[2.0, 3.0]]
    assert out["block01_cls"].tolist() == [[0.0, 0.0]]
    # every block gets the same fixed values from this fake model
    assert out["block12_mean"].tolist() == [[2.0, 3.0]]


# -- dinov2_grid_layers() / dinov2_grid_layer_colors(): picking + coloring
# the top grid-search configs from a dumped grid_results.pkl ---------------


def _write_grid_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_pickle(path)


def test_dinov2_grid_layers_returns_none_when_file_missing(patch_result_dirs):
    assert embeddings.dinov2_grid_layers() is None


def test_dinov2_grid_layers_returns_none_when_only_baseline_rows_present(patch_result_dirs):
    _write_grid_results(config.DINOV2_GRID_SEARCH_RESULTS_PATH, [
        {"config_key": "resnet50_places365_avgpool", "block": None, "pooling": "resnet50_places365_avgpool", "avg_vector_r": 0.9},
    ])
    assert embeddings.dinov2_grid_layers() is None


def test_dinov2_grid_layers_ranks_by_avg_vector_r_pooled_across_rows(patch_result_dirs):
    _write_grid_results(config.DINOV2_GRID_SEARCH_RESULTS_PATH, [
        {"config_key": "block11_cls", "block": 11.0, "pooling": "cls", "avg_vector_r": 0.8},
        {"config_key": "block11_cls", "block": 11.0, "pooling": "cls", "avg_vector_r": 0.4},  # avg -> 0.6
        {"config_key": "block12_mean", "block": 12.0, "pooling": "mean", "avg_vector_r": 0.9},
        {"config_key": "block01_cls", "block": 1.0, "pooling": "cls", "avg_vector_r": 0.1},
        {"config_key": "resnet50_places365_avgpool", "block": None, "pooling": "resnet50_places365_avgpool", "avg_vector_r": 0.99},
    ])

    layers = embeddings.dinov2_grid_layers(top_n=2)

    assert list(layers.keys()) == ["block12_mean", "block11_cls"]
    assert layers == {"block12_mean": "Block12_Mean", "block11_cls": "Block11_Cls"}


def test_dinov2_grid_layers_defaults_top_n_to_config(patch_result_dirs, monkeypatch):
    monkeypatch.setattr(config, "DINOV2_GRID_TOP_N", 1)
    _write_grid_results(config.DINOV2_GRID_SEARCH_RESULTS_PATH, [
        {"config_key": "block11_cls", "block": 11.0, "pooling": "cls", "avg_vector_r": 0.8},
        {"config_key": "block12_mean", "block": 12.0, "pooling": "mean", "avg_vector_r": 0.9},
    ])

    assert list(embeddings.dinov2_grid_layers().keys()) == ["block12_mean"]


# -- dinov2_pinned_layers() / dinov2_layers(): the a-priori config set, which
# takes precedence over any grid-search ranking ----------------------------


def test_pinned_layers_returns_configs_in_declared_order(monkeypatch):
    monkeypatch.setattr(config, "DINOV2_PINNED_CONFIGS", ("block04_cls", "block04_mean", "block12_cls"))
    monkeypatch.setattr(config, "DINOV2_PINNED_DEFAULT_CONFIG", "block12_cls")

    layers = embeddings.dinov2_pinned_layers()

    assert list(layers.keys()) == ["block04_cls", "block04_mean", "block12_cls"]
    assert layers["block04_mean"] == "Block04_Mean"


def test_pinned_layers_none_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "DINOV2_PINNED_CONFIGS", ())
    assert embeddings.dinov2_pinned_layers() is None


def test_pinned_layers_rejects_unknown_config_key(monkeypatch):
    monkeypatch.setattr(config, "DINOV2_PINNED_CONFIGS", ("block04_cls", "block99_cls"))
    with pytest.raises(ValueError, match="unknown config key"):
        embeddings.dinov2_pinned_layers()


def test_pinned_layers_rejects_default_outside_the_pinned_set(monkeypatch):
    monkeypatch.setattr(config, "DINOV2_PINNED_CONFIGS", ("block04_cls",))
    monkeypatch.setattr(config, "DINOV2_PINNED_DEFAULT_CONFIG", "block12_cls")
    with pytest.raises(ValueError, match="DINOV2_PINNED_DEFAULT_CONFIG"):
        embeddings.dinov2_pinned_layers()


def test_dinov2_layers_prefers_pinned_over_grid_results(patch_result_dirs, monkeypatch):
    """The whole point of pinning: a grid_results.pkl on disk must not
    influence which configs the main pipeline reports."""
    _write_grid_results(config.DINOV2_GRID_SEARCH_RESULTS_PATH, [
        {"config_key": "block11_cls", "block": 11.0, "pooling": "cls", "avg_vector_r": 0.99},
    ])
    monkeypatch.setattr(config, "DINOV2_PINNED_CONFIGS", ("block04_cls", "block12_cls"))
    monkeypatch.setattr(config, "DINOV2_PINNED_DEFAULT_CONFIG", "block12_cls")

    assert list(embeddings.dinov2_layers().keys()) == ["block04_cls", "block12_cls"]


def test_dinov2_layers_falls_back_to_grid_when_pinning_disabled(patch_result_dirs, monkeypatch):
    _write_grid_results(config.DINOV2_GRID_SEARCH_RESULTS_PATH, [
        {"config_key": "block11_cls", "block": 11.0, "pooling": "cls", "avg_vector_r": 0.99},
    ])
    monkeypatch.setattr(config, "DINOV2_PINNED_CONFIGS", ())

    assert list(embeddings.dinov2_layers().keys()) == ["block11_cls"]


def test_shipped_pinned_config_crosses_depth_and_pooling():
    """Guards the a-priori design itself: every pinned depth must appear with
    both poolings, so depth and pooling stay separable in the figures."""
    layers = embeddings.dinov2_pinned_layers()
    assert layers is not None, "the shipped config pins a set; see config.DINOV2_PINNED_CONFIGS"

    by_depth = {}
    for key in layers:
        block, pooling = key.split("_")
        by_depth.setdefault(block, set()).add(pooling)

    assert len(by_depth) == 3
    assert all(poolings == set(config.DINOV2_GRID_POOLINGS) for poolings in by_depth.values())
    assert len(config.DINOV2_GRID_TOP_N_COLORS) >= len(layers)


def test_dinov2_grid_layer_colors_assigns_in_ranked_order():
    layers = {"block12_mean": "Block12_Mean", "block11_cls": "Block11_Cls"}
    colors = embeddings.dinov2_grid_layer_colors(layers)
    assert colors == {
        "block12_mean": config.DINOV2_GRID_TOP_N_COLORS[0],
        "block11_cls": config.DINOV2_GRID_TOP_N_COLORS[1],
    }
