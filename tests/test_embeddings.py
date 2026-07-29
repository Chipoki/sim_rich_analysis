import pickle

import numpy as np
import pytest

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
        assert (config.EMBEDDINGS_DIR / f"Full_{layer}.pkl").exists()


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
    cache_path = config.EMBEDDINGS_DIR / "Full_classifier.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(fake_cached, f)

    extractor = StubExtractor()
    result = extractor.extract_level("Full", image_dir, ["avgpool", "classifier"])

    assert extractor.forward_calls == 1  # only "avgpool" was missing
    assert np.array_equal(result["classifier"]["001"], fake_cached["001"])
    assert np.array_equal(result["classifier"]["002"], fake_cached["002"])
    assert set(result["avgpool"].keys()) == {"001", "002"}


def test_extract_level_raises_when_no_images(patch_result_dirs, tiny_image_dir):
    empty_dir = tiny_image_dir([])

    with pytest.raises(FileNotFoundError):
        StubExtractor().extract_level("Full", empty_dir, ["avgpool"])


def test_get_extractor_dispatch():
    assert isinstance(embeddings.get_extractor("resnet50_places365"), embeddings.ResNetPlacesExtractor)
    assert isinstance(embeddings.get_extractor("dinov2"), embeddings.DinoV2Extractor)
    with pytest.raises(ValueError):
        embeddings.get_extractor("not_a_real_backend")
