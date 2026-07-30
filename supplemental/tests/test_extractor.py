import sys
from pathlib import Path

import torch

# No __init__.py in this dir (nor in tests/) -- pytest's default "prepend"
# import mode would otherwise collide both directories' bare "conftest"
# module names, so imports are resolved here per-file instead of via a
# shared supplemental/tests/conftest.py. See tests/conftest.py for the
# project's own version of this same sys.path pattern.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dinov2_layer_grid_search"))

from extractor import N_BLOCKS, POOLINGS, DinoV2GridExtractor  # noqa: E402


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


def test_all_keys_covers_every_block_pooling_combination():
    assert len(DinoV2GridExtractor.ALL_KEYS) == N_BLOCKS * len(POOLINGS)
    assert len(set(DinoV2GridExtractor.ALL_KEYS)) == len(DinoV2GridExtractor.ALL_KEYS)
    assert "block01_cls" in DinoV2GridExtractor.ALL_KEYS
    assert "block12_mean" in DinoV2GridExtractor.ALL_KEYS


def test_forward_batch_returns_all_block_pooling_keys_with_correct_shape():
    extractor = DinoV2GridExtractor()
    model = _FakeDinoV2Model()
    batch = [torch.rand(3, 224, 224) for _ in range(4)]

    out = extractor._forward_batch(model, batch)

    assert set(out.keys()) == set(DinoV2GridExtractor.ALL_KEYS)
    for vec in out.values():
        assert vec.shape == (4, model.embed_dim)


def test_forward_batch_mean_pooling_averages_patch_tokens():
    extractor = DinoV2GridExtractor()

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
