"""
DINOv2 embedding extractor exposing every transformer block x pooling
combination, instead of only the final pooled CLS token (embeddings.py's
DinoV2Extractor). Used by run_grid_search.py to find which depth/pooling
combination best correlates with human peripheral-similarity judgments.

Verified against the real dinov2_vits14 hub model (depth=12, embed_dim=384):
model.get_intermediate_layers(x, n=list(range(12)), return_class_token=True,
norm=True) returns a 12-tuple of (patch_tokens (B, N, C), cls_token (B, C))
pairs, one per block, in a single forward pass -- so probing all 12 blocks
costs the same wall-clock time as probing just one. Block 12's cls_token is
bit-identical to DinoV2Extractor's plain model(x) output, confirming the
kwargs match the model's own default forward path.
"""

import numpy as np
from PIL import Image

import config
import embeddings

N_BLOCKS = 12
POOLINGS = ("cls", "mean")


class DinoV2GridExtractor(embeddings.EmbeddingExtractor):
    """One embedding per (block, pooling) combination, keyed e.g. "block06_mean"."""

    ALL_KEYS = [f"block{i:02d}_{pooling}" for i in range(1, N_BLOCKS + 1) for pooling in POOLINGS]

    def _load_model(self):
        import torch

        model = torch.hub.load("facebookresearch/dinov2", config.DINOV2_MODEL_NAME)
        model.eval()
        return model

    def _preprocess(self, image: Image.Image):
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize(config.IMAGE_RESIZE),
            T.CenterCrop(config.IMAGE_CROP),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image)

    def _forward_batch(self, model, batch) -> dict[str, np.ndarray]:
        import torch

        with torch.no_grad():
            x = torch.stack(batch, dim=0)
            per_block = model.get_intermediate_layers(
                x, n=list(range(N_BLOCKS)), return_class_token=True, norm=True
            )

        out = {}
        for i, (patch_tokens, cls_token) in enumerate(per_block, start=1):
            out[f"block{i:02d}_cls"] = cls_token.numpy()
            out[f"block{i:02d}_mean"] = patch_tokens.mean(dim=1).numpy()
        return out
