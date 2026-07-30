"""
CPU-only image embedding extraction with on-disk caching.

Backends behind one interface:
  - ResNetPlacesExtractor (default): ResNet50 trained on Places365, using
    the local resnet50_places365.pth.tar weights. Exposes the 3 layers the
    original MATLAB analysis examined:
      * "avgpool"    -- 2048-d global-average-pooled penultimate features
      * "layer4"     -- flattened last conv block (2048x7x7 = 100352-d)
      * "classifier" -- 365-d final-layer (Places365) logits
  - DinoV2Extractor: a single-layer (patch-averaged) embedding from a
    torch.hub DINOv2 backbone. Needs internet on first run to fetch
    pretrained weights.
  - DinoV2GridExtractor: every transformer block x pooling combination from
    the same DINOv2 backbone in one forward pass. Not used directly by
    get_extractor() unless supplemental/dinov2_layer_grid_search has already
    ranked configs (see dinov2_grid_layers() below) -- otherwise DinoV2Extractor's
    single pooled embedding is used, unchanged.

Caching: for a given (level, layer), embeddings for every image in that
level's folder are cached to
`<EMBEDDINGS_DIR>/<config.BACKEND_DIR_NAMES[BACKEND_KEY]>/<level>_<layer>.pkl`
as {image_id: np.ndarray}, so re-running the pipeline is instant after the
first pass. Each concrete extractor sets its own BACKEND_KEY (matching
get_extractor()'s dispatch keys) so DinoV2Extractor and DinoV2GridExtractor
-- both DINOv2-derived -- land in the same data/embeddings/dinov2/ sub-dir
and can reuse each other's cache (see DinoV2GridExtractor's docstring).
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config
import progress


class EmbeddingExtractor(ABC):
    """Shared batching/caching logic; subclasses provide model + layers."""

    BACKEND_KEY: str  # subclasses must set -- namespaces the on-disk cache by model, see config.BACKEND_DIR_NAMES

    def __init__(self):
        self._model = None  # lazily loaded on first use, so instantiating
        # the extractor (e.g. to just check the cache) never touches torch.

    @abstractmethod
    def _load_model(self):
        """Return the loaded model (framework-specific)."""

    @abstractmethod
    def _forward_batch(self, model, batch) -> dict[str, np.ndarray]:
        """Run one batch through `model`, return {layer_name: (B, D) array}."""

    @abstractmethod
    def _preprocess(self, image: Image.Image):
        """PIL image -> framework tensor, ready to stack into a batch."""

    def extract_level(self, level: str, image_dir: Path, layers: list[str]) -> dict[str, dict[str, np.ndarray]]:
        """
        Compute (or load from cache) embeddings for every image in
        `image_dir`, for each requested layer.

        Returns {layer_name: {image_id: np.ndarray}}.
        """
        cache_dir = config.EMBEDDINGS_DIR / config.BACKEND_DIR_NAMES.get(self.BACKEND_KEY, self.BACKEND_KEY)
        cache_paths = {layer: cache_dir / f"{level}_{layer}.pkl" for layer in layers}
        cached = {}
        missing_layers = []
        for layer, path in cache_paths.items():
            if path.exists():
                with open(path, "rb") as f:
                    cached[layer] = pickle.load(f)
            else:
                missing_layers.append(layer)

        if not missing_layers:
            return cached

        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in config.IMAGE_EXTENSIONS and p.stem.startswith("image_")
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found under {image_dir}")

        if self._model is None:
            self._model = self._load_model()

        results = {layer: {} for layer in missing_layers}
        batch, batch_ids = [], []

        def flush():
            if not batch:
                return
            out = self._forward_batch(self._model, batch)
            for layer in missing_layers:
                for img_id, vec in zip(batch_ids, out[layer]):
                    results[layer][img_id] = vec
            batch.clear()
            batch_ids.clear()

        for p in progress.bar(image_paths, desc=f"{level} images", depth=progress.STEP):
            img_id = p.stem.replace("image_", "")
            with Image.open(p) as im:
                im = im.convert("RGB")
                batch.append(self._preprocess(im))
            batch_ids.append(img_id)
            if len(batch) >= config.BATCH_SIZE:
                flush()
        flush()

        for layer, embs in results.items():
            cache_paths[layer].parent.mkdir(parents=True, exist_ok=True)
            with open(cache_paths[layer], "wb") as f:
                pickle.dump(embs, f)
            cached[layer] = embs

        return cached


class ResNetPlacesExtractor(EmbeddingExtractor):
    """ResNet50 trained on Places365, run on CPU."""

    BACKEND_KEY = "resnet50_places365"

    def _load_model(self):
        import torch
        import torchvision

        model = torchvision.models.resnet50(num_classes=config.RESNET_NUM_CLASSES)
        checkpoint = torch.load(config.RESNET_WEIGHTS_PATH, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()

        # Forward hooks to grab layer4's raw output alongside avgpool.
        self._layer4_out = {}

        def hook(_module, _input, output):
            self._layer4_out["value"] = output

        model.layer4.register_forward_hook(hook)
        return model

    def _preprocess(self, image: Image.Image):
        import torch
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
            logits = model(x)  # triggers the layer4 hook as a side effect
            layer4 = self._layer4_out["value"]  # (B, 2048, 7, 7)
            avgpool = torch.flatten(
                torch.nn.functional.adaptive_avg_pool2d(layer4, 1), 1
            )  # (B, 2048)

        return {
            "avgpool": avgpool.numpy(),
            "layer4": layer4.flatten(start_dim=1).numpy(),  # (B, 100352)
            "classifier": logits.numpy(),  # (B, 365)
        }


class DinoV2Extractor(EmbeddingExtractor):
    """Single-layer (patch-token-averaged) DINOv2 embedding. Needs internet
    on first run to download pretrained weights via torch.hub."""

    BACKEND_KEY = "dinov2"

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
            feats = model(x)  # (B, D) pooled CLS-token embedding
        return {"dinov2": feats.numpy()}


class DinoV2GridExtractor(EmbeddingExtractor):
    """DINOv2 embedding exposing every transformer block x pooling
    combination (config.DINOV2_GRID_N_BLOCKS x config.DINOV2_GRID_POOLINGS),
    computed in one forward pass, instead of only the final pooled CLS token
    (DinoV2Extractor above). get_extractor() switches to this in place of
    DinoV2Extractor whenever dinov2_grid_layers() finds the supplemental grid
    search has already ranked configs -- extract_level's inherited
    caching/batching already filters down to whichever layer subset the
    caller requests, so this only ever computes the configs actually asked
    for (and reuses the grid search's own cache if already populated)."""

    BACKEND_KEY = "dinov2"  # same subdir as DinoV2Extractor -- same underlying model, shared cache

    ALL_KEYS = [
        f"block{i:02d}_{pooling}"
        for i in range(1, config.DINOV2_GRID_N_BLOCKS + 1)
        for pooling in config.DINOV2_GRID_POOLINGS
    ]

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
                x, n=list(range(config.DINOV2_GRID_N_BLOCKS)), return_class_token=True, norm=True
            )

        out = {}
        for i, (patch_tokens, cls_token) in enumerate(per_block, start=1):
            # cls_token is a non-contiguous view (out[:, 0]) into the full
            # (B, N, C) per-block activation -- .numpy() on it would share
            # that whole buffer, keeping all N patch tokens alive per stored
            # image just to retrieve the 1 cls row. .contiguous() forces a
            # real copy of just the (B, C) slice so the parent buffer can be
            # freed.
            out[f"block{i:02d}_cls"] = cls_token.contiguous().numpy()
            out[f"block{i:02d}_mean"] = patch_tokens.mean(dim=1).numpy()
        return out


def dinov2_grid_layers(top_n: int = None) -> dict[str, str] | None:
    """Top `top_n` (default config.DINOV2_GRID_TOP_N) DINOv2 (block, pooling)
    configs from the supplemental grid search's dumped results
    (config.DINOV2_GRID_SEARCH_RESULTS_PATH), ranked by avg_vector_r averaged
    across every level/condition present in that file -- the same pooled
    ranking supplemental/dinov2_layer_grid_search/reporting.py's
    pooled_ranking() uses. Returns None if the grid search hasn't been run
    yet (file missing) or has no grid rows, so callers fall back to the
    single pooled "dinov2" embedding.

    Return value: {config_key: human_label}, ordered best-first, e.g.
    {"block11_cls": "Block11_Cls", "block12_mean": "Block12_Mean", ...} --
    mirrors config.LAYERS' internal-key -> human-label shape so it can be
    used as a drop-in "layers" dict wherever config.LAYERS is.
    """
    if top_n is None:
        top_n = config.DINOV2_GRID_TOP_N

    path = config.DINOV2_GRID_SEARCH_RESULTS_PATH
    if not path.exists():
        return None

    grid_results = pd.read_pickle(path)
    grid_only = grid_results[grid_results["block"].notna()]
    if grid_only.empty:
        return None

    ranked = (
        grid_only.groupby("config_key", as_index=False)["avg_vector_r"]
        .mean()
        .sort_values("avg_vector_r", ascending=False)
        .head(top_n)
    )
    return {row.config_key: _dinov2_config_label(row.config_key) for row in ranked.itertuples()}


def _dinov2_config_label(config_key: str) -> str:
    """"block11_cls" -> "Block11_Cls" (Title_Case, matching config.LAYERS' style)."""
    block, pooling = config_key.split("_")
    return f"{block.capitalize()}_{pooling.capitalize()}"


def dinov2_grid_layer_colors(layers: dict[str, str]) -> dict[str, str]:
    """Assign config.DINOV2_GRID_TOP_N_COLORS to grid-selected configs, in
    ranked order (the order `layers` -- from dinov2_grid_layers() -- iterates)."""
    return dict(zip(layers, config.DINOV2_GRID_TOP_N_COLORS))


def get_extractor(backend: str = config.BACKEND) -> EmbeddingExtractor:
    if backend == "resnet50_places365":
        return ResNetPlacesExtractor()
    if backend == "dinov2":
        return DinoV2GridExtractor() if dinov2_grid_layers() is not None else DinoV2Extractor()
    raise ValueError(f"Unknown backend: {backend}")
