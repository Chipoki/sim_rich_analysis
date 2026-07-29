"""
CPU-only image embedding extraction with on-disk caching.

Two swappable backends behind one interface:
  - ResNetPlacesExtractor (default): ResNet50 trained on Places365, using
    the local resnet50_places365.pth.tar weights. Exposes the 3 layers the
    original MATLAB analysis examined:
      * "avgpool"    -- 2048-d global-average-pooled penultimate features
      * "layer4"     -- flattened last conv block (2048x7x7 = 100352-d)
      * "classifier" -- 365-d final-layer (Places365) logits
  - DinoV2Extractor: a single-layer (patch-averaged) embedding from a
    torch.hub DINOv2 backbone. Needs internet on first run to fetch
    pretrained weights.
  - DinoV3Extractor: same idea, one layer newer (DINOv3 backbone). Needs
    internet on first run to fetch pretrained weights.

Caching: for a given (level, layer), embeddings for every image in that
level's folder are cached to
`<EMBEDDINGS_DIR>/<level>_<layer>.pkl` as {image_id: np.ndarray}, so
re-running the pipeline is instant after the first pass.
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image

import config
import progress


class EmbeddingExtractor(ABC):
    """Shared batching/caching logic; subclasses provide model + layers."""

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
        cache_paths = {layer: config.EMBEDDINGS_DIR / f"{level}_{layer}.pkl" for layer in layers}
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


class DinoV3Extractor(EmbeddingExtractor):
    """Single-layer (patch-token-averaged) DINOv3 embedding. Needs internet
    on first run to download pretrained weights via torch.hub. The upstream
    repo's hubconf.py eagerly imports its full eval suite (classifiers/
    depthers/detectors/dinotxt/segmentors), not just the plain backbone
    called here, so loading it needs omegaconf/torchmetrics/termcolor on
    top of torch/torchvision even though this project never touches those
    other entrypoints -- see environment.yml."""

    def _load_model(self):
        import torch

        model = torch.hub.load("facebookresearch/dinov3", config.DINOV3_MODEL_NAME)
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
        return {"dinov3": feats.numpy()}


def get_extractor(backend: str = config.BACKEND) -> EmbeddingExtractor:
    if backend == "resnet50_places365":
        return ResNetPlacesExtractor()
    if backend == "dinov2":
        return DinoV2Extractor()
    if backend == "dinov3":
        return DinoV3Extractor()
    raise ValueError(f"Unknown backend: {backend}")
