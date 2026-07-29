"""
Turn per-image embeddings into per-pair distance/similarity vectors,
aligned 1:1 with the human similarity vectors (same pair_id order).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import progress


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    cos_sim = float(np.dot(a, b) / denom)
    return 1.0 - cos_sim


def pair_distances(embeddings: dict[str, np.ndarray], pair_ids: list[str]) -> pd.Series:
    """
    embeddings: {image_id: vector}
    pair_ids: e.g. ["093_090", ...] -- split on "_" to get the two image ids.
    Returns a Series indexed by pair_id of cosine distances.
    """
    dists = {}
    for pid in pair_ids:
        id1, id2 = pid.split("_")
        dists[pid] = cosine_distance(embeddings[id1], embeddings[id2])
    return pd.Series(dists, name="distance").sort_index()


def build_dist_vectors(
    embeddings_by_level_layer: dict[tuple[str, str], dict[str, np.ndarray]],
    pair_ids: list[str],
) -> pd.DataFrame:
    """
    embeddings_by_level_layer: {(level, layer): {image_id: vector}}
    Returns a tidy DataFrame: index = pair_id, columns = MultiIndex
    (level, layer, metric) where metric in {"distance", "similarity"}.
    """
    cols = {}
    items = progress.bar(
        embeddings_by_level_layer.items(),
        total=len(embeddings_by_level_layer),
        desc="Distance vectors",
        depth=progress.STAGE,
    )
    for (level, layer), embs in items:
        d = pair_distances(embs, pair_ids)
        cols[(level, layer, "distance")] = d
        cols[(level, layer, "similarity")] = 1.0 - d
    df = pd.DataFrame(cols)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["level", "layer", "metric"])
    return df.sort_index()


def get_similarity(dist_vectors: pd.DataFrame, level: str, layer: str) -> pd.Series:
    return dist_vectors[(level, layer, "similarity")]


def save(dist_vectors: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dist_vectors, f)


def load(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        return pickle.load(f)
