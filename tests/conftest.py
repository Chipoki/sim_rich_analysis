"""
Shared fixtures for the pipeline's pytest suite.

Everything here is synthetic (tiny generated JSON trial streams, tiny
generated images, a stub embedding extractor with no real torch/ResNet
compute) so the fast suite runs offline and in well under a second per test,
per CLAUDE.md's guardrail to prefer synthetic smoke tests over full-scale
runs for performance-heavy pipeline code.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: plt.show() must never block a test run

import numpy as np
import pytest
from PIL import Image

# tests/ has no __init__.py, so the project root (flat modules: config.py,
# parsing.py, ...) isn't on sys.path by pytest's default import mode.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import embeddings  # noqa: E402

TIMING_DEFAULTS = {
    f: 100
    for f in [
        "fixation_actual_ms", "fixation_actual_frames",
        "image1_actual_ms", "image1_actual_frames",
        "mask1_actual_ms", "mask1_actual_frames",
        "image2_actual_ms", "image2_actual_frames",
        "mask2_actual_ms", "mask2_actual_frames",
    ]
}


def make_participant_trials(
    participant_id: str,
    condition: str,
    pair_ids: list[str],
    ratings: dict[str, int] | None = None,
    letter_correct: dict[str, bool] | None = None,
    shuffle_order: bool = False,
) -> list[dict]:
    """
    Build one participant's raw jsPsych-style trial-object array in the shape
    parsing.py expects: session_metadata, participant_info, then per pair_id
    a psychophysics (+ letter_question, letters condition only) +
    similarity_question block, in that order.

    `shuffle_order` reverses the pair order within the stream while keeping
    trial_index increasing, so tests can check that parsing associates rows
    by pair_id and never relies on trial_index/position (see parsing.py).
    """
    ratings = ratings or {pid: 3 for pid in pair_ids}
    letter_correct = letter_correct if letter_correct is not None else {pid: True for pid in pair_ids}

    trials = [
        {"trial_type": "session_metadata", "run_id": f"run_{participant_id}"},
        {
            "trial_type": "participant_info",
            "participant_id": participant_id,
            "participant_age": 25,
            "participant_gender": "female",
            "participant_handedness": "right",
        },
    ]

    order = list(reversed(pair_ids)) if shuffle_order else list(pair_ids)

    for i, pid in enumerate(order):
        id1, id2 = pid.split("_")
        psychophysics = {
            "trial_type": "psychophysics",
            "phase": "main",
            "pair_id": pid,
            "image1_id": id1,
            "image2_id": id2,
            "block_index": 0,
            "trial_index_within_block": i,
            "trial_index": 100 + i,
            **TIMING_DEFAULTS,
        }
        if condition == "letters":
            psychophysics.update(
                {
                    "letter1_shown": "L",
                    "letter2_shown": "T",
                    "letter1_rotation_deg": 0,
                    "letter2_rotation_deg": 90,
                }
            )
        trials.append(psychophysics)

        if condition == "letters":
            trials.append(
                {
                    "trial_type": "letter_question",
                    "pair_id": pid,
                    "trial_index": 200 + i,
                    "letter_discrimination_correct": letter_correct[pid],
                    "rt": 500,
                }
            )

        trials.append(
            {
                "trial_type": "similarity_question",
                "pair_id": pid,
                "trial_index": 300 + i,
                "participant_similarity_rating": ratings[pid],
                "rt": 800,
            }
        )

    return trials


@pytest.fixture
def pair_ids() -> list[str]:
    return ["001_002", "003_004", "005_006"]


@pytest.fixture
def write_result_file(tmp_path):
    """Write a synthetic result file (one JSON array per line, one line per
    participant) for a given condition; returns the file path."""

    def _write(filename: str, condition: str, participant_specs: list[dict]) -> Path:
        path = tmp_path / filename
        with open(path, "w") as f:
            for spec in participant_specs:
                trials = make_participant_trials(condition=condition, **spec)
                f.write(json.dumps(trials) + "\n")
        return path

    return _write


@pytest.fixture
def tiny_image_dir(tmp_path):
    """Write tiny synthetic RGB JPGs named image_XXX.jpg for the given ids."""

    def _make(image_ids: list[str], subdir: str = "images") -> Path:
        d = tmp_path / subdir
        d.mkdir(parents=True, exist_ok=True)
        for img_id in image_ids:
            rng = np.random.default_rng(int(img_id))
            arr = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"image_{img_id}.jpg")
        return d

    return _make


@pytest.fixture
def patch_result_dirs(monkeypatch, tmp_path):
    """Redirect every config output directory to tmp_path so tests never
    write into the real web_analysis_results/ sibling folder."""
    results_dir = tmp_path / "web_analysis_results"
    data_dir = results_dir / "data"
    embeddings_dir = data_dir / "embeddings"
    diagrams_dir = results_dir / "diagrams"
    for d in (results_dir, data_dir, embeddings_dir, diagrams_dir):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "EMBEDDINGS_DIR", embeddings_dir)
    monkeypatch.setattr(config, "DIAGRAMS_DIR", diagrams_dir)
    # Also redirect the DINOv2 grid-search results path so tests never pick
    # up a real supplemental/dinov2_layer_grid_search/data/grid_results.pkl
    # that happens to exist on the machine running the suite -- this path
    # doesn't exist under tmp_path, so it deterministically exercises the
    # "grid search not run yet" fallback unless a test explicitly writes one.
    monkeypatch.setattr(
        config,
        "DINOV2_GRID_SEARCH_RESULTS_PATH",
        results_dir / "supplemental" / "dinov2_layer_grid_search" / "data" / "grid_results.pkl",
    )
    return results_dir


class StubExtractor(embeddings.EmbeddingExtractor):
    """
    Deterministic fake extractor: no real torch/ResNet compute, so
    EmbeddingExtractor.extract_level's batching/caching logic is exercised
    without the cost (or the weights-file dependency) of a real CNN forward
    pass. Each image's embedding is derived from its own pixel content, so
    distinct images get distinct (but reproducible) vectors.
    """

    # Covers every layer key any real extractor ever requests: resnet50_places365's
    # 3 layers, each single-layer backend's own key (config.SINGLE_LAYER_BACKEND_LABELS),
    # and the a-priori dinov2 configs (config.DINOV2_PINNED_CONFIGS) that run.py asks
    # for whenever pinning is on -- so this one stub works for extract_level()
    # regardless of which backend is under test.
    LAYER_DIMS = {
        "avgpool": 8, "layer4": 16, "classifier": 4, "dinov2": 8,
        **{config_key: 8 for config_key in config.DINOV2_PINNED_CONFIGS},
    }
    BACKEND_KEY = "stub"

    def __init__(self):
        super().__init__()
        self.load_calls = 0
        self.forward_calls = 0

    def _load_model(self):
        self.load_calls += 1
        return "stub-model"

    def _preprocess(self, image):
        return np.asarray(image)

    def _forward_batch(self, model, batch):
        self.forward_calls += 1
        out = {layer: np.zeros((len(batch), dim), dtype=np.float32) for layer, dim in self.LAYER_DIMS.items()}
        for i, img in enumerate(batch):
            seed = int(np.asarray(img).sum()) % (2**32)
            rng = np.random.default_rng(seed)
            for layer, dim in self.LAYER_DIMS.items():
                out[layer][i] = rng.random(dim)
        return out


@pytest.fixture
def stub_extractor():
    return StubExtractor()
