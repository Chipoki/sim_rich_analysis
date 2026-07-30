"""
Grid search over DINOv2 (block x pooling) embedding-extraction strategies,
scored by correlation against human peripheral-similarity judgments.

Standalone supplemental script -- not part of run.py's main pipeline, though
run.py's own dinov2 backend automatically picks up this script's dumped
results once they exist (see embeddings.dinov2_grid_layers()). Reuses the
main pipeline's parsing/sim_vectors/distances/analysis/embeddings modules
exactly as run.py does, feeding them a much larger set of "layers" (24 DINOv2
block x pooling combinations, computed in one forward pass per image via
embeddings.DinoV2GridExtractor, plus 2 baseline configs for comparison: the
existing resnet50_places365 avgpool and dinov2 defaults already produced by
run.py).

Run with:  python supplemental/dinov2_layer_grid_search/run_grid_search.py

Outputs (correlation tables, markdown summary, heatmap figures) are written
to a sibling-of-the-repo results tree, matching the main pipeline's
convention of never writing generated output inside the repo itself:
  <config.DINOV2_GRID_SEARCH_DATA_DIR>
  <config.DINOV2_GRID_SEARCH_DIAGRAMS_DIR>

Needs internet on first run to fetch DINOv2 weights via torch.hub (same
requirement as running the main pipeline with "dinov2" in
config.EXTRA_BACKENDS), and real src_data/ stimuli on disk.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402

import analysis  # noqa: E402
import config  # noqa: E402
import distances  # noqa: E402
import embeddings  # noqa: E402
import parsing  # noqa: E402
import sim_vectors  # noqa: E402
from embeddings import DinoV2GridExtractor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reporting  # noqa: E402

BASELINE_KEYS = ["resnet50_places365_avgpool", "dinov2_default"]
DATA_DIR = config.DINOV2_GRID_SEARCH_DATA_DIR
DIAGRAMS_DIR = config.DINOV2_GRID_SEARCH_DIAGRAMS_DIR


def compute_all_embeddings(pair_ids: list[str]) -> dict[tuple[str, str], dict]:
    grid_extractor = DinoV2GridExtractor()
    resnet_extractor = embeddings.get_extractor("resnet50_places365")
    # Not embeddings.get_extractor("dinov2") -- that dispatch is grid-results-
    # aware (see embeddings.get_extractor) and would return DinoV2GridExtractor
    # instead once this very script's own grid_results.pkl exists, breaking
    # the "dinov2_default" baseline this loop is trying to compute.
    dinov2_extractor = embeddings.DinoV2Extractor()

    out = {}
    for level in config.ALL_LEVELS:
        print(f"Extracting embeddings for level={level} ...")
        image_dir = config.SRC_DATA_DIR / config.LEVEL_FOLDERS[level]

        for layer, embs in grid_extractor.extract_level(level, image_dir, DinoV2GridExtractor.ALL_KEYS).items():
            out[(level, layer)] = embs

        resnet_by_layer = resnet_extractor.extract_level(level, image_dir, ["avgpool"])
        out[(level, "resnet50_places365_avgpool")] = resnet_by_layer["avgpool"]

        dinov2_by_layer = dinov2_extractor.extract_level(level, image_dir, ["dinov2"])
        out[(level, "dinov2_default")] = dinov2_by_layer["dinov2"]

    return out


def run():
    all_config_keys = DinoV2GridExtractor.ALL_KEYS + BASELINE_KEYS

    print("Parsing result files ...")
    df = parsing.load_all(config.RESULT_FILES)
    wide = sim_vectors.build_human_similarity_vectors(df)
    excluded = analysis.get_excluded_participants(df)
    pair_ids = sorted(df["pair_id"].unique())

    embs = compute_all_embeddings(pair_ids)
    dist_vectors = distances.build_dist_vectors(embs, pair_ids)

    scored_rows = []
    best_rows = []
    for condition in ["letters", "no_letters"]:
        for level in config.ALL_LEVELS:
            similarity_by_config = {key: distances.get_similarity(dist_vectors, level, key) for key in all_config_keys}
            scored_rows.append(reporting.score_configs(wide, condition, level, similarity_by_config, excluded[condition]))
            best_rows.append(
                reporting.best_config_per_participant(wide, condition, level, similarity_by_config, excluded[condition])
            )

    grid_results = pd.concat(scored_rows, ignore_index=True)
    best_config_df = pd.concat(best_rows, ignore_index=True)
    baseline_results = grid_results[grid_results["block"].isna()].reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    grid_results.to_csv(DATA_DIR / "grid_results.csv", index=False)
    grid_results.to_pickle(DATA_DIR / "grid_results.pkl")
    best_config_df.to_csv(DATA_DIR / "best_config_per_participant.csv", index=False)
    best_config_df.to_pickle(DATA_DIR / "best_config_per_participant.pkl")

    summary_text = reporting.build_summary_text(grid_results, best_config_df, baseline_results)
    (DATA_DIR / "summary.md").write_text(summary_text)
    print(summary_text)

    reporting.plot_heatmaps(grid_results, DIAGRAMS_DIR)
    print(f"\nDone. Tables/summary in {DATA_DIR} | heatmaps in {DIAGRAMS_DIR}")


if __name__ == "__main__":
    run()
