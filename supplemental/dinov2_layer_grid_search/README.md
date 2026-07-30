# DINOv2 layer x pooling grid search

Standalone, supplemental analysis -- **not** part of the main `run.py` pipeline.

`embeddings.py`'s `DinoV2Extractor` only exposes DINOv2's final pooled CLS
token as a single embedding. That's an arbitrary choice: DINOv2 ViT-S/14 has
12 transformer blocks, and peripheral/crowded vision is often modeled as
relying on local texture-statistics pooling rather than object-level
semantics -- so a shallower block, or mean-pooling over patch tokens instead
of using the CLS token, might correlate better with human peripheral-
similarity judgments than the default final-block CLS embedding, especially
at degraded/tunnel-vision levels.

This script extracts embeddings from every DINOv2 block x pooling
combination (12 blocks x {cls, mean} = 24 configs) across every level in
`config.ALL_LEVELS`, correlates each against the human similarity vectors
exactly the way `run.py`/`analysis.py` already do, and reports which
combination(s) best match human data -- both on average (AVG-vector / mean
per-participant correlation) and per-subject (which config wins for the most
participants) -- alongside a comparison against the existing
`resnet50_places365` (`avgpool`) and `dinov2` (final CLS token) defaults.

## Why this is cheap

`model.get_intermediate_layers(x, n=list(range(12)), return_class_token=True,
norm=True)` computes all 12 blocks' outputs in a **single forward pass** per
image, so probing all 24 configs costs the same wall-clock time as the
existing single-output `DinoV2Extractor` -- verified: block 12's `cls` output
is bit-identical to `DinoV2Extractor`'s plain `model(x)` output.

## Running it

```bash
python supplemental/dinov2_layer_grid_search/run_grid_search.py
```

Needs internet on first run (DINOv2 weights via `torch.hub`, same requirement
as running the main pipeline with `"dinov2"` in `config.EXTRA_BACKENDS`) and
real `src_data/` stimuli on disk. Embeddings are cached through the same
`EmbeddingExtractor.extract_level` mechanism `embeddings.py` uses, so re-runs
after the first are fast.

## Outputs

Written to `<config.RESULTS_DIR>/supplemental/dinov2_layer_grid_search/`,
consistent with the main pipeline's convention of writing all generated
output outside the repo:

```
data/
  grid_results.csv / .pkl              # long format: level, block, pooling, config_key,
                                        # condition, avg_vector_r, mean_participant_r, n_participants
  best_config_per_participant.csv / .pkl  # level, condition, participant_id, best_config, r
  summary.md                            # top configs by each metric, most-preferred
                                         # configs, baseline comparison (also printed to stdout)
diagrams/
  <level>_<condition>_<metric>.png      # block x pooling heatmap, one per level/condition/metric
```

## Files

- `extractor.py` -- `DinoV2GridExtractor`, the 24-key embedding extractor.
- `reporting.py` -- scoring (`score_configs`, `best_config_per_participant`),
  ranking (`pooled_ranking`, `top_configs`, `most_preferred_configs`), and
  plotting (`plot_heatmaps`) helpers. Kept separate from `run_grid_search.py`
  so this logic is unit-testable against small hand-built tables
  (`supplemental/tests/test_reporting.py`) without any real embeddings.
- `run_grid_search.py` -- orchestration; reuses `parsing`/`sim_vectors`/
  `distances`/`analysis` exactly as `run.py` does.
