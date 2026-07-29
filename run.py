"""
End-to-end pipeline. Run with:  python run.py

Steps:
  1. Parse both result files into one tidy DataFrame.
  2. Build human similarity vectors (cached to data/sim_vectors.pkl).
  3. For config.BACKEND and every backend in config.EXTRA_BACKENDS (e.g.
     "dinov2"), extract embeddings for every stimulus level/layer needed
     (cached under data/embeddings/), compute distance/similarity vectors
     (cached to data/dist_vectors.pkl for config.BACKEND, data/dist_vectors_
     <backend>.pkl for each extra one).
  4. Run the analyses from spec §4, once per backend, and save one figure
     per analysis to diagrams/, organized into sub-directories by what the
     figure varies: diagrams/degradation/, diagrams/desaturation/,
     diagrams/experimental_blocks/, diagrams/individual_differences/.
     config.BACKEND's figures use those paths directly; every extra
     backend's figures go under the equivalent diagrams/<backend>/ sub-tree
     instead, so nothing is overwritten. The (purely behavioral, embedding-
     independent) letters_accuracy_by_block figure is only ever generated
     once, from config.BACKEND's run.

Progress is reported via a pinned multi-line tqdm dashboard (see progress.py):
depth 0 tracks the 4 stages above, depth 1/2 track each stage's sub-loops.
"""

import pickle

import pandas as pd

import analysis
import config
import distances
import embeddings
import parsing
import plotting
import progress
import sim_vectors

STAGES = ["Parsing", "Similarity vectors", "Embeddings", "Analysis"]


def get_pair_ids(df: pd.DataFrame) -> list[str]:
    return sorted(df["pair_id"].unique())


def _layers_for_backend(backend: str) -> dict[str, str]:
    """Internal layer key -> human label for whichever backend is active.
    resnet50_places365 exposes the 3 examined layers (config.LAYERS); every
    other backend exposes a single pooled embedding keyed by its own name."""
    if backend == "resnet50_places365":
        return config.LAYERS
    return {backend: config.SINGLE_LAYER_BACKEND_LABELS[backend]}


def _default_layer_for_backend(backend: str) -> str:
    return config.DEFAULT_LAYER if backend == "resnet50_places365" else backend


def _layer_colors_for_backend(backend: str) -> dict[str, str]:
    if backend == "resnet50_places365":
        return config.LAYER_COLORS
    return {backend: config.SINGLE_LAYER_BACKEND_COLORS[backend]}


def compute_all_embeddings(pair_ids: list[str], backend: str) -> dict[tuple[str, str], dict]:
    extractor = embeddings.get_extractor(backend)
    layers = list(_layers_for_backend(backend).keys())

    out = {}
    levels = progress.bar(
        config.ALL_LEVELS, total=len(config.ALL_LEVELS), desc=f"Embedding levels ({backend})", depth=progress.STAGE
    )
    for level in levels:
        image_dir = config.SRC_DATA_DIR / config.LEVEL_FOLDERS[level]
        by_layer = extractor.extract_level(level, image_dir, layers)
        for layer, embs in by_layer.items():
            out[(level, layer)] = embs
    return out


def _run_condition(
    condition: str,
    wide: pd.DataFrame,
    dist_vectors: pd.DataFrame,
    block_assignment: pd.DataFrame,
    df: pd.DataFrame,
    excl: set,
    backend: str,
) -> dict:
    layers = _layers_for_backend(backend)
    default_layer = _default_layer_for_backend(backend)
    default_layer_label = layers[default_layer]
    layer_colors = _layer_colors_for_backend(backend)
    # The primary backend keeps the existing un-prefixed diagram paths;
    # every extra backend (see config.EXTRA_BACKENDS) gets its own
    # diagrams/<backend>/ sub-tree so nothing overwrites the primary output.
    prefix = "" if backend == config.BACKEND else f"{backend}/"

    # (1) Across degradation levels, default layer
    r_by_level = {}
    for level in progress.bar(config.DEGRADATION_LEVELS, desc=f"{condition}: degradation levels", depth=progress.STEP):
        r_by_level[level] = analysis.per_participant_correlations(
            wide, condition, distances.get_similarity(dist_vectors, level, default_layer), excl
        )

    # (1b) Same per-participant correlation, but for every examined layer,
    # not just the default -- needed for the cross-condition per-layer
    # comparison figures built once both conditions have run (see run()).
    r_by_level_per_layer = {default_layer: r_by_level}
    other_layers = [layer for layer in layers if layer != default_layer]
    for layer in progress.bar(other_layers, desc=f"{condition}: other layers", depth=progress.STEP):
        r_by_level_per_layer[layer] = {
            level: analysis.per_participant_correlations(
                wide, condition, distances.get_similarity(dist_vectors, level, layer), excl
            )
            for level in config.DEGRADATION_LEVELS
        }

    # (2) Best layer per participant, per level (degenerate -- always the
    # single layer itself -- for single-layer backends, but kept on the same
    # code path rather than special-cased, so every backend produces the
    # same figure set).
    best_layer_by_level = {}
    for level in progress.bar(config.DEGRADATION_LEVELS, desc=f"{condition}: best layer", depth=progress.STEP):
        sim_by_layer = {layer: distances.get_similarity(dist_vectors, level, layer) for layer in layers}
        best_layer_by_level[level] = analysis.best_layer_per_participant(wide, condition, sim_by_layer, excl)
    best_r_by_level = {lv: table["pearson_r"] for lv, table in best_layer_by_level.items()}
    best_layer_cat_by_level = {lv: table["best_layer"] for lv, table in best_layer_by_level.items()}

    # (3) AVG-vector analysis
    avg_r_by_level = {}
    for level in progress.bar(config.DEGRADATION_LEVELS, desc=f"{condition}: AVG vector", depth=progress.STEP):
        avg_r = analysis.avg_vector_correlation(
            wide, condition, distances.get_similarity(dist_vectors, level, default_layer), excl
        )
        avg_r_by_level[level] = pd.Series([avg_r])

    # (4) Colors desaturation
    r_by_desat = {}
    for level in progress.bar(config.DESATURATION_LEVELS, desc=f"{condition}: desaturation", depth=progress.STEP):
        r_by_desat[level] = analysis.per_participant_correlations(
            wide, condition, distances.get_similarity(dist_vectors, level, default_layer), excl
        )

    # (5) Performance across experimental presentation blocks (Full level,
    # default layer -- the actual stimulus condition participants saw; block
    # membership is independently randomized per participant, see
    # sim_vectors.build_block_assignment).
    r_by_block_raw = analysis.per_participant_correlations_by_block(
        wide, block_assignment, condition, distances.get_similarity(dist_vectors, "Full", default_layer), excl
    )
    r_by_block = {f"Block {b}": series for b, series in r_by_block_raw.items()}

    # Reliability: LOO / noise ceiling + split-half. Both are purely
    # human-human quantities (computed from `wide` alone, no embeddings
    # involved), so they're identical across backends for the same
    # condition -- recomputed here anyway since it's cheap relative to the
    # embedding extraction this function is otherwise gated behind.
    loo = analysis.leave_one_out_reliability(wide, condition, excl)
    split_half = analysis.split_half_reliability(wide, condition, excl)
    progress.write(
        f"   [{condition}] N={len(loo)}  R_LOO(noise ceiling)={loo.mean():.3f}  split-half(SB)={split_half:.3f}"
    )
    noise_ceiling_by_level = {level: loo for level in config.DEGRADATION_LEVELS}
    # Symbolic mention of reliability provenance, shown on the noise-ceiling
    # figure below: which r values are LOO vs split-half, and split-half's
    # iteration count (config.SPLIT_HALF_N_ITER), so it's never left implicit.
    reliability_footnote = (
        f"Noise ceiling = leave-one-out (LOO) human-human reliability, R_LOO={loo.mean():.3f} (N={len(loo)}). "
        f"Split-half reliability (Spearman-Brown corrected, {config.SPLIT_HALF_N_ITER} iterations): {split_half:.3f}."
    )

    # Figures
    figure_calls = [
        lambda: plotting.bar_with_dots(
            r_by_level, f"{condition} - degradation levels (layer: {default_layer_label})",
            xlabel="Degradation level", filename=f"{prefix}degradation/{condition}_degradation_levels.png",
        ),
        lambda: plotting.bar_with_dots(
            best_r_by_level,
            f"{condition} - layer with highest correlation (of {len(layers)} examined)",
            xlabel="Degradation level",
            dot_categories_by_level=best_layer_cat_by_level,
            category_colors=layer_colors,
            category_labels=layers,
            filename=f"{prefix}degradation/{condition}_best_layer.png",
        ),
        lambda: plotting.bar_with_dots(
            avg_r_by_level, f"{condition} - degradation levels, AVG vector (layer: {default_layer_label})",
            xlabel="Degradation level", dot_label="AVG-vector value (all participants pooled)",
            filename=f"{prefix}degradation/{condition}_avg_vector.png",
        ),
        lambda: plotting.bar_with_dots(
            r_by_desat, f"colors desaturation - {condition} (layer: {default_layer_label})",
            xlabel="Desaturation level", filename=f"{prefix}desaturation/{condition}_desaturation.png",
        ),
        lambda: plotting.bar_with_noise_ceiling(
            r_by_level, noise_ceiling_by_level,
            f"{condition} - degradation levels vs noise ceiling (layer: {default_layer_label})",
            xlabel="Degradation level", filename=f"{prefix}degradation/{condition}_degradation_vs_noise_ceiling.png",
            footnote=reliability_footnote,
        ),
        lambda: plotting.bar_with_dots(
            r_by_block,
            f"{condition} - performance across experimental blocks (Full, layer: {default_layer_label})",
            xlabel="Experimental block", filename=f"{prefix}experimental_blocks/{condition}_by_block.png",
        ),
    ]
    acc_by_participant_block = None
    if condition == "letters":
        acc_by_participant_block = parsing.letter_accuracy_by_participant_and_block(df)
        # Purely behavioral (no embeddings involved) -- identical for every
        # backend, so the figure is only generated once, from the primary
        # backend's run, to avoid writing byte-identical duplicate PNGs.
        if backend == config.BACKEND:
            # Deliberately NOT excluding participants here: this is the very
            # quantity the accuracy exclusion (config.LETTER_ACCURACY_THRESHOLD)
            # is computed from, so showing everyone lets the excluded
            # participant's trajectory visibly justify the exclusion instead of
            # silently disappearing.
            acc_by_block = {
                f"Block {b}": acc_by_participant_block[b].dropna() for b in sorted(acc_by_participant_block.columns)
            }
            figure_calls.append(
                lambda: plotting.bar_with_dots(
                    acc_by_block,
                    f"letters - letter-discrimination accuracy across experimental blocks "
                    f"(all {len(acc_by_participant_block)} participants, incl. excluded)",
                    ylabel="Accuracy", xlabel="Experimental block",
                    reference_lines={0.5: "chance (2AFC)", config.LETTER_ACCURACY_THRESHOLD: "exclusion threshold"},
                    filename="experimental_blocks/letters_accuracy_by_block.png",
                )
            )

    for call in progress.bar(figure_calls, total=len(figure_calls), desc=f"{condition}: figures", depth=progress.STEP):
        call()

    # Stats note: paired test between adjacent levels
    for a, b in zip(config.DEGRADATION_LEVELS, config.DEGRADATION_LEVELS[1:]):
        note = analysis.paired_level_test(r_by_level, a, b)
        progress.write(f"   [{condition}] {a} vs {b}: Wilcoxon p={note['p_value']:.4f} (n={note['n']})")

    results = {
        "r_by_level": r_by_level,
        "r_by_level_per_layer": r_by_level_per_layer,
        "best_layer_by_level": best_layer_by_level,
        "avg_r_by_level": avg_r_by_level,
        "r_by_desat": r_by_desat,
        "r_by_block": r_by_block,
        "loo": loo,
        "split_half": split_half,
    }
    if condition == "letters":
        results["letter_accuracy_by_block"] = acc_by_participant_block
    return results


def _run_cross_condition_figures(condition_results: dict, df: pd.DataFrame, backend: str) -> None:
    """Figures that compare the two conditions directly, or relate a
    per-participant behavioral quantity (RT, letter accuracy) to model
    validity -- these need both conditions' results, so they're built after
    _run_condition has run for each (for this backend)."""
    conditions = list(condition_results.keys())
    layers = _layers_for_backend(backend)
    default_layer = _default_layer_for_backend(backend)
    default_layer_label = layers[default_layer]
    prefix = "" if backend == config.BACKEND else f"{backend}/"

    # Letters-vs-No-Letters, across degradation levels, once per examined
    # layer (not just the default) -- point (1) of the "per-layer" request:
    # does the condition gap hold up under every layer, or is it specific to
    # one representation?
    for layer in progress.bar(layers, desc="Cross-condition: per-layer comparisons", depth=progress.STAGE):
        letters_by_level = {
            lv: condition_results["letters"]["r_by_level_per_layer"][layer][lv] for lv in config.DEGRADATION_LEVELS
        }
        no_letters_by_level = {
            lv: condition_results["no_letters"]["r_by_level_per_layer"][layer][lv] for lv in config.DEGRADATION_LEVELS
        }
        plotting.bar_with_noise_ceiling(
            letters_by_level, no_letters_by_level,
            f"Letters vs No-Letters across degradation levels (layer: {layers[layer]})",
            xlabel="Degradation level",
            series_a_label="Letters", series_b_label="No-Letters",
            series_a_color="steelblue", series_b_color="salmon",
            series_a_dot_color="navy", series_b_dot_color="darkred",
            filename=f"{prefix}degradation/letters_vs_no_letters_{layer}.png",
        )

    # Individual differences: does how long someone took to judge similarity
    # relate to how well their judgments track the model?
    mean_rt_by_condition = {c: parsing.mean_similarity_rt_by_participant(df, c) for c in conditions}
    r_full_by_condition = {c: condition_results[c]["r_by_level"]["Full"] for c in conditions}
    plotting.scatter_relationship(
        mean_rt_by_condition, r_full_by_condition,
        f"Model validity vs mean response time (Full, layer: {default_layer_label})",
        xlabel="Mean similarity-judgment RT (ms)",
        filename=f"{prefix}individual_differences/r_vs_response_time.png",
    )

    # Individual differences (Letters only): does higher compliance with the
    # concurrent letter task predict better-model-tracking similarity
    # judgments? Uses every participant's accuracy (including excluded), but
    # scatter_relationship's index-intersection naturally restricts the
    # right-hand side to the included participants r_by_level already covers.
    letter_accuracy = parsing.letter_accuracy_by_participant(df)
    plotting.scatter_relationship(
        {"letters": letter_accuracy}, {"letters": condition_results["letters"]["r_by_level"]["Full"]},
        f"Model validity vs letter-discrimination accuracy (Full, layer: {default_layer_label})",
        xlabel="Letter-discrimination accuracy",
        reference_lines={config.LETTER_ACCURACY_THRESHOLD: "exclusion threshold"},
        filename=f"{prefix}individual_differences/letters_r_vs_letter_accuracy.png",
    )


def run():
    # config.BACKEND first, then any config.EXTRA_BACKENDS (e.g. "dinov2"),
    # de-duplicated in case a backend appears in both -- each one gets the
    # exact same figure set (see _run_condition/_run_cross_condition_figures),
    # with config.BACKEND alone keeping the original un-prefixed diagram
    # paths so existing outputs/consumers are unaffected.
    backends = list(dict.fromkeys([config.BACKEND] + config.EXTRA_BACKENDS))

    with progress.pipeline_bar(STAGES) as pbar:
        pbar.set_description(STAGES[0])
        df = parsing.load_all(config.RESULT_FILES)
        progress.write(f"1. Parsed {df.shape[0]} tidy trial rows across {df['participant_id'].nunique()} participants")
        pbar.update(1)

        pbar.set_description(STAGES[1])
        wide = sim_vectors.build_human_similarity_vectors(df)
        sim_vectors.save(wide, config.DATA_DIR / "sim_vectors.pkl")
        block_assignment = sim_vectors.build_block_assignment(df)
        excluded = analysis.get_excluded_participants(df)
        progress.write(f"2. Exclusions: {[(k, sorted(v)) for k, v in excluded.items()]}")
        pbar.update(1)

        pbar.set_description(STAGES[2])
        pair_ids = get_pair_ids(df)
        dist_vectors_by_backend = {}
        for backend in backends:
            embs = compute_all_embeddings(pair_ids, backend)
            dist_vectors = distances.build_dist_vectors(embs, pair_ids)
            suffix = "" if backend == config.BACKEND else f"_{backend}"
            distances.save(dist_vectors, config.DATA_DIR / f"dist_vectors{suffix}.pkl")
            dist_vectors_by_backend[backend] = dist_vectors
        pbar.update(1)

        pbar.set_description(STAGES[3])
        results = {}
        for backend in backends:
            condition_results = {}
            conditions = progress.bar(
                ["letters", "no_letters"], total=2, desc=f"Conditions ({backend})", depth=progress.STAGE
            )
            for condition in conditions:
                condition_results[condition] = _run_condition(
                    condition, wide, dist_vectors_by_backend[backend], block_assignment, df, excluded[condition], backend
                )
            _run_cross_condition_figures(condition_results, df, backend)
            results[backend] = condition_results
        pbar.update(1)

    with open(config.DATA_DIR / "analysis_results.pkl", "wb") as f:
        pickle.dump(results, f)

    progress.write(f"Done. Figures in {config.DIAGRAMS_DIR} | data in {config.DATA_DIR}")


if __name__ == "__main__":
    run()
