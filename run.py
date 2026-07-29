"""
End-to-end pipeline. Run with:  python run.py

Steps:
  1. Parse both result files into one tidy DataFrame.
  2. Build human similarity vectors (cached to data/sim_vectors.pkl).
  3. Extract embeddings for every stimulus level/layer needed (cached under
     data/embeddings/), compute distance/similarity vectors (cached to
     data/dist_vectors.pkl).
  4. Run the analyses from spec §4 and save one figure per analysis to
     diagrams/, organized into sub-directories by what the figure varies:
     diagrams/degradation/, diagrams/desaturation/, diagrams/experimental_blocks/,
     diagrams/individual_differences/.

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


def compute_all_embeddings(pair_ids: list[str]) -> dict[tuple[str, str], dict]:
    extractor = embeddings.get_extractor()
    layers = list(config.LAYERS.keys()) if config.BACKEND == "resnet50_places365" else ["dinov2"]

    out = {}
    levels = progress.bar(config.ALL_LEVELS, total=len(config.ALL_LEVELS), desc="Embedding levels", depth=progress.STAGE)
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
) -> dict:
    default_layer_label = config.LAYERS[config.DEFAULT_LAYER]

    # (1) Across degradation levels, default layer
    r_by_level = {}
    for level in progress.bar(config.DEGRADATION_LEVELS, desc=f"{condition}: degradation levels", depth=progress.STEP):
        r_by_level[level] = analysis.per_participant_correlations(
            wide, condition, distances.get_similarity(dist_vectors, level, config.DEFAULT_LAYER), excl
        )

    # (1b) Same per-participant correlation, but for every examined layer,
    # not just the default -- needed for the cross-condition per-layer
    # comparison figures built once both conditions have run (see run()).
    r_by_level_per_layer = {config.DEFAULT_LAYER: r_by_level}
    other_layers = [layer for layer in config.LAYERS if layer != config.DEFAULT_LAYER]
    for layer in progress.bar(other_layers, desc=f"{condition}: other layers", depth=progress.STEP):
        r_by_level_per_layer[layer] = {
            level: analysis.per_participant_correlations(
                wide, condition, distances.get_similarity(dist_vectors, level, layer), excl
            )
            for level in config.DEGRADATION_LEVELS
        }

    # (2) Best layer per participant, per level
    best_layer_by_level = {}
    for level in progress.bar(config.DEGRADATION_LEVELS, desc=f"{condition}: best layer", depth=progress.STEP):
        sim_by_layer = {layer: distances.get_similarity(dist_vectors, level, layer) for layer in config.LAYERS}
        best_layer_by_level[level] = analysis.best_layer_per_participant(wide, condition, sim_by_layer, excl)
    best_r_by_level = {lv: table["pearson_r"] for lv, table in best_layer_by_level.items()}
    best_layer_cat_by_level = {lv: table["best_layer"] for lv, table in best_layer_by_level.items()}

    # (3) AVG-vector analysis
    avg_r_by_level = {}
    for level in progress.bar(config.DEGRADATION_LEVELS, desc=f"{condition}: AVG vector", depth=progress.STEP):
        avg_r = analysis.avg_vector_correlation(
            wide, condition, distances.get_similarity(dist_vectors, level, config.DEFAULT_LAYER), excl
        )
        avg_r_by_level[level] = pd.Series([avg_r])

    # (4) Colors desaturation
    r_by_desat = {}
    for level in progress.bar(config.DESATURATION_LEVELS, desc=f"{condition}: desaturation", depth=progress.STEP):
        r_by_desat[level] = analysis.per_participant_correlations(
            wide, condition, distances.get_similarity(dist_vectors, level, config.DEFAULT_LAYER), excl
        )

    # (5) Performance across experimental presentation blocks (Full level,
    # default layer -- the actual stimulus condition participants saw; block
    # membership is independently randomized per participant, see
    # sim_vectors.build_block_assignment).
    r_by_block_raw = analysis.per_participant_correlations_by_block(
        wide, block_assignment, condition, distances.get_similarity(dist_vectors, "Full", config.DEFAULT_LAYER), excl
    )
    r_by_block = {f"Block {b}": series for b, series in r_by_block_raw.items()}

    # Reliability: LOO / noise ceiling + split-half
    loo = analysis.leave_one_out_reliability(wide, condition, excl)
    split_half = analysis.split_half_reliability(wide, condition, excl)
    progress.write(
        f"   [{condition}] N={len(loo)}  R_LOO(noise ceiling)={loo.mean():.3f}  split-half(SB)={split_half:.3f}"
    )
    noise_ceiling_by_level = {level: loo for level in config.DEGRADATION_LEVELS}

    # Figures
    figure_calls = [
        lambda: plotting.bar_with_dots(
            r_by_level, f"{condition} - degradation levels (layer: {default_layer_label})",
            xlabel="Degradation level", filename=f"degradation/{condition}_degradation_levels.png",
        ),
        lambda: plotting.bar_with_dots(
            best_r_by_level,
            f"{condition} - layer with highest correlation (of {len(config.LAYERS)} examined)",
            xlabel="Degradation level",
            dot_categories_by_level=best_layer_cat_by_level,
            category_colors=config.LAYER_COLORS,
            category_labels=config.LAYERS,
            filename=f"degradation/{condition}_best_layer.png",
        ),
        lambda: plotting.bar_with_dots(
            avg_r_by_level, f"{condition} - degradation levels, AVG vector (layer: {default_layer_label})",
            xlabel="Degradation level", dot_label="AVG-vector value (all participants pooled)",
            filename=f"degradation/{condition}_avg_vector.png",
        ),
        lambda: plotting.bar_with_dots(
            r_by_desat, f"colors desaturation - {condition} (layer: {default_layer_label})",
            xlabel="Desaturation level", filename=f"desaturation/{condition}_desaturation.png",
        ),
        lambda: plotting.bar_with_noise_ceiling(
            r_by_level, noise_ceiling_by_level,
            f"{condition} - degradation levels vs noise ceiling (layer: {default_layer_label})",
            xlabel="Degradation level", filename=f"degradation/{condition}_degradation_vs_noise_ceiling.png",
        ),
        lambda: plotting.bar_with_dots(
            r_by_block,
            f"{condition} - performance across experimental blocks (Full, layer: {default_layer_label})",
            xlabel="Experimental block", filename=f"experimental_blocks/{condition}_by_block.png",
        ),
    ]
    if condition == "letters":
        acc_by_participant_block = parsing.letter_accuracy_by_participant_and_block(df)
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


def _run_cross_condition_figures(results: dict, df: pd.DataFrame) -> None:
    """Figures that compare the two conditions directly, or relate a
    per-participant behavioral quantity (RT, letter accuracy) to model
    validity -- these need both conditions' results, so they're built after
    _run_condition has run for each."""
    conditions = list(results.keys())

    # Letters-vs-No-Letters, across degradation levels, once per examined
    # layer (not just the default) -- point (1) of the "per-layer" request:
    # does the condition gap hold up under every layer, or is it specific to
    # one representation?
    for layer in progress.bar(config.LAYERS, desc="Cross-condition: per-layer comparisons", depth=progress.STAGE):
        letters_by_level = {lv: results["letters"]["r_by_level_per_layer"][layer][lv] for lv in config.DEGRADATION_LEVELS}
        no_letters_by_level = {
            lv: results["no_letters"]["r_by_level_per_layer"][layer][lv] for lv in config.DEGRADATION_LEVELS
        }
        plotting.bar_with_noise_ceiling(
            letters_by_level, no_letters_by_level,
            f"Letters vs No-Letters across degradation levels (layer: {config.LAYERS[layer]})",
            xlabel="Degradation level",
            series_a_label="Letters", series_b_label="No-Letters",
            series_a_color="steelblue", series_b_color="salmon",
            series_a_dot_color="navy", series_b_dot_color="darkred",
            filename=f"degradation/letters_vs_no_letters_{layer}.png",
        )

    # Individual differences: does how long someone took to judge similarity
    # relate to how well their judgments track the model?
    mean_rt_by_condition = {c: parsing.mean_similarity_rt_by_participant(df, c) for c in conditions}
    r_full_by_condition = {c: results[c]["r_by_level"]["Full"] for c in conditions}
    plotting.scatter_relationship(
        mean_rt_by_condition, r_full_by_condition,
        "Model validity vs mean response time (Full, layer: " + config.LAYERS[config.DEFAULT_LAYER] + ")",
        xlabel="Mean similarity-judgment RT (ms)",
        filename="individual_differences/r_vs_response_time.png",
    )

    # Individual differences (Letters only): does higher compliance with the
    # concurrent letter task predict better-model-tracking similarity
    # judgments? Uses every participant's accuracy (including excluded), but
    # scatter_relationship's index-intersection naturally restricts the
    # right-hand side to the included participants r_by_level already covers.
    letter_accuracy = parsing.letter_accuracy_by_participant(df)
    plotting.scatter_relationship(
        {"letters": letter_accuracy}, {"letters": results["letters"]["r_by_level"]["Full"]},
        "Model validity vs letter-discrimination accuracy (Full, layer: " + config.LAYERS[config.DEFAULT_LAYER] + ")",
        xlabel="Letter-discrimination accuracy",
        reference_lines={config.LETTER_ACCURACY_THRESHOLD: "exclusion threshold"},
        filename="individual_differences/letters_r_vs_letter_accuracy.png",
    )


def run():
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
        embs = compute_all_embeddings(pair_ids)
        dist_vectors = distances.build_dist_vectors(embs, pair_ids)
        distances.save(dist_vectors, config.DATA_DIR / "dist_vectors.pkl")
        pbar.update(1)

        pbar.set_description(STAGES[3])
        results = {}
        conditions = progress.bar(["letters", "no_letters"], total=2, desc="Conditions", depth=progress.STAGE)
        for condition in conditions:
            results[condition] = _run_condition(condition, wide, dist_vectors, block_assignment, df, excluded[condition])
        _run_cross_condition_figures(results, df)
        pbar.update(1)

    with open(config.DATA_DIR / "analysis_results.pkl", "wb") as f:
        pickle.dump(results, f)

    progress.write(f"Done. Figures in {config.DIAGRAMS_DIR} | data in {config.DATA_DIR}")


if __name__ == "__main__":
    run()
