"""
Correlational analyses: per-participant Pearson R against model vectors,
group-average (AVG-vector) correlation, split-half reliability, leave-one-out
(LOO) human-human reliability / noise ceiling, and a light paired-test note.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon

import config
import parsing
import progress


def get_excluded_participants(df: pd.DataFrame) -> dict[str, set]:
    """
    Participant ids to drop per condition (PDF §exclusions, N=19 each).
    Letters: computed from letter accuracy < threshold (matches the PDF's
    N=19 exactly against the real data). No-Letters: manual, since it's a
    self-report not present in the JSON -- see config.py.
    """
    acc = parsing.letter_accuracy_by_participant(df)
    letters_excluded = set(acc[acc < config.LETTER_ACCURACY_THRESHOLD].index)

    no_letters_excluded = set()
    if config.NO_LETTERS_MANUAL_EXCLUDE_PARTICIPANT_ID:
        no_letters_excluded.add(config.NO_LETTERS_MANUAL_EXCLUDE_PARTICIPANT_ID)

    return {"letters": letters_excluded, "no_letters": no_letters_excluded}


def _included_participants(wide: pd.DataFrame, condition: str, excluded: set) -> list:
    all_pids = wide[condition].columns
    return [p for p in all_pids if p not in excluded]


def per_participant_correlations(
    wide: pd.DataFrame, condition: str, model_similarity: pd.Series, excluded: set = frozenset()
) -> pd.Series:
    """One Pearson R per included participant, human vector vs model vector."""
    pids = _included_participants(wide, condition, excluded)
    model_similarity = model_similarity.reindex(wide.index)
    out = {}
    for pid in progress.bar(pids, desc=f"{condition} participants", depth=progress.STEP):
        human = wide[(condition, pid)]
        r, _p = pearsonr(human, model_similarity)
        out[pid] = r
    return pd.Series(out, name="pearson_r")


def best_layer_per_participant(
    wide: pd.DataFrame, condition: str, similarity_by_layer: dict[str, pd.Series], excluded: set = frozenset()
) -> pd.Series:
    """
    Per level (caller passes one level's similarity_by_layer = {layer: series}),
    for each participant pick whichever of the examined layers correlates best.
    """
    per_layer = {
        layer: per_participant_correlations(wide, condition, sim, excluded)
        for layer, sim in similarity_by_layer.items()
    }
    table = pd.DataFrame(per_layer)
    return table.max(axis=1)


def avg_vector_correlation(
    wide: pd.DataFrame, condition: str, model_similarity: pd.Series, excluded: set = frozenset()
) -> float:
    """Single R: group-averaged human vector vs model vector (PDF's AVG-vector)."""
    pids = _included_participants(wide, condition, excluded)
    group_avg = wide[condition][pids].mean(axis=1)
    model_similarity = model_similarity.reindex(wide.index)
    r, _p = pearsonr(group_avg, model_similarity)
    return r


def split_half_reliability(
    wide: pd.DataFrame, condition: str, excluded: set = frozenset(),
    n_iter: int = config.SPLIT_HALF_N_ITER, seed: int = config.SPLIT_HALF_SEED,
) -> float:
    """
    Split-half reliability of human ratings: repeatedly split participants
    into two random halves, correlate the two halves' group-average
    vectors, Spearman-Brown correct, and average over iterations.
    """
    pids = np.array(_included_participants(wide, condition, excluded))
    rng = np.random.default_rng(seed)
    n = len(pids)
    half = n // 2
    rs = []
    for _ in progress.bar(range(n_iter), total=n_iter, desc=f"{condition} split-half iterations", depth=progress.STEP):
        perm = rng.permutation(pids)
        group_a, group_b = perm[:half], perm[half : 2 * half]
        vec_a = wide[condition][group_a].mean(axis=1)
        vec_b = wide[condition][group_b].mean(axis=1)
        r, _p = pearsonr(vec_a, vec_b)
        rs.append(r)
    mean_r = float(np.mean(rs))
    # Spearman-Brown correction for doubling from half to full length
    return (2 * mean_r) / (1 + mean_r)


def leave_one_out_reliability(wide: pd.DataFrame, condition: str, excluded: set = frozenset()) -> pd.Series:
    """
    For each included participant, correlate their vector against the
    mean of everyone else's ("rest of group"). This is both the
    human-human noise ceiling and the PDF's headline R_LOO (mean of this
    series).
    """
    pids = _included_participants(wide, condition, excluded)
    group = wide[condition][pids]
    out = {}
    for pid in progress.bar(pids, desc=f"{condition} LOO participants", depth=progress.STEP):
        rest_mean = group.drop(columns=pid).mean(axis=1)
        r, _p = pearsonr(group[pid], rest_mean)
        out[pid] = r
    return pd.Series(out, name="pearson_r")


def paired_level_test(r_by_level: dict[str, pd.Series], level_a: str, level_b: str) -> dict:
    """
    A short stats note: paired Wilcoxon signed-rank test between two
    levels' per-participant R values (same participants -> paired).
    """
    a = r_by_level[level_a]
    b = r_by_level[level_b]
    common = a.index.intersection(b.index)
    diffs = a.loc[common] - b.loc[common]
    if np.allclose(diffs, 0):
        return {"level_a": level_a, "level_b": level_b, "n": len(common), "statistic": float("nan"), "p_value": float("nan")}
    stat, p = wilcoxon(a.loc[common], b.loc[common])
    return {"level_a": level_a, "level_b": level_b, "n": len(common), "statistic": stat, "p_value": p}
