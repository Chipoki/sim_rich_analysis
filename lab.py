"""
The lab experiment's similarity ratings, brought into the pipeline (run type:
`python run.py --lab`, orchestrated by lab_run.py).

The lab experiment shows the web pilot's 200 fixed pairs, so a lab
participant's ratings are one more vector over the pilot's own pair_ids and
everything analysis.py does to a pilot group applies unchanged. This module
adds what lab data needs before and beyond that:

  - loading: run folders (one per participant, in batch folders under
    config.LAB_RUNS_DIR) -> ratings (pair_id x participant), a per-trial
    table and a per-participant table. The lab randomizes which of a pair's
    two images comes first, so pairs are matched unordered.
  - the model on the lab's own pixels (config.LAB_LEVEL_DIRS), through
    embeddings.py / distances.py, cached apart from the web set's.
  - agreement statistics that do not depend on group size -- the lab pool
    grows batch by batch, so R_LOO alone cannot be compared across runs --
    and the two formulas that carry them to any group size.
  - relations: a fixed set of per-trial and within-condition checks of what
    goes with a rating and with its response time.

The web pilot is the yardstick throughout, read from the caches `python
run.py` leaves in config.DATA_DIR.
"""

import json
import pickle
from itertools import combinations
from math import comb
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

import config
import distances
import embeddings
import sim_vectors

PILOT_GROUPS = ("letters", "no_letters")
PHASES = ("image1", "mask1", "image2", "mask2")
PHASE_NAMES = ("image 1", "mask 1", "image 2", "mask 2")
SPEEDS = ("fast", "middle", "slow")

# What a rating, or its response time, is set against trial by trial (trial_relations).
RELATION_X = ["trial number", "pilot consensus", "pilot dispute", "colour changes", "dot response time", "missed a change"]
RELATION_X_RT_ONLY = ["own rating", "rating extremity"]
# Trial subsets within which agreement is recomputed (conditional_agreement), with their levels in order.
CONDITIONS = {
    "block": None,  # 1..n, read from the data
    "colour changes": None,
    "dot task": ("caught all", "missed one"),
    "last change in": PHASE_NAMES,
    "rating speed": SPEEDS,
}
MIN_CELL_TRIALS = 10
# Two subsets of a participant's trials whose agreement is tested against the rest's (subset_contrasts). Both were
# picked on the first four participants (2026-11-22), so from the second batch on they are tests fixed in advance.
SUBSET_CONTRASTS = {
    "no colour change vs some": lambda t: t["changes"] == 0,
    "fastest third vs the rest": lambda t: t["rt"] <= t.groupby("participant")["rt"].transform(lambda rt: rt.quantile(1 / 3)),
}


# ----------------------------------------------------------------------------- loading

def find_runs(batches: list[str] | None = None) -> list[Path]:
    """Every completed run folder of the named batches (default: every batch), batch by batch in serial order.

    A test run's folder carries a suffix after `_complete` (`_short-run`, `_mock-et`, `_photodiode`) and is left out.
    """
    root = config.LAB_RUNS_DIR
    names = batches or sorted(d.name for d in root.iterdir() if d.is_dir())
    runs = []
    for name in names:
        found = sorted((root / name).glob("P*_complete"), key=lambda d: int(participant_of(d)[1:]))
        if not found:
            raise FileNotFoundError(f"no completed run under {root / name}")
        runs += found
    ids = [participant_of(d) for d in runs]
    repeated = sorted({p for p in ids if ids.count(p) > 1})
    if repeated:
        raise ValueError(f"participant ids in more than one run: {repeated}")
    return runs


def participant_of(run_dir: Path) -> str:
    return run_dir.name.split("_")[0]


def local_to_web() -> dict:
    """The lab's image file name -> the web pilot's image id."""
    web_to_local = json.loads(config.LAB_ID_MAP.read_text())
    out = {local: web for web, local in web_to_local.items()}
    if len(out) != len(web_to_local):
        raise ValueError("two web ids map to one lab file")
    return out


def canonical_pairs(pair_index) -> dict:
    """Unordered image pair -> the pilot's pair_id."""
    return {frozenset(p.split("_")): p for p in pair_index}


def _trial_row(t: dict) -> dict:
    cc = t["stim_timings"].get("color_change_task") or {}
    times = cc.get("change_times", [])
    hit_rts = [x for x in cc.get("hit_rts", []) if x is not None]
    last = None
    if times:  # which phase of the sequence the trial's last colour change fell in
        ends = np.cumsum([t["stim_timings"][p]["nominal"] for p in PHASES])
        last = PHASE_NAMES[min(int(np.searchsorted(ends, times[-1], side="right")), len(PHASES) - 1)]
    return dict(
        rating=t["similarity_rating"], rt=t["reaction_time"], changes=len(times), hits=cc.get("hits", 0),
        false_alarms=cc.get("false_alarms", 0), dot_rt=float(np.mean(hit_rts)) if hit_rts else np.nan, last_change=last,
    )


def load_runs(run_dirs: list[Path], pair_index) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ratings, trials) for the given run folders.

    ratings: the pilot's pair_ids (in `pair_index` order) x participants (P<serial>), each rating as given.
    trials: one row per main trial -- participant, batch, pair_id, order (0-based), block, rating, rt (s) and
      the colour-change task on that trial: changes, hits, false_alarms, dot_rt (mean hit RT, s), last_change
      (the phase its last change fell in; missing on a trial without one).
    """
    canon, to_web, rows = canonical_pairs(pair_index), local_to_web(), []
    for run in run_dirs:
        (path,) = run.glob("*_trials_#*.json")
        main = [t for t in json.loads(path.read_text()) if t["trial_type"] == "experimental"]
        for order, t in enumerate(main):
            ids = frozenset(to_web[PureWindowsPath(t[k]).name] for k in ("image1_path", "image2_path"))
            rows.append(dict(participant=participant_of(run), batch=run.parent.name, pair_id=canon[ids], order=order,
                             block=order // config.LAB_TRIALS_PER_BLOCK, **_trial_row(t)))
    trials = pd.DataFrame(rows)
    ratings = trials.pivot(index="pair_id", columns="participant", values="rating")  # raises on a repeated pair
    ratings = ratings.loc[list(pair_index), [participant_of(d) for d in run_dirs]]
    lo, hi = config.RATING_SCALE
    if ratings.isna().any().any() or not ratings.isin(range(lo, hi + 1)).all().all():
        raise ValueError(f"every participant needs one {lo}-{hi} rating for each of the {len(pair_index)} fixed pairs")
    return ratings.astype(float), trials


def _participant_fields(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # `notes`, the last field, is filled in by hand: an unescaped quote there must not cost the fields before it.
        return json.loads(text.split('"notes"')[0].rstrip().rstrip(",") + "}")


def load_participants(run_dirs: list[Path]) -> pd.DataFrame:
    """One row per participant: batch, the dialog's fields, and the experimenter's labels where a
    config.LAB_PARTICIPANT_LABELS file gives any. The free-text `notes` (which may name the participant) are never read in.
    """
    rows = []
    for run in run_dirs:
        (path,) = run.glob("*_participant_#*.json")
        fields = _participant_fields(path)
        rows.append(dict(participant=participant_of(run), batch=run.parent.name,
                         **{k: fields.get(k) for k in ("age", "gender", "dominant_hand", "dominant_eye")}))
    table = pd.DataFrame(rows).set_index("participant")
    if config.LAB_PARTICIPANT_LABELS.exists():
        table = table.join(pd.read_csv(config.LAB_PARTICIPANT_LABELS, dtype=str).set_index("participant"))
    return table


def label_columns() -> list[str]:
    """The experimenter's label columns, if there is a labels file."""
    if not config.LAB_PARTICIPANT_LABELS.exists():
        return []
    return [c for c in pd.read_csv(config.LAB_PARTICIPANT_LABELS, nrows=0).columns if c != "participant"]


def load_pilot() -> dict:
    """The web pilot as `python run.py` cached it: each group's included raters and the web images' model vectors."""
    try:
        wide = sim_vectors.load(config.DATA_DIR / "sim_vectors.pkl")
        model = distances.load(config.DATA_DIR / "dist_vectors.pkl")
        with open(config.DATA_DIR / "analysis_results.pkl", "rb") as f:
            results = pickle.load(f)
    except FileNotFoundError as error:
        raise FileNotFoundError("the lab run type is scored against the web pilot: run `python run.py` once first") from error
    included = {g: list(results[config.BACKEND][g]["loo"].index) for g in PILOT_GROUPS}
    return dict(pair_index=list(wide.index), ratings={g: wide[g][included[g]].astype(float) for g in PILOT_GROUPS}, model=model)


def as_wide(group: pd.DataFrame, condition: str) -> pd.DataFrame:
    """analysis.py's wide layout (columns = (condition, participant)) for one group."""
    return pd.concat({condition: group}, axis=1, names=["condition", "participant_id"])


# ----------------------------------------------------------------------------- the model

def model_vectors(pair_index, extractor: embeddings.EmbeddingExtractor | None = None) -> pd.DataFrame:
    """distances.py's table for the lab's own pixels: pair_id x (level, layer, metric), cached in config.LAB_DATA_DIR.

    The cached table is rebuilt when it lacks a configured level or layer (a level added to config.LAB_LEVEL_DIRS),
    since nothing else would notice.
    """
    cache = config.LAB_DATA_DIR / "dist_vectors.pkl"
    layers = list(config.LAYERS)
    wanted = {(level, layer) for level in config.LAB_LEVEL_DIRS for layer in layers}
    if cache.exists():
        vectors = distances.load(cache)
        if wanted <= {(level, layer) for level, layer, _ in vectors.columns} and list(vectors.index) == sorted(pair_index):
            return vectors
    extractor = extractor or embeddings.get_extractor("resnet50_places365")
    web_to_local = json.loads(config.LAB_ID_MAP.read_text())
    embs = {}
    for level, folder in config.LAB_LEVEL_DIRS.items():
        images = {web: folder / local for web, local in web_to_local.items()}
        missing = [str(p) for p in images.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{level}: {len(missing)} images missing, e.g. {missing[0]}")
        by_layer = extractor.extract_level(level, folder, layers, cache_root=config.LAB_DATA_DIR / "embeddings", images=images)
        for layer, by_image in by_layer.items():
            embs[(level, layer)] = by_image
    vectors = distances.build_dist_vectors(embs, list(pair_index))
    distances.save(vectors, cache)
    return vectors


def pair_references(pilot: dict, vectors: pd.DataFrame, layer: str = config.DEFAULT_LAYER) -> pd.DataFrame:
    """Per pair, what a rating can be set against: the No-Letters pilot's mean rating (`consensus`), how much its
    raters disagreed on the pair (`dispute`, their SD), and the model's similarity at each lab level (`model <level>`)."""
    no_letters = pilot["ratings"]["no_letters"]
    refs = pd.DataFrame({"consensus": no_letters.mean(axis=1), "dispute": no_letters.std(axis=1)})
    for level in config.LAB_LEVEL_DIRS:
        refs[f"model {level}"] = distances.get_similarity(vectors, level, layer)
    return refs


# ----------------------------------------------------------------------------- agreement

def mean_pairwise_r(group: pd.DataFrame) -> float:
    """Mean Pearson r over every two raters: one rater against one other, whatever the group's size."""
    c = group.corr().to_numpy()
    return float(c[np.triu_indices_from(c, k=1)].mean())


def subgroup_agreement(group: pd.DataFrame, k: int, max_groups: int | None = None, seed: int = config.LAB_BOOTSTRAP_SEED) -> np.ndarray:
    """mean_pairwise_r of the k-rater subgroups of `group`: what a group of k shows by chance of who came.

    Every subgroup while there are at most `max_groups` (default config.LAB_MAX_SUBGROUPS), else that many drawn at random.
    """
    c = group.corr().to_numpy()
    n, upper = c.shape[0], np.triu_indices(k, k=1)
    max_groups = config.LAB_MAX_SUBGROUPS if max_groups is None else max_groups
    if comb(n, k) <= max_groups:
        members = combinations(range(n), k)
    else:
        rng = np.random.default_rng(seed)
        members = (rng.choice(n, k, replace=False) for _ in range(max_groups))
    return np.array([c[np.ix_(m, m)][upper].mean() for m in members])


def spearman_brown(r: float, k) -> np.ndarray:
    """Reliability of the mean of k raters whose single-rater agreement is r."""
    k = np.asarray(k, float)
    return k * r / (1 + (k - 1) * r)


def expected_loo(r: float, n) -> np.ndarray:
    """Expected leave-one-out r (one rater against the mean of the other n-1) at single-rater agreement r."""
    k = np.asarray(n, float) - 1
    return r * np.sqrt(k / (1 + (k - 1) * r))


def per_rater_r(group: pd.DataFrame, reference: pd.Series) -> pd.Series:
    """Pearson r of each rater's vector with one reference vector (a model's, or another group's mean)."""
    return group.apply(lambda col: np.corrcoef(col, reference.loc[group.index])[0, 1])


def bootstrap_pairs(stat, n_pairs: int, n_boot: int = config.LAB_BOOTSTRAP_N, seed: int = config.LAB_BOOTSTRAP_SEED) -> np.ndarray:
    """95% interval of stat(row positions) over resampled image pairs, the raters held fixed.

    Returns [low, high]; for a vector-valued stat, one column per component.
    """
    rng = np.random.default_rng(seed)
    draws = np.array([stat(rng.integers(0, n_pairs, n_pairs)) for _ in range(n_boot)])
    return np.percentile(draws, [2.5, 97.5], axis=0)


# ----------------------------------------------------------------------------- relations

def _with_references(trials: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    t = trials.join(refs, on="pair_id")
    lo, hi = config.RATING_SCALE
    t["trial number"], t["own rating"] = t["order"] + 1, t["rating"]
    t["pilot consensus"], t["pilot dispute"] = t["consensus"], t["dispute"]
    t["colour changes"], t["dot response time"] = t["changes"], t["dot_rt"]
    t["rating extremity"] = (t["rating"] - (lo + hi) / 2).abs()
    t["missed a change"] = np.where(t["changes"] > 0, (t["hits"] < t["changes"]).astype(float), np.nan)
    return t


def trial_relations(trials: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho, within each participant, of the rating and of its response time with each per-trial variable.

    One row per (participant, y, x): y in {"rating", "response time"}; x from RELATION_X, the model's similarity at
    each level, and for the response time also the rating itself and its distance from the scale's midpoint.
    rho is NaN where the participant has fewer than MIN_CELL_TRIALS usable trials or no variation (e.g. no missed change).
    """
    t = _with_references(trials, refs)
    xs = RELATION_X[:3] + [c for c in refs.columns if c.startswith("model ")] + RELATION_X[3:]
    rows = []
    for pid, d in t.groupby("participant", sort=False):
        for y, column, more in (("rating", "rating", []), ("response time", "rt", RELATION_X_RT_ONLY)):
            for x in xs + more:
                ok = d[[x, column]].dropna()
                usable = len(ok) >= MIN_CELL_TRIALS and ok[x].nunique() > 1 and ok[column].nunique() > 1
                rho = ok[x].corr(ok[column], method="spearman") if usable else np.nan
                rows.append(dict(participant=pid, y=y, x=x, rho=rho, n=len(ok)))
    return pd.DataFrame(rows)


def conditional_agreement(trials: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """Pearson r of a participant's ratings with each reference, recomputed within subsets of that participant's trials.

    One row per (participant, condition, level, reference): the conditions are CONDITIONS (rating speed is the
    participant's own response times cut in thirds; the dot-task conditions use only trials with a colour change),
    the references are `consensus` and the model at each level. r is NaN below MIN_CELL_TRIALS trials or without variation.
    """
    t = trials.join(refs, on="pair_id")
    t["block"], t["colour changes"], t["last change in"] = t["block"] + 1, t["changes"], t["last_change"]
    t["dot task"] = np.where(t["changes"] > 0, np.where(t["hits"] == t["changes"], "caught all", "missed one"), None)
    t["rating speed"] = t.groupby("participant")["rt"].transform(lambda rt: pd.qcut(rt, len(SPEEDS), labels=list(SPEEDS)).astype(str))
    references = ["consensus"] + [c for c in refs.columns if c.startswith("model ")]
    rows = []
    for pid, d in t.groupby("participant", sort=False):
        for condition, levels in CONDITIONS.items():
            for level in levels or sorted(d[condition].dropna().unique()):
                cell = d[d[condition] == level]
                for reference in references:
                    usable = len(cell) >= MIN_CELL_TRIALS and cell["rating"].nunique() > 1
                    r = np.corrcoef(cell["rating"], cell[reference])[0, 1] if usable else np.nan
                    rows.append(dict(participant=pid, condition=condition, level=level, reference=reference, r=r, n=len(cell)))
    return pd.DataFrame(rows)


def _rowwise_r(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


def subset_contrasts(trials: pd.DataFrame, refs: pd.DataFrame, n_perm: int | None = None, seed: int = config.LAB_BOOTSTRAP_SEED) -> pd.DataFrame:
    """For each of SUBSET_CONTRASTS and each reference: r of the ratings with the reference inside the subset minus
    r outside it, per participant (`lowest`..`highest`) and averaged (`difference`).

    `p` is two-sided, by shuffling within each participant which trials count as the subset: the share of shuffles
    whose mean difference is at least as far from zero. One row per (contrast, reference); p is not corrected for
    the number of rows.
    """
    n_perm = config.LAB_PERMUTATIONS if n_perm is None else n_perm
    t = trials.join(refs, on="pair_id")
    references = ["consensus"] + [c for c in refs.columns if c.startswith("model ")]
    rng = np.random.default_rng(seed)
    rows = []
    for name, in_subset in SUBSET_CONTRASTS.items():
        mask = in_subset(t)
        for reference in references:
            observed, shuffled = [], []
            for _, d in t.groupby("participant", sort=False):
                x, y = d["rating"].to_numpy(float), d[reference].to_numpy(float)
                k = int(mask.loc[d.index].sum())
                # row 0: the subset first, as it is; the other rows: the participant's trials in shuffled order
                order = np.vstack([np.argsort(~mask.loc[d.index].to_numpy(), kind="stable"),
                                   rng.permuted(np.tile(np.arange(len(d)), (n_perm, 1)), axis=1)])
                diff = _rowwise_r(x[order[:, :k]], y[order[:, :k]]) - _rowwise_r(x[order[:, k:]], y[order[:, k:]])
                observed.append(diff[0]), shuffled.append(diff[1:])
            means = np.nanmean(shuffled, axis=0)
            rows.append(dict(contrast=name, reference=reference, difference=np.mean(observed), lowest=np.min(observed),
                             highest=np.max(observed), p=float((np.abs(means) >= abs(np.mean(observed))).mean())))
    return pd.DataFrame(rows)


def response_times(trials: pd.DataFrame) -> pd.DataFrame:
    """Median response time (s) per participant within each rating, block, and phase of the last colour change
    ("none": a trial without one). One row per (participant, by, level)."""
    t = trials.assign(by_rating=trials["rating"].astype(int), by_block=trials["block"] + 1,
                      by_change=trials["last_change"].fillna("none"))
    rows = []
    for by, column in (("own rating", "by_rating"), ("block", "by_block"), ("last change in", "by_change")):
        med = t.groupby(["participant", column], sort=False)["rt"].agg(["median", "size"])
        rows += [dict(participant=p, by=by, level=level, median=m, n=n) for (p, level), (m, n) in zip(med.index, med.to_numpy())]
    return pd.DataFrame(rows)


def participant_table(trials: pd.DataFrame, participants: pd.DataFrame, columns: dict[str, pd.Series]) -> pd.DataFrame:
    """One row per participant: how they rated, how they did on the dot task, the given per-participant
    `columns` (agreement and model fits), then the dialog's fields and the experimenter's labels."""
    g = trials.groupby("participant", sort=False)
    table = pd.DataFrame({
        "trials": g.size(),
        "mean rating": g["rating"].mean(),
        "rating sd": g["rating"].std(),
        "rated lowest": g["rating"].apply(lambda r: (r == config.RATING_SCALE[0]).mean()),
        "median rating rt": g["rt"].median(),
        "changes caught": g["hits"].sum() / g["changes"].sum(),
        "stray presses": g["false_alarms"].sum(),
        "median dot rt": g["dot_rt"].median(),
    })
    return table.join(pd.DataFrame(columns)).join(participants)
