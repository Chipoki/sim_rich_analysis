"""
The lab experiment's similarity ratings, brought into the pipeline (run type:
`python run.py --lab`, orchestrated by lab_run.py).

The lab experiment shows the web pilot's 200 fixed pairs, so a lab
participant's ratings are one more vector over the pilot's own pair_ids and
everything analysis.py does to a pilot group applies unchanged. This module
adds what lab data needs before and beyond that:

  - loading: run folders (one per participant, in batch folders under
    config.LAB_RUNS_DIR) -> ratings (pair_id x participant), a per-trial
    table and a per-participant table, plus each run's practice scores and
    its trial 201. The lab randomizes which of a pair's two images comes
    first, so pairs are matched unordered. A participant is S<k>, counted
    over every batch (participant_ids), not the lab PC's P<serial>.
  - the model on the lab's own pixels (config.LAB_LEVEL_DIRS), through
    embeddings.py / distances.py, cached apart from the web set's.
  - agreement statistics that do not depend on group size -- the lab pool
    grows batch by batch, so R_LOO alone cannot be compared across runs --
    and the formulas that carry them to any group size.
  - the noise ceiling, lower (R_LOO) and upper (noise_ceiling), and the
    pilot's at the lab group's own size (subgroup_ceilings).
  - relations: a fixed set of per-trial and within-condition checks of what
    goes with a rating and with its response time.
  - groups: participants whose practice was gated (scored, and repeated when
    it fell short) against the rest, measure by measure, with exact p-values.
  - the second task: which overlay task a run had (run_task) decides what is
    asked of it (TASKS): the colour-change dot, or the letters with their
    count question. Runs of different tasks are never pooled (find_runs).
  - biases: whether the dot task moves the rating itself (rating_shifts,
    with the pilot's letter effect as yardstick), whether a rating leans on
    the one before it, and whether the order of a pair's two images matters.

The web pilot is the yardstick throughout, read from the caches `python
run.py` leaves in config.DATA_DIR.
"""

import json
import pickle
import re
from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path, PureWindowsPath
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

import config
import distances
import embeddings
import progress
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
# Two kinds of a participant's trials whose ratings themselves are compared (rating_shifts): name -> a function giving
# (the trials compared, which of them are the first-named kind). They ask of the dot what the pilot's letters answered
# badly: there, two images under the same letter were rated more alike. Fixed on the first two batches (2026-11-24),
# so tests fixed in advance from the third on.
RATING_SHIFTS = {
    "no colour change vs some": lambda t: (t["changes"] >= 0, t["changes"] == 0),
    "no colour change vs one": lambda t: (t["changes"] <= 1, t["changes"] == 0),
    "no colour change vs two": lambda t: (t["changes"].isin([0, 2]), t["changes"] == 0),
    "two colour changes vs one": lambda t: (t["changes"].isin([1, 2]), t["changes"] == 2),
    "dot the same colour on both images vs not": lambda t: (t["changes"] >= 1, t["dot_same"].eq(True)),
    "a change while an image was on vs only in a mask": lambda t: (t["changes"] >= 1, t["change_in_image"].eq(True)),
    "a change missed vs all caught": lambda t: (t["changes"] >= 1, t["hits"] < t["changes"]),
    "a stray press vs none": lambda t: (t["changes"] >= 0, t["false_alarms"] > 0),
    "image order as in the pilot vs reversed": lambda t: (t["changes"] >= 0, t["pilot_order"].eq(True)),
}
# Pairs of per-participant measures related across participants (participant_relations).
PARTICIPANT_RELATIONS = [
    ("practice attention (first)", "clean dot trials"), ("practice rating (first)", "r with consensus"),
    ("practice rating (first)", "r with the others"), ("clean dot trials", "r with consensus"),
    ("median rating rt", "r with consensus"), ("rated lowest", "r with consensus"),
    ("r with colour similarity", "r with consensus"),
]

# ----------------------------------------------------------------------------- the second task

DOT, LETTER_COUNT = "colour-change", "letters-count"


def _dot_columns(t: pd.DataFrame) -> None:
    t["colour changes"], t["dot response time"], t["last change in"] = t["changes"], t["dot_rt"], t["last_change"]
    t["missed a change"] = np.where(t["changes"] > 0, (t["hits"] < t["changes"]).astype(float), np.nan)
    t["dot task"] = np.where(t["changes"] > 0, np.where(t["hits"] == t["changes"], "caught all", "missed one"), None)


def _letter_columns(t: pd.DataFrame) -> None:
    t["target count"], t["letter response time"] = t["count_true"], t["letter_rt"]
    t["wrong count"] = (~t["clean"].astype(bool)).astype(float)
    t["letters"] = np.where(t["letters_match"].astype(bool), "same letter", "different letters")
    t["letter task"] = np.where(t["clean"].astype(bool), "count right", "count wrong")


@dataclass(frozen=True)
class Task:
    """What the analysis asks of one overlay task. The tables are the module's own for the colour-change task
    (RELATION_X, CONDITIONS, SUBSET_CONTRASTS, RATING_SHIFTS, PARTICIPANT_RELATIONS) and their counterparts for
    another; the rest says where that task's numbers sit in a trials table."""
    name: str
    noun: str                     # in a title or a sentence: "dot task", "letter task"
    relation_x: list
    conditions: dict
    subset_contrasts: dict
    rating_shifts: dict
    participant_relations: list
    add_columns: Callable         # adds to a trials table the named columns relation_x and conditions use
    participant_columns: Callable  # trials grouped by participant -> {measure: one value per participant}
    correct: str                  # of those measures, the share of trials with the task answered right
    load: str                     # the trials column of what the task showed on the trial (rating_by_load)
    load_names: dict              # its values, named
    unshifted: Callable           # trials -> the trials free of what rating_shifts' first contrast singles out
    response_time_by: tuple       # response_times' third breakdown: (its name, the trials column, the name of a missing value)


TASKS = {
    DOT: Task(
        name=DOT, noun="dot task", relation_x=RELATION_X, conditions=CONDITIONS, subset_contrasts=SUBSET_CONTRASTS,
        rating_shifts=RATING_SHIFTS, participant_relations=PARTICIPANT_RELATIONS, add_columns=_dot_columns,
        participant_columns=lambda g: {
            "changes caught": g["hits"].sum() / g["changes"].sum(), "stray presses": g["false_alarms"].sum(),
            "clean dot trials": g["clean"].mean(), "median dot rt": g["dot_rt"].median()},
        correct="clean dot trials", load="changes", load_names={0: "no change", 1: "one change", 2: "two changes"},
        unshifted=lambda t: t["changes"] > 0, response_time_by=("last change in", "last_change", "none")),
    # The letters, asked how many of a pair's two were the block's target letter (prepared 2026-11-25; no run yet).
    # Its first rating shift is the pilot's own, which the count question is there to remove: the same letter on
    # both images against different letters (LETTER_SHIFT is its yardstick). "On both vs on neither" then compares
    # the two kinds of same-letter trial, which differ only in the answer, 2 or 0.
    LETTER_COUNT: Task(
        name=LETTER_COUNT, noun="letter task",
        relation_x=["trial number", "pilot consensus", "pilot dispute", "target count", "letter response time", "wrong count"],
        conditions={"block": None, "target count": None, "letter task": ("count right", "count wrong"),
                    "letters": ("same letter", "different letters"), "rating speed": SPEEDS},
        subset_contrasts={"same letters vs different": lambda t: t["letters_match"].astype(bool),
                          "fastest third vs the rest": SUBSET_CONTRASTS["fastest third vs the rest"]},
        rating_shifts={
            "the same letter on both images vs different": lambda t: (t["rating"].notna(), t["letters_match"].astype(bool)),
            "the target on neither image vs on some": lambda t: (t["rating"].notna(), t["count_true"] == 0),
            "the target on both images vs on one": lambda t: (t["count_true"] >= 1, t["count_true"] == 2),
            "the target on both images vs on neither": lambda t: (t["count_true"].isin([0, 2]), t["count_true"] == 2),
            "answered 2 vs answered 0": lambda t: (t["count_response"].isin([0, 2]), t["count_response"] == 2),
            "a wrong count vs a right one": lambda t: (t["rating"].notna(), ~t["clean"].astype(bool)),
            "counting T vs counting L": lambda t: (t["rating"].notna(), t["count_target"] == "T"),
            "image order as in the pilot vs reversed": lambda t: (t["rating"].notna(), t["pilot_order"].eq(True)),
        },
        participant_relations=[(x.replace("clean dot trials", "letter answers right"), y.replace("clean dot trials", "letter answers right"))
                               for x, y in PARTICIPANT_RELATIONS],
        add_columns=_letter_columns,
        participant_columns=lambda g: {"letter answers right": g["clean"].mean(), "median letter rt": g["letter_rt"].median()},
        correct="letter answers right", load="count_true",
        load_names={0: "target on neither", 1: "target on one", 2: "target on both"},
        unshifted=lambda t: ~t["letters_match"].astype(bool), response_time_by=("target count", "count_true", "none")),
}


def task_of(trials: pd.DataFrame) -> Task:
    """The one overlay task a trials table is of (its `task` column; the colour-change task without the column)."""
    names = list(trials["task"].unique()) if "task" in trials else [DOT]
    if len(names) != 1:
        raise ValueError(f"trials of more than one task ({', '.join(sorted(names))}): they are not analysed together")
    return TASKS[names[0]]


# ----------------------------------------------------------------------------- loading

def run_task(run_dir: Path) -> str:
    """The overlay task a run had, read from its first main trial's record: DOT, LETTER_COUNT, or the name of one
    this run type does not analyse ("letters-same", "center-dot-task", ..., "no task")."""
    (path,) = run_dir.glob("*_trials_#*.json")
    record = next(t for t in json.loads(path.read_text()) if t["trial_type"] == "experimental")["stim_timings"]
    if "letter_task" in record:
        return LETTER_COUNT if record["letter_task"].get("question") == "count" else "letters-same"
    if "color_change_task" in record:
        return DOT
    return next((k.replace("_", "-") for k in record if k.endswith("_task")), "no task")


def run_tasks(batches: list[str] | None = None) -> dict[Path, str]:
    """Every completed run folder of the named batches (default: every batch) -> its task (run_task), in the order
    they were run. A test run's folder carries a suffix after `_complete` (`_short-run`, `_mock-et`, `_photodiode`)
    and is left out."""
    root = config.LAB_RUNS_DIR
    names = batches or sorted(d.name for d in root.iterdir() if d.is_dir())
    runs = []
    for name in names:
        found = list((root / name).glob("P*_complete"))
        if not found:
            raise FileNotFoundError(f"no completed run under {root / name}")
        runs += found
    return {run: run_task(run) for run in sorted(runs, key=_started)}


def find_runs(batches: list[str] | None = None, task: str | None = None) -> list[Path]:
    """The completed run folders of the named batches (default: every batch) that had `task`, in the order they
    were run. Participants who did different second tasks are not one group, so without `task` the runs must all
    share one, and a selection that mixes tasks is refused with the tasks it holds."""
    tasks = run_tasks(batches)
    found = sorted(set(tasks.values()))
    if task is None:
        if len(found) > 1:
            counts = ", ".join(f"{name} ({sum(t == name for t in tasks.values())})" for name in found)
            raise ValueError(f"the selected runs had different tasks: {counts}. Name one: find_runs(task=...), "
                             f"`python run.py --lab --task <name>`")
        task = found[0]
    if task not in TASKS:
        raise ValueError(f"the lab run type analyses {' and '.join(TASKS)}, not {task!r}")
    runs = [run for run, name in tasks.items() if name == task]
    if not runs:
        raise FileNotFoundError(f"no completed run of the {task} task among {', '.join(found)}")
    return runs


def _started(run_dir: Path) -> str:
    """When a run began, `<date>_<HHMM>`, from its folder's name (P<serial>_<date>_<HHMM>_to_<HHMM>_complete)."""
    return "_".join(run_dir.name.split("_")[1:3])


def serial_of(run_dir: Path) -> str:
    """The lab PC's own id for the run, `P<serial>`: unique within a batch folder only."""
    return run_dir.name.split("_")[0]


def participant_ids() -> dict[Path, str]:
    """Run folder -> participant id `S<k>`: the k-th completed run under config.LAB_RUNS_DIR, by start time.

    The lab PC's serial starts over whenever its history file leaves with a batch (two batches both hold a P1), so
    it cannot name a participant across batches. The number is counted over every batch, so a participant keeps
    it whichever batches a run selects.
    """
    runs = sorted(config.LAB_RUNS_DIR.glob("*/P*_complete"), key=_started)
    starts = [_started(d) for d in runs]
    if len(set(starts)) != len(starts):
        raise ValueError("two completed runs began in the same minute: their order, and so their ids, is undefined")
    return {d: f"S{k}" for k, d in enumerate(runs, 1)}


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


def _letter_trial_row(t: dict) -> dict:
    """A main trial under the letters' count question: `clean` is the count answered right."""
    lt = t["stim_timings"]["letter_task"]
    return dict(
        task=LETTER_COUNT, rating=t["similarity_rating"], rt=t["reaction_time"], clean=bool(lt["count_correct"]),
        letter_rt=lt["count_rt"], letters_match=bool(lt["letters_match"]), letter1=lt["letter1"], letter2=lt["letter2"],
        count_target=lt["count_target"], count_true=lt["count_true"], count_response=lt["count_response"],
    )


def _trial_row(t: dict) -> dict:
    if "letter_task" in t["stim_timings"]:
        return _letter_trial_row(t)
    cc = t["stim_timings"].get("color_change_task") or {}
    times = cc.get("change_times", [])
    hit_rts = [x for x in cc.get("hit_rts", []) if x is not None]
    colours = cc.get("colors") or []
    ends = np.cumsum([t["stim_timings"][p]["nominal"] for p in PHASES])  # when image 1, mask 1, image 2 and mask 2 end
    last = dot_same = change_in_image = None
    if times:  # which phase of the sequence the trial's last colour change fell in
        last = PHASE_NAMES[min(int(np.searchsorted(ends, times[-1], side="right")), len(PHASES) - 1)]
        change_in_image = any(x < ends[0] or ends[1] <= x < ends[2] for x in times)
    if colours:  # the dot's colour as each image came on: colours[k] shows from the k-th change
        dot_same = colours[0] == colours[sum(x <= ends[1] for x in times)]
    hits, false_alarms = cc.get("hits", 0), cc.get("false_alarms", 0)
    return dict(
        task=DOT, rating=t["similarity_rating"], rt=t["reaction_time"], changes=len(times), hits=hits, false_alarms=false_alarms,
        clean=hits == len(times) and false_alarms == 0,  # practice's attention criterion (practice_scores)
        dot_rt=float(np.mean(hit_rts)) if hit_rts else np.nan, last_change=last,
        dot_same=dot_same, change_in_image=change_in_image, dot_start=colours[0] if colours else None,
    )


def load_runs(run_dirs: list[Path], pair_index) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ratings, trials) for the given run folders.

    ratings: the pilot's pair_ids (in `pair_index` order) x participants (participant_ids), each rating as given.
    trials: one row per main trial -- participant, batch, pair_id, order (0-based), block, rating, rt (s) and
      the colour-change task on that trial: changes, hits, false_alarms, clean (every change answered and no other
      press), dot_rt (mean hit RT, s), last_change (the phase its last change fell in; missing on a trial without one),
      dot_same (the dot had one colour as image 1 and as image 2 came on), change_in_image (a change came while an image,
      not a mask, was on; missing without a change), dot_start (its first colour) and pilot_order (the two images came in
      the order the pilot showed them; the lab draws the order per participant). Under the letters' count question
      the task's columns are instead: clean (the count answered right), letter_rt (s), letters_match, letter1,
      letter2, count_target (the letter counted), count_true and count_response. `task` names which (task_of).
    """
    canon, to_web, pid, rows = canonical_pairs(pair_index), local_to_web(), participant_ids(), []
    for run in run_dirs:
        (path,) = run.glob("*_trials_#*.json")
        main = [t for t in json.loads(path.read_text()) if t["trial_type"] == "experimental"]
        for order, t in enumerate(main):
            ids = frozenset(to_web[PureWindowsPath(t[k]).name] for k in ("image1_path", "image2_path"))
            first = to_web[PureWindowsPath(t["image1_path"]).name]
            rows.append(dict(participant=pid[run], batch=run.parent.name, pair_id=canon[ids], order=order,
                             block=order // config.LAB_TRIALS_PER_BLOCK, pilot_order=canon[ids].split("_")[0] == first,
                             **_trial_row(t)))
    trials = pd.DataFrame(rows)
    task_of(trials)  # one task, or a refusal
    ratings = trials.pivot(index="pair_id", columns="participant", values="rating")  # raises on a repeated pair
    ratings = ratings.loc[list(pair_index), [pid[d] for d in run_dirs]]
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


LABEL_KEYS = ["batch", "serial"]  # how the experimenter's labels file names a run: its batch folder and its P<serial>


def load_participants(run_dirs: list[Path]) -> pd.DataFrame:
    """One row per participant: batch, serial, the dialog's fields, and the experimenter's labels where a
    config.LAB_PARTICIPANT_LABELS file gives any. The free-text `notes` (which may name the participant) are never read in.
    """
    pid, rows = participant_ids(), []
    for run in run_dirs:
        (path,) = run.glob("*_participant_#*.json")
        fields = _participant_fields(path)
        rows.append(dict(participant=pid[run], batch=run.parent.name, serial=serial_of(run),
                         **{k: fields.get(k) for k in ("age", "gender", "dominant_hand", "dominant_eye")}))
    table = pd.DataFrame(rows)
    if config.LAB_PARTICIPANT_LABELS.exists():
        table = table.merge(pd.read_csv(config.LAB_PARTICIPANT_LABELS, dtype=str), on=LABEL_KEYS, how="left", validate="one_to_one")
    return table.set_index("participant")


def label_columns() -> list[str]:
    """The experimenter's label columns, if there is a labels file."""
    if not config.LAB_PARTICIPANT_LABELS.exists():
        return []
    return [c for c in pd.read_csv(config.LAB_PARTICIPANT_LABELS, nrows=0).columns if c not in LABEL_KEYS]


def practice_scores(trials: list[dict]) -> tuple[float, float]:
    """The two scores the experiment repeats practice on (its `_practice_scores`), for one iteration's trials.

    attention: the share of trials with every colour change answered and no other press (under the letters' count
      question: with the count right).
    rating: of every (pair named similar, pair named different) comparison, the share the similar pair wins, a tie
      counting half -- 0.5 tells nothing apart, wherever on the scale the ratings lie. NaN without a pair of each kind.
    """
    def clean(t):
        if "letter_task" in t["stim_timings"]:  # the count answered right
            return bool(t["stim_timings"]["letter_task"]["count_correct"])
        cc = t["stim_timings"]["color_change_task"]
        return cc["hits"] == len(cc["change_times"]) and cc["false_alarms"] == 0

    similar, different = ([t["similarity_rating"] for t in trials if f"_{kind}_" in PureWindowsPath(t["image1_path"]).stem]
                          for kind in ("similar", "different"))
    wins = [(s > d) + (s == d) / 2 for s in similar for d in different]
    return float(np.mean([clean(t) for t in trials])), float(np.mean(wins)) if wins else np.nan


def practice_iterations(trials: list[dict]) -> list[int]:
    """Each practice trial's iteration (1-based), for one run's practice trials in the order they ran.

    Since 2026-11-25 an iteration shows the pairs of its own folder (config.LAB_PRACTICE_ITERATION_PREFIX + k), so
    the folder of a trial's images says which it is. Before, every iteration showed the same pairs again, either way
    round, and a trial's iteration is its place among the practice trials over the number of different pairs."""
    prefix = config.LAB_PRACTICE_ITERATION_PREFIX
    folders = [PureWindowsPath(t["image1_path"]).parent.name for t in trials]
    if trials and all(f.startswith(prefix) for f in folders):
        return [int(f[len(prefix):]) for f in folders]
    size = len({frozenset((t["image1_path"], t["image2_path"])) for t in trials})
    if not trials or len(trials) % size:
        raise ValueError(f"{len(trials)} practice trials do not divide into passes over {size} pairs")
    return [k // size + 1 for k in range(len(trials))]


def load_practice(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.Series]:
    """(practice, gated): how each participant's practice went, and which runs' practice was gated.

    practice: one row per (participant, iteration, 1-based) -- attention and rating (practice_scores), trials, and
      `rating_criterion`: whether the experiment held that iteration's rating score to a minimum as well (only the
      runs of 2026-11-23 did). The rating score is NaN for an iteration without the named pairs: since 2026-11-25
      every iteration but the first (practice_iterations).
    gated: per participant, whether the run scored practice and repeated it when it fell short (a run whose
      participant file gives practice_max_iterations 1 scored it and could not repeat it: not gated). Both are read from
      the run's log (its `practice iteration <k>: attention ... (min ...)[, rating ... (min ...)]: ...` lines), since
      a gated run that passes at once also has one iteration.
    """
    pid, rows, gated = participant_ids(), [], {}
    for run in run_dirs:
        (path,) = run.glob("*_trials_#*.json")
        practice = [t for t in json.loads(path.read_text()) if t["trial_type"] == "practice"]
        try:
            iteration = practice_iterations(practice)
        except ValueError as error:
            raise ValueError(f"{run.name}: {error}") from None
        (log,) = run.glob("*_log_#*.json")
        scored = dict(re.findall(r'Experiment: practice iteration (\d+): ([^"]*)', log.read_text(encoding="utf-8")))
        for k in sorted(set(iteration)):
            trials = [t for t, i in zip(practice, iteration) if i == k]
            attention, rating = practice_scores(trials)
            rows.append(dict(participant=pid[run], iteration=k, attention=attention, rating=rating, trials=len(trials),
                             rating_criterion="rating" in scored.get(str(k), "")))
        # A run held to one practice session scores it too, and can repeat nothing (its participant file's
        # practice_max_iterations, written since 2026-11-25): not gated.
        (fields,) = run.glob("*_participant_#*.json")
        gated[pid[run]] = bool(scored) and _participant_fields(fields).get("practice_max_iterations") != 1
    return pd.DataFrame(rows), pd.Series(gated, name="practice gated")


def load_surprise(run_dirs: list[Path]) -> pd.DataFrame:
    """Trial 201 per participant: `said noticed` (yes to any of its three questions), `noticed when told` and
    `picked the altered image` (the 4-AFC). The typed elaborations are not read in. Rows only for runs with the file."""
    pid, rows = participant_ids(), {}
    for run in run_dirs:
        for path in run.glob("*_surprise_trial_#*.json"):
            s = json.loads(path.read_text())
            rows[pid[run]] = {"said noticed": any(s.get(f"q{k}_noticed") for k in (1, 2, 3)),
                              "noticed when told": s.get("noticed_when_told"),
                              "picked the altered image": s.get("identification_correct")}
    return pd.DataFrame.from_dict(rows, orient="index")


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


def colour_similarity(pair_index) -> pd.Series:
    """Per pair, how alike the two images' colours are, wherever in the image they sit: Pearson r of their RGB
    histograms (config.LAB_COLOUR_BINS bins a channel, every pixel of the first lab level's image).

    The crudest reading of "similar colours", there to check the strategy participants report against their
    ratings. Added after the second batch's reports (2026-11-23), so a reference fixed in advance from the third on.
    """
    folder = next(iter(config.LAB_LEVEL_DIRS.values()))
    histograms = {web_id: colour_histogram(folder / local) for web_id, local in json.loads(config.LAB_ID_MAP.read_text()).items()}
    return pd.Series({p: np.corrcoef(*(histograms[i] for i in p.split("_")))[0, 1] for p in pair_index}, name="colour")


def colour_histogram(path: Path) -> np.ndarray:
    bins = config.LAB_COLOUR_BINS
    with Image.open(path) as image:
        cell = np.asarray(image.convert("RGB")).reshape(-1, 3).astype(np.int64) * bins // 256
    return np.bincount((cell * [bins * bins, bins, 1]).sum(axis=1), minlength=bins ** 3)


def practice_pair_colours() -> pd.DataFrame:
    """The colour_similarity of each practice pair the rating score is computed on: one row per pair of
    config.LAB_PRACTICE_PAIRS_DIR named practice_<kind>_<k>_a/_b with kind similar or different (the folder's other
    pairs carry no kind), and its `colour`. It says what the rating criterion of the runs that had one makes of a
    rater who goes by colour alone. Empty without the folder."""
    rows = []
    for a in sorted(config.LAB_PRACTICE_PAIRS_DIR.glob("practice_*_a.*")):
        kind, k = a.stem.split("_")[1:3]
        if kind not in ("similar", "different"):
            continue
        b = a.with_name(a.name.replace("_a.", "_b."))
        rows.append(dict(kind=kind, pair=k, colour=np.corrcoef(colour_histogram(a), colour_histogram(b))[0, 1]))
    return pd.DataFrame(rows, columns=["kind", "pair", "colour"])


def pair_references(pilot: dict, vectors: pd.DataFrame, layer: str = config.DEFAULT_LAYER) -> pd.DataFrame:
    """Per pair, what a rating can be set against: the No-Letters pilot's mean rating (`consensus`), how much its
    raters disagreed on the pair (`dispute`, their SD), the model's similarity at each lab level (`model <level>`)
    and the two images' colour_similarity (`colour`)."""
    no_letters = pilot["ratings"]["no_letters"]
    refs = pd.DataFrame({"consensus": no_letters.mean(axis=1), "dispute": no_letters.std(axis=1)})
    for level in config.LAB_LEVEL_DIRS:
        refs[f"model {level}"] = distances.get_similarity(vectors, level, layer)
    refs["colour"] = colour_similarity(refs.index)
    return refs


# ----------------------------------------------------------------------------- agreement

def mean_pairwise_r(group: pd.DataFrame) -> float:
    """Mean Pearson r over every two raters: one rater against one other, whatever the group's size."""
    c = group.corr().to_numpy()
    return float(c[np.triu_indices_from(c, k=1)].mean())


def _subgroup_members(n: int, k: int, max_groups: int | None = None, seed: int = config.LAB_BOOTSTRAP_SEED) -> np.ndarray:
    """The k-rater subgroups of n raters, one row of rater positions each: every subgroup while there are at most
    `max_groups` (default config.LAB_MAX_SUBGROUPS), else that many drawn at random."""
    max_groups = config.LAB_MAX_SUBGROUPS if max_groups is None else max_groups
    if comb(n, k) <= max_groups:
        members = list(combinations(range(n), k))
    else:
        rng = np.random.default_rng(seed)
        members = [rng.choice(n, k, replace=False) for _ in range(max_groups)]
    return np.array(members, dtype=int).reshape(-1, k)


def subgroup_agreement(group: pd.DataFrame, k: int, max_groups: int | None = None, seed: int = config.LAB_BOOTSTRAP_SEED) -> np.ndarray:
    """mean_pairwise_r of the k-rater subgroups of `group`: what a group of k shows by chance of who came.

    Every subgroup while there are at most `max_groups` (default config.LAB_MAX_SUBGROUPS), else that many drawn at random.
    """
    c = group.corr().to_numpy()
    upper = np.triu_indices(k, k=1)
    return np.array([c[np.ix_(m, m)][upper].mean() for m in _subgroup_members(c.shape[0], k, max_groups, seed)])


def _ceilings(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """noise_ceiling's (lower, upper) per rater from the raters' covariance matrix, one group's (k x k) or a stack
    of them (groups x k x k): r of each rater with the sum of the others, and with the sum of all."""
    own, with_all = np.diagonal(cov, axis1=-2, axis2=-1), cov.sum(axis=-1)
    total = cov.sum(axis=(-2, -1))[..., None]
    return (with_all - own) / np.sqrt(own * (total - 2 * with_all + own)), with_all / np.sqrt(own * total)


def noise_ceiling(group: pd.DataFrame) -> pd.DataFrame:
    """The two bounds on how well one vector can follow a group's raters, per rater: `lower`, the rater against the
    mean rating of the others (analysis.leave_one_out_reliability, the report's R_LOO), and `upper`, against the
    mean rating of all, the rater included. A group's noise ceiling is their means over its raters.

    Both depend on the group's size, the lower one more, so two groups' ceilings are compared at one size
    (subgroup_ceilings) or through the formulas (expected_loo, expected_upper)."""
    lower, upper = _ceilings(group.cov().to_numpy())
    return pd.DataFrame({"lower": lower, "upper": upper}, index=group.columns)


def subgroup_ceilings(group: pd.DataFrame, k: int, measures: pd.DataFrame | None = None, max_groups: int | None = None,
                      seed: int = config.LAB_BOOTSTRAP_SEED) -> pd.DataFrame:
    """The noise_ceiling of the k-rater subgroups of `group`, which are subgroup_agreement's and in its order.

    One row per subgroup: its `lower` and `upper` (means over its raters) and, for each column of `measures` (one
    value per rater of `group`, e.g. r with the model at a level), the mean over the subgroup's raters -- so that a
    fit can be set against the ceiling of the very raters it was scored on.
    """
    members = _subgroup_members(group.shape[1], k, max_groups, seed)
    cov = group.cov().to_numpy()
    lower, upper = _ceilings(cov[members[:, :, None], members[:, None, :]])
    out = pd.DataFrame({"lower": lower.mean(axis=1), "upper": upper.mean(axis=1)})
    for name in ([] if measures is None else measures.columns):
        out[name] = measures[name].reindex(group.columns).to_numpy()[members].mean(axis=1)
    return out


def spearman_brown(r: float, k) -> np.ndarray:
    """Reliability of the mean of k raters whose single-rater agreement is r."""
    k = np.asarray(k, float)
    return k * r / (1 + (k - 1) * r)


def expected_loo(r: float, n) -> np.ndarray:
    """Expected leave-one-out r (one rater against the mean of the other n-1) at single-rater agreement r."""
    k = np.asarray(n, float) - 1
    return r * np.sqrt(k / (1 + (k - 1) * r))


def expected_upper(r: float, n) -> np.ndarray:
    """Expected upper noise ceiling (one rater against the mean of all n, the rater included) at single-rater agreement r."""
    n = np.asarray(n, float)
    return np.sqrt((1 + (n - 1) * r) / n)


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
    t["rating extremity"] = (t["rating"] - (lo + hi) / 2).abs()
    task_of(trials).add_columns(t)
    return t


def trial_relations(trials: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho, within each participant, of the rating and of its response time with each per-trial variable.

    One row per (participant, y, x): y in {"rating", "response time"}; x from the task's relation_x, the model's similarity at
    each level, and for the response time also the rating itself and its distance from the scale's midpoint.
    rho is NaN where the participant has fewer than MIN_CELL_TRIALS usable trials or no variation (e.g. no missed change).
    """
    t = _with_references(trials, refs)
    relation_x = task_of(trials).relation_x
    xs = relation_x[:3] + [c for c in refs.columns if c.startswith("model ")] + relation_x[3:]
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

    One row per (participant, condition, level, reference): the conditions are the task's (rating speed is the
    participant's own response times cut in thirds; the dot-task conditions use only trials with a colour change),
    the references are `consensus` and the model at each level. r is NaN below MIN_CELL_TRIALS trials or without variation.
    """
    t = trials.join(refs, on="pair_id")
    t["block"] = t["block"] + 1
    task_of(trials).add_columns(t)
    t["rating speed"] = t.groupby("participant")["rt"].transform(lambda rt: pd.qcut(rt, len(SPEEDS), labels=list(SPEEDS)).astype(str))
    references = ["consensus"] + [c for c in refs.columns if c.startswith("model ")]
    rows = []
    for pid, d in t.groupby("participant", sort=False):
        for condition, levels in task_of(trials).conditions.items():
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
    """For each of the task's subset_contrasts and each reference: r of the ratings with the reference inside the subset minus
    r outside it, per participant (`lowest`..`highest`) and averaged (`difference`).

    `p` is two-sided, by shuffling within each participant which trials count as the subset: the share of shuffles
    whose mean difference is at least as far from zero. One row per (contrast, reference); p is not corrected for
    the number of rows. A participant whose ratings do not vary inside the subset or outside it (one who gives
    nearly every pair the same rating) has no difference to give and is left out; `participants` counts the rest.
    """
    n_perm = config.LAB_PERMUTATIONS if n_perm is None else n_perm
    t = trials.join(refs, on="pair_id")
    references = ["consensus"] + [c for c in refs.columns if c.startswith("model ")]
    rng = np.random.default_rng(seed)
    rows = []
    for name, in_subset in task_of(trials).subset_contrasts.items():
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
                if not np.isnan(diff[0]):
                    observed.append(diff[0]), shuffled.append(diff[1:])
            row = dict(contrast=name, reference=reference, difference=np.nan, lowest=np.nan, highest=np.nan, p=np.nan,
                       participants=len(observed))
            if observed:
                means = np.nanmean(shuffled, axis=0)
                row.update(difference=np.mean(observed), lowest=np.min(observed), highest=np.max(observed),
                           p=float((np.abs(means) >= abs(np.mean(observed))).mean()))
            rows.append(row)
    return pd.DataFrame(rows)


def rating_shifts(trials: pd.DataFrame, refs: pd.DataFrame, contrasts: dict | None = None, n_perm: int | None = None,
                  seed: int = config.LAB_BOOTSTRAP_SEED) -> pd.DataFrame:
    """Is one kind of trial rated higher than another? For each contrast (default: the task's rating_shifts, plus the
    dot's two starting colours where it has two): the mean rating of a participant's trials of the first-named kind minus that
    of the other kind, averaged over participants.

    One row per contrast: `difference` in rating points, `lowest`..`highest` and `positive` (how many participants
    have a positive one), `adjusted` (the same difference after taking out, within each participant, what the pilot's
    consensus says of the pair: a check that the two kinds did not happen to get different pairs), `share` (how much
    of a participant's rating variance the split accounts for, averaged) and `p`, two-sided, by shuffling within
    each participant which of the compared trials are of the first kind. A participant with fewer than
    MIN_CELL_TRIALS trials of either kind is left out; `participants` counts the rest. Uncorrected.
    """
    n_perm = config.LAB_PERMUTATIONS if n_perm is None else n_perm
    own = contrasts is None
    contrasts = dict(task_of(trials).rating_shifts if own else contrasts)
    starts = sorted(trials["dot_start"].dropna().unique()) if own and "dot_start" in trials else []
    if len(starts) == 2:
        contrasts[f"dot starts {starts[0]} vs {starts[1]}"] = lambda t: (t["dot_start"].notna(), t["dot_start"] == starts[0])
    t = trials.join(refs["consensus"], on="pair_id").reset_index(drop=True)
    t["residual"] = np.nan
    for _, d in t.groupby("participant", sort=False):
        slope, intercept = np.polyfit(d["consensus"], d["rating"], 1)
        t.loc[d.index, "residual"] = d["rating"] - (intercept + slope * d["consensus"])
    rng = np.random.default_rng(seed)
    rows = []
    for name, split in contrasts.items():
        compared, first = split(t)
        observed, adjusted, shares, shuffled = [], [], [], []
        for _, d in t[compared].groupby("participant", sort=False):
            kind, y = first.loc[d.index].to_numpy(bool), d["rating"].to_numpy(float)
            k = int(kind.sum())
            if min(k, len(d) - k) < MIN_CELL_TRIALS:
                continue
            observed.append(y[kind].mean() - y[~kind].mean())
            adjusted.append(d["residual"].to_numpy()[kind].mean() - d["residual"].to_numpy()[~kind].mean())
            shares.append(np.corrcoef(y, kind)[0, 1] ** 2 if y.std() > 0 else 0.0)
            sums = y[rng.permuted(np.tile(np.arange(len(d)), (n_perm, 1)), axis=1)[:, :k]].sum(axis=1)
            shuffled.append(sums / k - (y.sum() - sums) / (len(d) - k))
        row = dict(contrast=name, difference=np.nan, adjusted=np.nan, lowest=np.nan, highest=np.nan, positive=0, share=np.nan,
                   p=np.nan, participants=len(observed))
        if observed:
            row.update(difference=np.mean(observed), adjusted=np.mean(adjusted), lowest=np.min(observed), highest=np.max(observed),
                       positive=int(np.sum(np.array(observed) > 0)), share=np.mean(shares),
                       p=float((np.abs(np.mean(shuffled, axis=0)) >= abs(np.mean(observed)) - 1e-12).mean()))
        rows.append(row)
    return pd.DataFrame(rows)


LETTER_SHIFT = "the pilot's letters: the same on both images vs different"


def pilot_letter_trials(pilot: dict) -> pd.DataFrame:
    """The Letters pilot's included raters trial by trial: participant, pair_id, rating and `same` (the two images
    carried the same letter). Read from the pilot's raw files (config.RESULT_FILES); empty where they are not at hand."""
    try:
        import parsing
        with progress.silenced():
            raw = pd.DataFrame(parsing.parse_file(config.RESULT_FILES["letters"], "letters"))
    except FileNotFoundError:
        return pd.DataFrame(columns=["participant", "pair_id", "rating", "same"])
    raw = raw[raw["participant_id"].isin(pilot["ratings"]["letters"].columns)]
    return pd.DataFrame(dict(participant=raw["participant_id"], pair_id=raw["pair_id"],
                             rating=raw["participant_similarity_rating"].astype(float), same=raw["letter1_shown"] == raw["letter2_shown"]))


def pilot_letter_shift(letter_trials: pd.DataFrame, refs: pd.DataFrame, n_perm: int | None = None) -> pd.DataFrame:
    """The yardstick for rating_shifts: the Letters pilot's own shift, the same letter on both images against two
    different letters, scored by the same function. Its one row holds no numbers (participants 0) without the trials."""
    if letter_trials.empty:
        return pd.DataFrame([dict(contrast=LETTER_SHIFT, participants=0)])
    return rating_shifts(letter_trials, refs, {LETTER_SHIFT: lambda t: (t["rating"].notna(), t["same"].astype(bool))}, n_perm)


def rating_by_load(trials: pd.DataFrame) -> pd.DataFrame:
    """Each participant's mean rating by what the task showed on the trial (the task's `load`: the number of colour
    changes, or how many of the two letters were the target): participants x numbers."""
    return trials.groupby(["participant", task_of(trials).load], sort=False)["rating"].mean().unstack().sort_index(axis=1)


def sequential_dependence(trials: pd.DataFrame, n_perm: int | None = None, seed: int = config.LAB_BOOTSTRAP_SEED) -> dict:
    """Does a rating lean on the one before it? Per participant, r between a rating and the previous trial's, over
    consecutive trials of one block (`per`); their mean (`mean`) and its two-sided p by shuffling each block's
    trial order (`p`). The pairs come in an order drawn per participant, so nothing but the rater links neighbours."""
    n_perm = config.LAB_PERMUTATIONS if n_perm is None else n_perm
    rng = np.random.default_rng(seed)

    def lag_r(blocks):  # blocks: one array of ratings per block, each (shuffles, trials)
        a = np.concatenate([b[:, 1:] for b in blocks], axis=1)
        b = np.concatenate([b[:, :-1] for b in blocks], axis=1)
        return _rowwise_r(a, b)

    per, shuffled = {}, []
    for pid, d in trials.sort_values("order").groupby("participant", sort=False):
        blocks = [g["rating"].to_numpy(float)[None, :] for _, g in d.groupby("block")]
        per[pid] = float(lag_r(blocks)[0])
        shuffled.append(lag_r([rng.permuted(np.tile(b, (n_perm, 1)), axis=1) for b in blocks]))
    per = pd.Series(per, name="lag-1 r")
    return dict(per=per, mean=float(per.mean()), p=float((np.abs(np.nanmean(shuffled, axis=0)) >= abs(per.mean()) - 1e-12).mean()))


def order_agreement(trials: pd.DataFrame, n_perm: int = 500, seed: int = config.LAB_BOOTSTRAP_SEED) -> dict:
    """Do two raters agree more on the pairs they saw in the same order? For every two raters, r over the pairs both
    saw in one order minus r over the pairs they saw in opposite orders: `mean` over rater pairs, `positive` of
    `pairs` rater pairs, and `p`, two-sided, by redrawing which pairs each rater saw in the pilot's order."""
    rating = trials.pivot(index="pair_id", columns="participant", values="rating").to_numpy(float)
    order = trials.pivot(index="pair_id", columns="participant", values="pilot_order").to_numpy(bool)
    i, j = np.triu_indices(rating.shape[1], k=1)

    def gaps(order):
        out = []
        for a, b in zip(i, j):
            same = order[:, a] == order[:, b]
            out.append(np.corrcoef(rating[same, a], rating[same, b])[0, 1] - np.corrcoef(rating[~same, a], rating[~same, b])[0, 1])
        return np.array(out)

    observed, rng = gaps(order), np.random.default_rng(seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        shuffled = np.array([np.nanmean(gaps(rng.permuted(order, axis=0))) for _ in range(n_perm)])
    return dict(mean=float(np.nanmean(observed)), positive=int((observed > 0).sum()), pairs=len(observed),
                p=float((np.abs(shuffled) >= abs(np.nanmean(observed)) - 1e-12).mean()))


def participant_relations(table: pd.DataFrame, n_perm: int | None = None, seed: int = config.LAB_BOOTSTRAP_SEED) -> pd.DataFrame:
    """Spearman rho across participants for each of PARTICIPANT_RELATIONS that `table` holds, with a two-sided p by
    shuffling one of the two measures over participants. A handful of participants: a screen, uncorrected."""
    n_perm = config.LAB_PERMUTATIONS if n_perm is None else n_perm
    rng = np.random.default_rng(seed)
    rows = []
    letters = TASKS[LETTER_COUNT]
    relations = letters.participant_relations if letters.correct in table else PARTICIPANT_RELATIONS
    for x, y in relations:
        if x not in table or y not in table:
            continue
        d = table[[x, y]].astype(float).dropna().rank()
        if len(d) < 4 or d[x].nunique() < 2 or d[y].nunique() < 2:
            continue
        rho = np.corrcoef(d[x], d[y])[0, 1]
        shuffled = _rowwise_r(rng.permuted(np.tile(d[x].to_numpy(), (n_perm, 1)), axis=1), np.tile(d[y].to_numpy(), (n_perm, 1)))
        rows.append(dict(x=x, y=y, rho=rho, n=len(d), p=float((np.abs(shuffled) >= abs(rho) - 1e-12).mean())))
    return pd.DataFrame(rows, columns=["x", "y", "rho", "n", "p"])


def response_times(trials: pd.DataFrame) -> pd.DataFrame:
    """Median response time (s) per participant within each rating, block, and the task's own breakdown: the phase
    of the last colour change ("none": a trial without one), or the target's count. One row per (participant, by, level)."""
    third, column, missing = task_of(trials).response_time_by
    t = trials.assign(by_rating=trials["rating"].astype(int), by_block=trials["block"] + 1,
                      by_change=trials[column].fillna(missing))
    rows = []
    for by, column in (("own rating", "by_rating"), ("block", "by_block"), (third, "by_change")):
        med = t.groupby(["participant", column], sort=False)["rt"].agg(["median", "size"])
        rows += [dict(participant=p, by=by, level=level, median=m, n=n) for (p, level), (m, n) in zip(med.index, med.to_numpy())]
    return pd.DataFrame(rows)


def block_table(trials: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """One row per (participant, block, 1-based): the share of its trials that were `clean` on the task, the
    median rating response time (s) and r of its ratings with the pilot's consensus."""
    t = trials.join(refs["consensus"], on="pair_id")
    rows = [dict(participant=p, block=b + 1, clean=d["clean"].mean(), rt=d["rt"].median(),
                 consensus=np.corrcoef(d["rating"], d["consensus"])[0, 1] if d["rating"].nunique() > 1 else np.nan)
            for (p, b), d in t.groupby(["participant", "block"], sort=False)]
    return pd.DataFrame(rows)


def _splits(n: int, k: int, seed: int = config.LAB_BOOTSTRAP_SEED) -> np.ndarray:
    """Every way to pick k of n participants as one row of a boolean array (row 0 is not special); a random
    config.LAB_PERMUTATIONS of them once there are more than config.LAB_MAX_SUBGROUPS."""
    if comb(n, k) <= config.LAB_MAX_SUBGROUPS:
        picks = list(combinations(range(n), k))
    else:
        rng = np.random.default_rng(seed)
        picks = [rng.choice(n, k, replace=False) for _ in range(config.LAB_PERMUTATIONS)]
    masks = np.zeros((len(picks), n), bool)
    for row, pick in zip(masks, picks):
        row[list(pick)] = True
    return masks


def group_contrasts(table: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    """Each numeric column of the per-participant `table`, compared between the participants `group` marks True
    and those it marks False.

    One row per measure: its mean in each group (`with`, `without`) and their sizes, `difference` (with minus
    without) and `p`, two-sided and exact: of every way to pick that many participants as the True group, the share
    whose difference is at least as far from zero. Four against five allow 126 ways, so no p below 1/126. Not
    corrected for the number of measures. A participant without a value is left out of that measure; a measure
    with fewer than two values in either group, or none that vary, gets no row.
    """
    group = group.reindex(table.index).astype(bool)
    rows = []
    for measure in table.columns[[pd.api.types.is_numeric_dtype(t) or pd.api.types.is_bool_dtype(t) for t in table.dtypes]]:
        ok = table[measure].notna()
        x, g = table.loc[ok, measure].to_numpy(float), group[ok].to_numpy()
        if min(g.sum(), (~g).sum()) < 2 or np.ptp(x) == 0:
            continue
        masks = _splits(len(x), int(g.sum()))
        diffs = (masks * x).sum(1) / masks.sum(1) - (~masks * x).sum(1) / (~masks).sum(1)
        observed = x[g].mean() - x[~g].mean()
        rows.append({"measure": measure, "with": x[g].mean(), "without": x[~g].mean(), "difference": observed,
                     "n with": int(g.sum()), "n without": int((~g).sum()),
                     "p": float((np.abs(diffs) >= abs(observed) - 1e-12).mean())})
    return pd.DataFrame(rows, columns=["measure", "with", "without", "difference", "n with", "n without", "p"])


def group_agreement(ratings: pd.DataFrame, group: pd.Series) -> dict:
    """Agreement between two raters, by whether both, neither or one of them is in the True group.

    Returns `pairs` (one row per two raters: a, b, r, kind in "both with" / "both without" / "one of each"), the
    mean r of each kind (`mean`), and two p-values, exact as in group_contrasts (the raters relabelled, the ratings
    kept): `p` for the difference between the two within-group means (is one group noisier?), and `p_between` for
    the mean over all same-group pairs minus the mean over mixed pairs (do the groups rate by different things?).
    Both are NaN unless each group holds two raters.
    """
    g = group.reindex(ratings.columns).astype(bool).to_numpy()
    c = ratings.corr().to_numpy()
    upper = np.triu(np.ones_like(c, bool), k=1)

    def within(mask):
        return c[np.ix_(mask, mask)][np.triu_indices(mask.sum(), k=1)].mean()

    i, j = np.nonzero(upper)
    kind = np.where(g[i] & g[j], "both with", np.where(~g[i] & ~g[j], "both without", "one of each"))
    pairs = pd.DataFrame(dict(a=ratings.columns[i], b=ratings.columns[j], r=c[i, j], kind=kind))
    def apart(mask):  # same-group pairs minus mixed pairs
        same = mask[i] == mask[j]
        return c[i, j][same].mean() - c[i, j][~same].mean()

    p = p_between = np.nan
    if min(g.sum(), (~g).sum()) >= 2:
        splits = _splits(len(g), int(g.sum()))
        diffs = np.array([within(m) - within(~m) for m in splits])
        p = float((np.abs(diffs) >= abs(within(g) - within(~g)) - 1e-12).mean())
        p_between = float((np.array([apart(m) for m in splits]) >= apart(g) - 1e-12).mean())
    return dict(pairs=pairs, mean=pairs.groupby("kind").r.mean(), p=p, p_between=p_between)


def participant_table(trials: pd.DataFrame, participants: pd.DataFrame, columns: dict[str, pd.Series]) -> pd.DataFrame:
    """One row per participant: how they rated, how they did on the task (its participant_columns), the given per-participant
    `columns` (agreement and model fits), then the dialog's fields and the experimenter's labels."""
    g = trials.groupby("participant", sort=False)
    table = pd.DataFrame({
        "trials": g.size(),
        "mean rating": g["rating"].mean(),
        "rating sd": g["rating"].std(),
        "rated lowest": g["rating"].apply(lambda r: (r == config.RATING_SCALE[0]).mean()),
        "median rating rt": g["rt"].median(),
        **task_of(trials).participant_columns(g),
    })
    return table.join(pd.DataFrame(columns)).join(participants)
