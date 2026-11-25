"""
Tests for the lab run type (lab.py, lab_run.py): synthetic run folders, tiny
images, the stub extractor and a made-up pilot cache -- nothing real is read.
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import config
import distances
import embeddings
import lab
import lab_run
import progress
from conftest import StubExtractor

IDS = [f"{i:03d}" for i in range(1, 81)]
PAIR_INDEX = [f"{a}_{b}" for a, b in zip(IDS[:40], IDS[40:])]
LEVELS = ("Full", "Deg3", "Tunnel")
NOMINAL = {"image1": 0.3, "mask1": 0.3, "image2": 0.3, "mask2": 0.3}


def trial(pair_id, rating, rt=1.0, change_times=(), hit_rts=(), false_alarms=0, swap=False, colors=None):
    """One main trial as the experiment writes it (the fields lab.py reads). `colors`: the dot's colours in turn."""
    a, b = pair_id.split("_")[::-1] if swap else pair_id.split("_")
    dot = dict(change_times=list(change_times), hit_rts=list(hit_rts), hits=sum(x is not None for x in hit_rts), false_alarms=false_alarms)
    if colors is not None:
        dot["colors"] = list(colors)
    return dict(
        trial_type="experimental", similarity_rating=rating, reaction_time=rt,
        image1_path=rf"D:\itay_tomer\stimuli\Chosen_425\scene{a}.png", image2_path=rf"D:\itay_tomer\stimuli\Chosen_425\scene{b}.png",
        stim_timings=dict(**{phase: dict(onset=0.0, duration=d, nominal=d) for phase, d in NOMINAL.items()}, color_change_task=dot),
    )


def practice_trial(kind, k, rating, missed=False, stray=False, swap=False, folder="task_familiarity_pairs"):
    """One practice trial on the pair `practice_<kind>_<k>` (kind: similar / different / other), with one colour change.
    `folder`: where its images lie (until 2026-11-25 one folder for every iteration, then `practice_iteration_<k>`)."""
    a, b = (rf"D:\itay_tomer\stimuli\{folder}\practice_{kind}_{k}_{side}.jpg" for side in ("ba" if swap else "ab"))
    return dict(trial_type="practice", image1_path=a, image2_path=b, similarity_rating=rating, stim_timings=dict(
        color_change_task=dict(change_times=[0.4], hits=0 if missed else 1, false_alarms=int(stray))))


def practice_pass(similar=(5, 6), different=(1, 2), **dot):
    """One pass over four practice pairs, two of each kind, rated as given."""
    return [practice_trial("similar", k, r, **dot) for k, r in enumerate(similar)] + \
           [practice_trial("different", k, r, **dot) for k, r in enumerate(different)]


@pytest.fixture
def lab_env(monkeypatch, tmp_path, patch_result_dirs):
    """A synthetic lab tree and pilot cache. Returns (runs root, write_run)."""
    experiment = tmp_path / "experiment"
    runs = experiment / "wet_runs"
    runs.mkdir(parents=True)
    id_map = experiment / "id_map.json"
    id_map.write_text(json.dumps({i: f"scene{i}.png" for i in IDS}))
    level_dirs = {}
    for k, level in enumerate(LEVELS):
        level_dirs[level] = experiment / "stimuli" / level
        level_dirs[level].mkdir(parents=True)
        for i in IDS:
            pixels = np.random.default_rng(1000 * k + int(i)).integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(level_dirs[level] / f"scene{i}.png")
    for name, value in dict(LAB_RUNS_DIR=runs, LAB_ID_MAP=id_map, LAB_LEVEL_DIRS=level_dirs, LAB_TRIALS_PER_BLOCK=8,
                            LAB_RESULTS_DIR=tmp_path / "lab_results", LAB_DATA_DIR=tmp_path / "lab_results" / "data",
                            LAB_PARTICIPANT_LABELS=runs / "participant_labels.csv", LAB_BOOTSTRAP_N=20,
                            LAB_PRACTICE_PAIRS_DIR=experiment / "stimuli" / "practice_pairs",
                            RESULT_FILES={g: experiment / f"pilot_{g}.txt" for g in lab.PILOT_GROUPS}).items():  # no such files
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(lab, "MIN_CELL_TRIALS", 3)
    monkeypatch.setattr(embeddings, "get_extractor", lambda backend=None: StubExtractor())

    # the web pilot's caches, as `python run.py` leaves them
    rng = np.random.default_rng(0)
    truth = rng.integers(1, 7, size=len(PAIR_INDEX))
    raters = {(g, f"{g}_{k}"): np.clip(truth + rng.integers(-1, 2, size=len(truth)), 1, 6) for g in lab.PILOT_GROUPS for k in range(5)}
    wide = pd.DataFrame(raters, index=pd.Index(PAIR_INDEX, name="pair_id"))
    wide.columns.names = ["condition", "participant_id"]
    wide.to_pickle(config.DATA_DIR / "sim_vectors.pkl")
    web = {(level, layer): {i: rng.random(4) for i in IDS} for level in LEVELS for layer in config.LAYERS}
    distances.save(distances.build_dist_vectors(web, PAIR_INDEX), config.DATA_DIR / "dist_vectors.pkl")
    with open(config.DATA_DIR / "analysis_results.pkl", "wb") as f:
        pickle.dump({config.BACKEND: {g: {"loo": pd.Series(0.5, index=[f"{g}_{k}" for k in range(5)])} for g in lab.PILOT_GROUPS}}, f)

    def write_run(batch, serial, ratings=None, suffix="_complete", trials=None, notes='"fine"', started=None, practice=None,
                  gated=False, surprise=None):
        """A run folder. `started` (default: one minute per serial) orders the runs; `practice` is its practice trials
        (default: one pass); `gated` writes the log line a gated practice leaves (True: iteration 1's, of the runs
        that held the rating to a minimum too; or the lines themselves); `surprise` its trial-201 record."""
        run = runs / batch / f"P{serial}_{started or f'2026-11-22_{1000 + serial}'}_to_1034{suffix}"
        run.mkdir(parents=True)
        if trials is None:
            if ratings is None:
                ratings = np.clip(truth + np.random.default_rng(serial).integers(-2, 3, size=len(truth)), 1, 6)
            order = np.random.default_rng(100 + serial).permutation(len(PAIR_INDEX))
            trials = [trial(PAIR_INDEX[k], int(ratings[k]), rt=2.0 - 0.02 * step + 0.001 * k, swap=bool(k % 2),
                            change_times=[0.1, 0.7][: step % 3], hit_rts=[0.4, None][: step % 3])
                      for step, k in enumerate(order)]
        (run / f"P{serial}_x_trials_#{serial}.json").write_text(json.dumps([*(practice or practice_pass()), *trials]))
        log = [dict(time="t", message="EyeLinkTracker: created")]
        if gated is True:
            gated = ["Experiment: practice iteration 1: attention 0.500 (min 0.75), rating 1.000 (min 0.75): failed, repeating"]
        log += [dict(time="t", message=line) for line in gated or []]
        (run / f"P{serial}_x_log_#{serial}.json").write_text(json.dumps(log))
        if surprise is not None:
            (run / f"P{serial}_x_surprise_trial_#{serial}.json").write_text(json.dumps(surprise))
        (run / f"P{serial}_x_participant_#{serial}.json").write_text(
            f'{{"participant_id": "P{serial}", "age": 22, "gender": "Female", "dominant_hand": "Right",'
            f' "dominant_eye": "Left",\n  "notes": {notes}\n}}')
        return run

    return runs, write_run


# ----------------------------------------------------------------------------- loading

def test_find_runs_pools_every_batch_and_leaves_test_runs_out(lab_env):
    runs, write_run = lab_env
    write_run("batch_b", 5), write_run("batch_a", 3), write_run("batch_a", 1)
    write_run("batch_a", 2, suffix="_aborted"), write_run("batch_a", 4, suffix="_complete_short-run")
    assert [lab.serial_of(d) for d in lab.find_runs()] == ["P1", "P3", "P5"]
    assert [lab.serial_of(d) for d in lab.find_runs(["batch_b"])] == ["P5"]
    with pytest.raises(FileNotFoundError):
        lab.find_runs(["no_such_batch"])


def test_participants_are_numbered_over_the_whole_pool_whatever_their_serials(lab_env):
    # the lab PC's serial starts over with each batch: both batches hold a P1
    runs, write_run = lab_env
    write_run("batch_b", 1, started="2026-11-23_1304"), write_run("batch_b", 4, started="2026-11-23_1414")
    write_run("batch_a", 3, started="2026-11-22_1005"), write_run("batch_a", 1, started="2026-11-22_1116")  # P3 ran before P1
    write_run("batch_a", 2, started="2026-11-22_1110", suffix="_aborted")  # not a participant: takes no number
    config.LAB_PARTICIPANT_LABELS.write_text("batch,serial,strategy\nbatch_a,P1,meaning\nbatch_b,P1,colour\n")
    ratings, trials = lab.load_runs(lab.find_runs(), PAIR_INDEX)
    assert list(ratings.columns) == ["S1", "S2", "S3", "S4"]
    table = lab.load_participants(lab.find_runs())
    assert table.serial.tolist() == ["P3", "P1", "P1", "P4"] and table.batch.tolist() == ["batch_a"] * 2 + ["batch_b"] * 2
    assert table.strategy.fillna("-").tolist() == ["-", "meaning", "colour", "-"]
    # a participant keeps the number when only its batch is run
    assert list(lab.load_runs(lab.find_runs(["batch_b"]), PAIR_INDEX)[0].columns) == ["S3", "S4"]


def test_two_runs_begun_in_one_minute_are_refused(lab_env):
    runs, write_run = lab_env
    write_run("batch_a", 1, started="2026-11-22_1005"), write_run("batch_b", 1, started="2026-11-22_1005")
    with pytest.raises(ValueError, match="same minute"):
        lab.participant_ids()


def test_pairs_are_matched_whatever_their_order_on_screen_or_in_the_run(lab_env):
    runs, write_run = lab_env
    ratings = [1 + k % 6 for k in range(len(PAIR_INDEX))]
    write_run("batch_a", 1, trials=[trial(p, r) for p, r in zip(PAIR_INDEX, ratings)])
    write_run("batch_a", 3, trials=[trial(p, r, swap=True) for p, r in zip(PAIR_INDEX, ratings)][::-1])
    table, trials = lab.load_runs(lab.find_runs(), PAIR_INDEX)
    assert list(table.columns) == ["S1", "S2"] and list(table.index) == PAIR_INDEX
    assert (table.S1 == ratings).all() and (table.S2 == ratings).all()
    assert trials[trials.participant == "S2"].block.tolist() == [k // 8 for k in range(len(PAIR_INDEX))]
    assert set(trials.batch) == {"batch_a"}


def test_a_run_that_shows_a_pair_twice_is_refused(lab_env):
    runs, write_run = lab_env
    write_run("batch_a", 1, trials=[trial(PAIR_INDEX[0], 3)] + [trial(p, 3) for p in PAIR_INDEX[:-1]])
    with pytest.raises(ValueError):
        lab.load_runs(lab.find_runs(), PAIR_INDEX)


def test_the_colour_change_task_is_read_per_trial(lab_env):
    runs, write_run = lab_env
    trials = [trial(p, 3) for p in PAIR_INDEX]
    trials[0] = trial(PAIR_INDEX[0], 3, change_times=[0.1, 0.7], hit_rts=[0.4, None], false_alarms=1)  # 0.7 s: in image 2
    trials[1] = trial(PAIR_INDEX[1], 3, change_times=[0.35], hit_rts=[0.5])  # 0.35 s: in mask 1
    trials[3] = trial(PAIR_INDEX[3], 3, change_times=[0.1], hit_rts=[None])  # missed, nothing else pressed
    trials[4] = trial(PAIR_INDEX[4], 3, change_times=[0.1], hit_rts=[0.4], false_alarms=1)  # caught, and a stray press
    write_run("batch_a", 1, trials=trials)
    t = lab.load_runs(lab.find_runs(), PAIR_INDEX)[1].set_index("pair_id")
    first, second, third = (t.loc[p] for p in PAIR_INDEX[:3])
    assert (first.changes, first.hits, first.false_alarms, first.dot_rt, first.last_change) == (2, 1, 1, 0.4, "image 2")
    assert (second.changes, second.hits, second.dot_rt, second.last_change) == (1, 1, 0.5, "mask 1")
    assert third.changes == 0 and np.isnan(third.dot_rt) and pd.isna(third.last_change)
    assert (first.clean, second.clean, third.clean) == (False, True, True)  # a missed change; all caught; nothing to catch
    assert not t.loc[PAIR_INDEX[3], "clean"] and not t.loc[PAIR_INDEX[4], "clean"]  # either failing alone spoils the trial


def test_the_dots_colours_and_the_order_of_the_two_images_are_read_per_trial(lab_env):
    # image 1 runs 0-0.3 s, mask 1 to 0.6, image 2 to 0.9 (NOMINAL)
    runs, write_run = lab_env
    trials = [trial(p, 3, colors=["white"]) for p in PAIR_INDEX]
    trials[0] = trial(PAIR_INDEX[0], 3, change_times=[0.35], hit_rts=[0.4], colors=["red", "white"])  # changed in mask 1
    trials[1] = trial(PAIR_INDEX[1], 3, change_times=[0.7], hit_rts=[0.4], colors=["red", "white"], swap=True)  # changed in image 2
    trials[2] = trial(PAIR_INDEX[2], 3, change_times=[0.1, 0.4], hit_rts=[0.4, 0.4], colors=["red", "white", "red"])  # there and back
    write_run("batch_a", 1, trials=trials)
    t = lab.load_runs(lab.find_runs(), PAIR_INDEX)[1].set_index("pair_id")
    a, b, c, d = (t.loc[p] for p in PAIR_INDEX[:4])
    assert (a.dot_same, a.change_in_image, a.dot_start, a.pilot_order) == (False, False, "red", True)
    assert (b.dot_same, b.change_in_image, b.pilot_order) == (True, True, False)  # still red as image 2 came on; images swapped
    assert (c.dot_same, c.change_in_image) == (True, True)  # red again before image 2
    assert d.dot_same and pd.isna(d.change_in_image) and d.dot_start == "white"  # nothing changed


def test_practice_scores_count_ties_as_half_and_a_stray_press_as_a_failed_trial():
    trials = practice_pass(similar=(5, 3), different=(3, 1))  # 5>3, 5>1, 3=3, 3>1: 3.5 of 4 comparisons
    trials[0] = practice_trial("similar", 0, 5, missed=True)
    trials[1] = practice_trial("similar", 1, 3, stray=True)
    assert lab.practice_scores(trials) == (0.5, 0.875)
    assert lab.practice_scores(practice_pass(similar=(1, 1), different=(6, 6)))[1] == 0  # the wrong way round
    assert lab.practice_scores(practice_pass(similar=(2, 2), different=(2, 2)))[1] == 0.5  # nothing told apart
    assert np.isnan(lab.practice_scores([practice_trial("other", 0, 3)])[1])


def test_practice_is_cut_into_its_iterations_and_the_gate_is_read_from_the_log(lab_env):
    runs, write_run = lab_env
    repeat = [practice_trial(t["image1_path"].split("_")[-3], t["image1_path"].split("_")[-2], 6 - t["similarity_rating"] + 1, swap=True)
              for t in practice_pass(missed=True)][::-1]  # the same four pairs: sides swapped, order reversed, ratings inverted
    write_run("batch_a", 1, practice=practice_pass(missed=True) + repeat, gated=True)
    write_run("batch_a", 2, gated=True)  # gated, and passed at once
    write_run("batch_a", 3)
    practice, gated = lab.load_practice(lab.find_runs())
    assert practice.participant.tolist() == ["S1", "S1", "S2", "S3"] and practice.iteration.tolist() == [1, 2, 1, 1]
    assert practice.attention.tolist() == [0, 1, 1, 1] and practice.rating.tolist() == [1, 0, 1, 1] and set(practice.trials) == {4}
    assert gated.to_dict() == {"S1": True, "S2": True, "S3": False}
    assert practice.rating_criterion.tolist() == [True, False, True, False]  # as the log has it: S1's second line is not written


def test_an_iteration_with_pairs_of_its_own_is_read_from_its_folder_and_the_rating_is_scored_where_the_named_pairs_are(lab_env):
    runs, write_run = lab_env
    first = [practice_trial(t["image1_path"].split("_")[-3], t["image1_path"].split("_")[-2], t["similarity_rating"], missed=True,
                            folder="practice_iteration_1") for t in practice_pass()]
    others = [practice_trial("pair", k, 3, folder=f"practice_iteration_{n}") for n, size in ((2, 3), (3, 4)) for k in range(size)]
    write_run("batch_a", 1, practice=first + others, gated=[
        "Experiment: practice iteration 1: attention 0.000 (min 0.75): failed, repeating",
        "Experiment: practice iteration 2: attention 1.000 (min 0.75): passed"])
    practice, gated = lab.load_practice(lab.find_runs())
    # eleven different pairs in three iterations of 4, 3 and 4: nothing repeats, so only the folders tell them apart
    assert practice.iteration.tolist() == [1, 2, 3] and practice.trials.tolist() == [4, 3, 4]
    assert practice.attention.tolist() == [0, 1, 1] and practice.rating[0] == 1 and practice.rating[1:].isna().all()
    assert gated["S1"] and not practice.rating_criterion.any()  # gated on the dot alone


def test_practice_that_repeats_one_set_of_pairs_must_divide_into_whole_passes(lab_env):
    runs, write_run = lab_env
    write_run("batch_a", 1, practice=practice_pass() + practice_pass()[:3])
    with pytest.raises(ValueError, match="P1_.*7 practice trials do not divide into passes over 4 pairs"):
        lab.load_practice(lab.find_runs())


def test_trial_201_is_read_without_its_typed_answers(lab_env):
    runs, write_run = lab_env
    write_run("batch_a", 1, surprise=dict(q1_noticed=False, q2_noticed=False, q3_noticed=True, q3_elaboration="it was blue",
                                          noticed_when_told=False, identification_correct=True))
    write_run("batch_a", 2)  # a run without trial 201
    table = lab.load_surprise(lab.find_runs())
    assert table.loc["S1"].tolist() == [True, False, True] and "S2" not in table.index
    assert "blue" not in table.to_string()


def test_participants_survive_hand_edited_notes_and_never_carry_them(lab_env):
    runs, write_run = lab_env
    write_run("batch_a", 1, notes='"Jane Roe, called the task "breezy" and left"')  # unescaped quotes: not valid JSON
    write_run("batch_a", 3)
    config.LAB_PARTICIPANT_LABELS.write_text("batch,serial,strategy\nbatch_a,P1,meaning\n")
    table = lab.load_participants(lab.find_runs())
    assert table.loc["S1", ["age", "dominant_eye", "strategy"]].tolist() == [22, "Left", "meaning"]
    assert pd.isna(table.loc["S2", "strategy"]) and lab.label_columns() == ["strategy"]
    assert "notes" not in table.columns and "Jane" not in table.to_string()


# ----------------------------------------------------------------------------- the model

def test_model_vectors_keep_a_cache_of_their_own_and_rebuild_for_a_new_level(lab_env, monkeypatch):
    extractor = StubExtractor()
    vectors = lab.model_vectors(PAIR_INDEX, extractor)
    calls = extractor.forward_calls
    assert calls > 0 and list(vectors.index) == PAIR_INDEX
    assert {level for level, _, _ in vectors.columns} == set(LEVELS)
    # the three levels' pixels differ, so must their vectors: equal ones would be one level's cache served for another
    assert not np.allclose(distances.get_similarity(vectors, "Full", "avgpool"), distances.get_similarity(vectors, "Deg3", "avgpool"))
    assert not any(config.EMBEDDINGS_DIR.rglob("*.pkl")), "the web set's embedding cache must stay untouched"
    assert (config.LAB_DATA_DIR / "embeddings" / "stub" / "Full_avgpool.pkl").exists()

    lab.model_vectors(PAIR_INDEX, extractor)
    assert extractor.forward_calls == calls, "a second call reads the cached table"

    monkeypatch.setattr(config, "LAB_LEVEL_DIRS", {**config.LAB_LEVEL_DIRS, "Deg05": config.LAB_LEVEL_DIRS["Deg3"]})
    assert "Deg05" in {level for level, _, _ in lab.model_vectors(PAIR_INDEX, extractor).columns}


# ----------------------------------------------------------------------------- statistics

def test_agreement_statistics_on_a_known_group():
    rng = np.random.default_rng(0)
    base = rng.normal(size=500)
    group = pd.DataFrame({k: base + rng.normal(size=500) for k in "abcd"})
    c = group.corr()
    pairs = [c.loc[a, b] for a, b in ("ab", "ac", "ad", "bc", "bd", "cd")]
    assert lab.mean_pairwise_r(group) == pytest.approx(np.mean(pairs))
    assert sorted(lab.subgroup_agreement(group, 2)) == pytest.approx(sorted(pairs))
    assert lab.subgroup_agreement(group, 4) == pytest.approx([np.mean(pairs)])
    drawn = lab.subgroup_agreement(group, 2, max_groups=3)  # fewer than the 6 there are: a random draw of them
    assert len(drawn) == 3 and all(any(d == pytest.approx(p) for p in pairs) for d in drawn)
    assert lab.per_rater_r(group, group.a).a == pytest.approx(1)


def test_projection_formulas_at_their_known_points():
    assert lab.spearman_brown(0.5, 2) == pytest.approx(2 / 3)
    assert lab.spearman_brown(0.3, 1) == pytest.approx(0.3)
    assert lab.expected_loo(0.3, 2) == pytest.approx(0.3)  # one rater against one other
    assert lab.expected_loo(0.25, 10 ** 9) == pytest.approx(0.5, abs=1e-4)  # against a noise-free mean: sqrt(r)


def test_bootstrap_interval_brackets_the_statistic_for_scalars_and_vectors():
    x = np.random.default_rng(1).normal(size=200)
    lo, hi = lab.bootstrap_pairs(lambda i: x[i].mean(), 200, n_boot=300)
    assert lo < x.mean() < hi
    assert lab.bootstrap_pairs(lambda i: [x[i].mean(), x[i].std()], 200, n_boot=50).shape == (2, 2)


# ----------------------------------------------------------------------------- relations

@pytest.fixture
def planted(lab_env):
    """One participant whose data hold known relations: response times fall with trial number, and the ratings
    equal the reference on trials without a colour change and run against it on the rest."""
    runs, write_run = lab_env
    refs = pd.DataFrame({"consensus": np.tile([1.0, 3.0, 4.0, 6.0], 10), "dispute": 1.0, "model Full": np.arange(40.0)}, index=PAIR_INDEX)
    trials = []
    for k, pair_id in enumerate(PAIR_INDEX):
        clean = k % 2 == 0
        rating = int(refs.consensus[pair_id]) if clean else 7 - int(refs.consensus[pair_id])
        rt = 50.0 if k == 0 else 5.0 - 0.1 * k  # one slow first trial: a block's median and its mean then differ
        trials.append(trial(pair_id, rating, rt=rt, change_times=[] if clean else [0.1], hit_rts=[] if clean else [0.4]))
    write_run("batch_a", 1, trials=trials)
    return lab.load_runs(lab.find_runs(), PAIR_INDEX)[1], refs


def test_trial_relations_recover_a_planted_relation_and_name_what_cannot_be_related(planted):
    trials, refs = planted
    rel = lab.trial_relations(trials, refs).set_index(["y", "x"])
    assert rel.loc[("response time", "trial number"), "rho"] == pytest.approx(-1)
    assert np.isnan(rel.loc[("rating", "missed a change"), "rho"])  # no change was ever missed: nothing to relate
    assert ("response time", "own rating") in rel.index and ("rating", "own rating") not in rel.index
    assert ("rating", "model Full") in rel.index


def test_conditional_agreement_is_computed_within_each_subset(planted):
    trials, refs = planted
    cond = lab.conditional_agreement(trials, refs)
    by_changes = cond[(cond.condition == "colour changes") & (cond.reference == "consensus")].set_index("level")
    assert by_changes.loc[0, "r"] == pytest.approx(1) and by_changes.loc[0, "n"] == 20
    assert by_changes.loc[1, "r"] == pytest.approx(-1)
    assert set(cond[cond.condition == "rating speed"].level) == set(lab.SPEEDS)
    assert set(cond.reference) == {"consensus", "model Full"}


def test_subset_contrasts_find_the_planted_difference_and_only_that(planted):
    trials, refs = planted
    table = lab.subset_contrasts(trials, refs, n_perm=400).set_index(["contrast", "reference"])
    row = table.loc[("no colour change vs some", "consensus")]
    assert row.difference == pytest.approx(2) and row.lowest == row.highest and row.p == 0  # r = 1 inside, -1 outside
    assert table.loc[("no colour change vs some", "model Full"), "p"] > 0.05  # the model reference holds no such split
    assert set(table.index.get_level_values("contrast")) == set(lab.SUBSET_CONTRASTS)
    assert set(table.participants) == {1}


def test_a_rater_whose_subset_ratings_do_not_vary_is_left_out_of_that_contrast(planted):
    trials, refs = planted
    flat = trials.assign(participant="S2", rating=np.where(trials.changes == 0, 1, trials.rating))  # all ones without a change
    table = lab.subset_contrasts(pd.concat([trials, flat], ignore_index=True), refs, n_perm=200).set_index(["contrast", "reference"])
    row = table.loc[("no colour change vs some", "consensus")]
    assert row.participants == 1 and row.difference == pytest.approx(2) and row.p == 0  # the planted rater alone, as before
    alone = lab.subset_contrasts(flat, refs, n_perm=50).set_index(["contrast", "reference"]).loc[("no colour change vs some", "consensus")]
    assert alone.participants == 0 and np.isnan(alone.difference) and np.isnan(alone.p)  # no number, not a p of zero


def test_response_times_are_medians_within_each_level(planted):
    trials, _ = planted
    rts = lab.response_times(trials)
    by_block = rts[rts.by == "block"].set_index("level")
    assert by_block.loc[1, "median"] == pytest.approx(np.median([50.0, *(5.0 - 0.1 * np.arange(1, 8))])) and by_block.loc[1, "n"] == 8
    assert set(rts[rts.by == "last change in"].level) == {"none", "image 1"}


def test_blocks_report_the_dot_task_and_the_agreement_block_by_block(planted):
    trials, refs = planted
    consensus = refs.consensus.loc[trials.pair_id].to_numpy()
    trials = trials.assign(rating=np.where(trials.block == 0, consensus, 7 - consensus),  # with the consensus in block 1, against it after
                           clean=trials.block != 1)  # and no clean trial in block 2
    blocks = lab.block_table(trials, refs).set_index("block")
    assert list(blocks.index) == [1, 2, 3, 4, 5] and blocks.clean.tolist() == [1, 0, 1, 1, 1]  # 40 trials, 8 a block
    assert blocks.loc[1, "rt"] == pytest.approx(np.median([50.0, *(5.0 - 0.1 * np.arange(1, 8))]))
    assert blocks.loc[1, "consensus"] == pytest.approx(1) and blocks.loc[2, "consensus"] == pytest.approx(-1)


def test_colour_similarity_reads_what_colours_two_images_hold_not_where(lab_env):
    folder = config.LAB_LEVEL_DIRS["Full"]
    red, green, blue = (np.array(c, np.uint8) for c in ([230, 20, 20], [20, 230, 20], [20, 20, 230]))
    half = np.concatenate([np.tile(red, (4, 8, 1)), np.tile(blue, (4, 8, 1))])  # red above blue
    Image.fromarray(half).save(folder / "scene001.png"), Image.fromarray(half[::-1].copy()).save(folder / "scene041.png")
    for name, colour in (("002", red), ("042", blue), ("003", green), ("043", blue), ("004", red), ("044", green)):
        Image.fromarray(np.tile(colour, (8, 8, 1))).save(folder / f"scene{name}.png")
    colour = lab.colour_similarity(PAIR_INDEX)
    assert list(colour.index) == PAIR_INDEX
    assert colour["001_041"] == pytest.approx(1)  # the same colours, the other way up
    assert max(colour["002_042"], colour["003_043"], colour["004_044"]) < 0  # no colour in common, whichever channel differs


def test_practice_pair_colours_are_read_per_named_pair(lab_env):
    assert lab.practice_pair_colours().empty  # no such folder: nothing to report, not an error
    folder = config.LAB_PRACTICE_PAIRS_DIR
    folder.mkdir()
    red, blue = (np.tile(np.array(c, np.uint8), (8, 8, 1)) for c in ([230, 20, 20], [20, 20, 230]))
    for name, pixels in (("similar_1_a", red), ("similar_1_b", red), ("different_1_a", red), ("different_1_b", blue),
                         ("pair_01_a", red), ("pair_01_b", blue)):  # the folder's other pairs carry no kind
        Image.fromarray(pixels).save(folder / f"practice_{name}.jpg")
    table = lab.practice_pair_colours().set_index("kind")
    assert sorted(table.index) == ["different", "similar"]
    assert table.loc["similar", "colour"] == pytest.approx(1) and table.loc["different", "colour"] < 0


# ----------------------------------------------------------------------------- biases

def shift_trials(shift: float, one_change: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three participants whose ratings are `shift` higher on trials without a colour change and `one_change` higher
    on trials with exactly one, and otherwise follow a pair value that is spread alike over every kind of trial.
    Returns (trials, refs)."""
    k = np.arange(len(PAIR_INDEX))
    base, changes = np.array([2.0, 3.0, 4.0])[k % 3], np.array([0, 1, 2, 0])[k % 4]
    rating = base + shift * (changes == 0) + one_change * (changes == 1)
    one = pd.DataFrame(dict(pair_id=PAIR_INDEX, order=k, block=k // 8, rating=rating, changes=changes, hits=changes,
                            false_alarms=0, dot_same=(changes == 0) | (k % 2 == 0), change_in_image=np.where(changes == 0, None, k % 8 < 4),
                            dot_start=np.where(k % 2, "red", "white"), pilot_order=(k // 2) % 2 == 0))
    trials = pd.concat([one.assign(participant=p) for p in ("S1", "S2", "S3")], ignore_index=True)
    return trials, pd.DataFrame({"consensus": base}, index=PAIR_INDEX)


def test_rating_shifts_find_a_planted_shift_and_no_other(lab_env, monkeypatch):
    monkeypatch.setattr(lab, "MIN_CELL_TRIALS", 5)
    table = lab.rating_shifts(*shift_trials(1.0), n_perm=1000).set_index("contrast")
    assert set(lab.RATING_SHIFTS) < set(table.index) and "dot starts red vs white" in table.index
    row = table.loc["no colour change vs some"]
    assert row.difference == pytest.approx(1.0, abs=0.15) and row.adjusted == pytest.approx(1.0, abs=0.15)
    assert row.positive == 3 and row.participants == 3 and row.p < 0.01 and 0.2 < row.share < 0.6
    assert abs(table.loc["two colour changes vs one", "difference"]) < 0.3 and table.loc["two colour changes vs one", "p"] > 0.05
    assert table.loc["image order as in the pilot vs reversed", "p"] > 0.05
    stray = table.loc["a stray press vs none"]  # nobody ever pressed without a change: no number, and not a p of zero
    assert stray.participants == 0 and np.isnan(stray.difference) and np.isnan(stray.p)
    flat = lab.rating_shifts(*shift_trials(0.0), n_perm=500).set_index("contrast")
    assert abs(flat.loc["no colour change vs some", "difference"]) < 0.15 and flat.loc["no colour change vs some", "p"] > 0.05
    # each contrast compares only the trials it names: one change sits 2 below, so "vs one", "vs two" and "vs some" differ
    trials, refs = shift_trials(1.0, one_change=-2.0)
    split = lab.rating_shifts(trials, refs, n_perm=200).set_index("contrast").difference
    assert split["no colour change vs one"] == pytest.approx(3.0, abs=0.3) and split["no colour change vs two"] == pytest.approx(1.0, abs=0.3)
    assert split["two colour changes vs one"] == pytest.approx(2.0, abs=0.3) and split["no colour change vs some"] == pytest.approx(2.0, abs=0.3)
    wild = trials.assign(rating=np.where(trials.order == 0, 60.0, trials.rating))  # one wild rating moves a mean, not a median
    mine = wild[(wild.participant == "S1") & (wild.changes == 0)].rating
    by = lab.rating_by_load(wild)
    assert list(by.columns) == [0, 1, 2] and by.loc["S1", 0] == pytest.approx(mine.sum() / 20) and mine.sum() / 20 > mine.median() + 2


def test_a_shift_that_only_mirrors_the_pairs_is_gone_once_the_pairs_are_taken_out(lab_env, monkeypatch):
    monkeypatch.setattr(lab, "MIN_CELL_TRIALS", 5)
    trials, refs = shift_trials(0.0)
    liked = trials.changes == 0  # the trials without a change happen to hold the pairs everyone calls more alike
    refs = pd.DataFrame({"consensus": np.where(liked[:len(PAIR_INDEX)], 5.0, 2.0)}, index=PAIR_INDEX)
    trials = trials.assign(rating=refs.consensus.loc[trials.pair_id].to_numpy())
    row = lab.rating_shifts(trials, refs, n_perm=200).set_index("contrast").loc["no colour change vs some"]
    assert row.difference == pytest.approx(3.0) and row.adjusted == pytest.approx(0, abs=1e-9)


def test_the_pilots_letter_shift_is_scored_from_its_raw_file_and_left_empty_without_it(lab_env, monkeypatch):
    from conftest import make_participant_trials
    pilot = lab.load_pilot()
    refs = pd.DataFrame({"consensus": pilot["ratings"]["no_letters"].mean(axis=1)})
    assert lab.pilot_letter_trials(pilot).empty and lab.pilot_letter_shift(lab.pilot_letter_trials(pilot), refs).iloc[0].participants == 0
    lines = []
    for rater in ("letters_0", "letters_1", "outsider"):  # the outsider is not among the pilot's included raters
        same = {p: k % 2 == 0 for k, p in enumerate(PAIR_INDEX)}
        raw = make_participant_trials(rater, "letters", PAIR_INDEX, ratings={p: 5 if same[p] else 3 for p in PAIR_INDEX})
        for t in raw:
            if t["trial_type"] == "psychophysics" and same[t["pair_id"]]:
                t["letter2_shown"] = t["letter1_shown"]
        lines.append(json.dumps(raw))
    config.RESULT_FILES["letters"].write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(lab, "MIN_CELL_TRIALS", 5)
    trials = lab.pilot_letter_trials(pilot)
    assert set(trials.participant) == {"letters_0", "letters_1"} and trials.same.sum() == len(PAIR_INDEX)
    row = lab.pilot_letter_shift(trials, refs, n_perm=300).iloc[0]
    assert row.contrast == lab.LETTER_SHIFT and row.difference == pytest.approx(2) and row.participants == 2 and row.p < 0.01


def test_sequential_dependence_sees_a_rater_who_alternates_and_not_one_who_does_not(planted):
    trials, _ = planted
    k = trials.order.to_numpy()
    swing = trials.assign(rating=np.where(k % 2, 6.0, 1.0))  # low, high, low, high: each rating the opposite of the last
    out = lab.sequential_dependence(swing, n_perm=500)
    assert out["per"]["S1"] == pytest.approx(-1) and out["mean"] == pytest.approx(-1) and out["p"] < 0.01
    # the same ratings, each block in an order that carries nothing over
    shuffled = swing.assign(rating=np.random.default_rng(3).permuted(swing.rating.to_numpy().reshape(5, 8), axis=1).ravel())
    out = lab.sequential_dependence(shuffled, n_perm=500)
    assert abs(out["mean"]) < 0.5 and out["p"] > 0.05


def test_order_agreement_sees_raters_who_agree_only_on_pairs_seen_the_same_way_round():
    k = np.arange(len(PAIR_INDEX))
    base = np.random.default_rng(0).integers(1, 7, len(k)).astype(float)
    flipped = k % 2 == 1  # the second rater saw every other pair the other way round, and rated those the other way
    a = pd.DataFrame(dict(participant="S1", pair_id=PAIR_INDEX, rating=base, pilot_order=True))
    b = pd.DataFrame(dict(participant="S2", pair_id=PAIR_INDEX, rating=np.where(flipped, 7 - base, base), pilot_order=~flipped))
    out = lab.order_agreement(pd.concat([a, b], ignore_index=True), n_perm=200)
    assert out["mean"] == pytest.approx(2) and (out["positive"], out["pairs"]) == (1, 1) and out["p"] < 0.05
    same = lab.order_agreement(pd.concat([a, b.assign(rating=base)], ignore_index=True), n_perm=100)
    assert same["mean"] == pytest.approx(0)


def test_participant_relations_rank_the_listed_measures_and_skip_a_missing_one(monkeypatch):
    monkeypatch.setattr(lab, "PARTICIPANT_RELATIONS", [("x", "up"), ("x", "down"), ("x", "absent"), ("x", "flat")])
    table = pd.DataFrame(dict(x=[1, 2, 3, 4, 5, 6], up=[1, 4, 9, 16, 25, 36], down=[9, 7, 5, 4, 2, 1], flat=3.0))
    out = lab.participant_relations(table, n_perm=2000).set_index("y")
    assert list(out.index) == ["up", "down"]  # nothing to relate a missing or a constant measure to
    assert out.loc["up", "rho"] == pytest.approx(1) and out.loc["down", "rho"] == pytest.approx(-1) and (out.n == 6).all()
    assert (out.p < 0.02).all()  # 2 of the 720 orders are that extreme


# ----------------------------------------------------------------------------- groups

def test_group_contrasts_give_exact_p_values_over_every_relabelling():
    group = pd.Series([True] * 4 + [False] * 5, index=list("abcdefghi"))
    table = pd.DataFrame({"apart": [9, 8, 7, 6, 1, 2, 3, 4, 5], "level": [1, 2, 3, 4, 5, 0, 1, 2, 2], "same": 3.0,
                          "holey": [1, 2, 3, np.nan, 2, 2, 2, 2, np.nan], "label": list("xxxxyyyyy")}, index=group.index)
    out = lab.group_contrasts(table, group).set_index("measure")
    assert list(out.index) == ["apart", "level", "holey"]  # no row for a constant or for text
    assert out.loc["apart", ["with", "without", "difference"]].tolist() == [7.5, 3.0, 4.5]
    assert out.loc["apart", "p"] == pytest.approx(2 / 126)  # of the 126 splits, this one and its mirror image (the four lowest)
    assert out.loc["level", "p"] > 0.5 and (out.loc["holey", ["n with", "n without"]] == [3, 4]).all()
    flipped = lab.group_contrasts(-table[["apart"]], group).iloc[0]
    assert flipped.difference == -4.5 and flipped.p == pytest.approx(2 / 126)  # two-sided
    lopsided = lab.group_contrasts(table[["apart"]].replace(9, 19), group).iloc[0]
    assert lopsided.p == pytest.approx(1 / 126)  # no other four reach a mean that far from the rest's


def test_group_agreement_tells_two_camps_from_one_noisier_group():
    rng = np.random.default_rng(0)
    one, other = rng.normal(size=300), rng.normal(size=300)
    camps = pd.DataFrame({f"r{k}": (one if k < 4 else other) + rng.normal(size=300) for k in range(9)})
    group = pd.Series([True] * 4 + [False] * 5, index=camps.columns)
    out = lab.group_agreement(camps, group)
    assert out["pairs"].kind.value_counts().to_dict() == {"one of each": 20, "both without": 10, "both with": 6}
    assert out["mean"]["both with"] > 0.4 and out["mean"]["both without"] > 0.4 and abs(out["mean"]["one of each"]) < 0.1
    assert out["p_between"] == pytest.approx(1 / 126) and out["p"] > 0.2  # two camps, neither the noisier
    noisier = camps.copy()
    noisier.iloc[:, :4] = np.column_stack([other + 4 * rng.normal(size=300) for _ in range(4)])  # one camp, four noisy raters
    out = lab.group_agreement(noisier, group)
    assert out["p"] < 0.05 and out["mean"]["both with"] < out["mean"]["both without"]
    assert np.isnan(lab.group_agreement(camps, pd.Series(False, index=camps.columns))["p"])  # nobody to compare with


# ----------------------------------------------------------------------------- the run type

def test_lab_run_end_to_end_smoke_pooled_then_one_batch(lab_env):
    runs, write_run = lab_env
    for serial in (1, 2, 3):
        write_run("batch_a", serial)
    for serial in (4, 5, 6):  # the second batch's practice is gated, and one of them needed a second pass
        write_run("batch_b", serial, gated=True, practice=practice_pass(missed=True) + practice_pass() if serial == 4 else None,
                  surprise=dict(q1_noticed=False, q2_noticed=False, q3_noticed=False, noticed_when_told=False, identification_correct=serial == 5))
    config.LAB_PARTICIPANT_LABELS.write_text("batch,serial,strategy\nbatch_a,P1,meaning\nbatch_a,P2,looks\n")

    A = lab_run.run()
    out = config.LAB_RESULTS_DIR / "pool"
    assert A["n"] == 6 and A["batches"] == ["batch_a", "batch_b"] and A["sub"] == {}  # 6 raters: no smaller than the pilot's 5
    assert {p.name for p in (out / "diagrams").iterdir()} == {f"{name}.png" for name in lab_run.FIGURES}
    assert all((out / f"{name}.csv").exists() for name in
               ("participants", "trial_relations", "conditional_agreement", "subset_contrasts", "response_times",
                "practice", "blocks", "group_contrasts", "rating_shifts", "participant_relations"))
    text = (out / "summary.md").read_text()
    assert "N = 6, 2 batches" in text and "batch_a" in text and "batch_b" in text
    assert -1 <= A["rbar"]["lab"] <= 1 and A["participants"].shape[0] == 6
    assert set(A["level_ci"].columns) == set(LEVELS) and len(A["contrasts"]) == 3
    # gated practice against the rest: three against three, so 20 relabellings and no p below 1/20
    assert A["gated"].tolist() == [False] * 3 + [True] * 3 and A["participants"]["practice iterations"].tolist() == [1, 1, 1, 2, 1, 1]
    contrasts = A["group_contrasts"].set_index("measure")
    assert {"r with consensus", "r with colour similarity", "mean rating", "practice attention (first)"} <= set(contrasts.index)
    assert contrasts.p.between(1 / 20 - 1e-9, 1).all() and set(A["group_agreement"]["mean"].index) == {"both with", "both without", "one of each"}
    assert "Gated practice (3) against the rest (3)" in text and "picked the altered image out of four: 1 of 3" in text
    assert text.count("gated*") == 3 and text.count("not gated") == 3  # batch_b's three were held to the rating too
    # the rating's own shifts: the fixture's raters have 13 or 14 trials of each number of colour changes
    shifts = A["rating_shifts"].set_index("contrast")
    assert set(lab.RATING_SHIFTS) <= set(shifts.index) and shifts.loc["no colour change vs some", "participants"] == 6
    assert A["letter_shift"].participants == 0 and "Does anything but the pair move the rating?" in text  # no raw pilot file here
    assert set(A["sequence"]["per"].index) == set(A["pids"]) and A["order_agreement"]["pairs"] == 15
    assert A["rbar_changed"] != pytest.approx(A["rbar"]["lab"])  # agreement over the pairs both raters saw with a change only

    B = lab_run.run(["batch_b"])
    assert B["pids"] == ["S4", "S5", "S6"] and set(B["sub"]) == set(lab.PILOT_GROUPS)  # 3 raters: groups of 3 from the pilot
    assert (config.LAB_RESULTS_DIR / "batch_b" / "diagrams" / "3_yardstick.png").exists()
    assert B["group_contrasts"].empty and (config.LAB_RESULTS_DIR / "batch_b" / "diagrams" / "12_groups.png").exists()  # all gated


def test_lab_run_needs_two_participants(lab_env):
    runs, write_run = lab_env
    write_run("batch_a", 1)
    with pytest.raises(ValueError, match="at least two"):
        lab_run.run()


def test_silenced_bars_draw_nothing_and_the_setting_ends_with_the_block():
    with progress.silenced():
        assert progress.bar(range(3)).disable
    assert not progress.bar(range(3), leave=False).disable


# ----------------------------------------------------------------------------- the letters' count question

def letter_trial(pair_id, rating, letters="LT", target="T", response=None, rt=1.0, letter_rt=0.6, swap=False):
    """One main trial under the letters' count question, as the experiment writes it. `response`: the true count by default."""
    a, b = pair_id.split("_")[::-1] if swap else pair_id.split("_")
    true = sum(letter == target for letter in letters)
    response = true if response is None else response
    return dict(
        trial_type="experimental", similarity_rating=rating, reaction_time=rt,
        image1_path=rf"D:\itay_tomer\stimuli\Chosen_425\scene{a}.png", image2_path=rf"D:\itay_tomer\stimuli\Chosen_425\scene{b}.png",
        stim_timings=dict(**{phase: dict(onset=0.0, duration=d, nominal=d) for phase, d in NOMINAL.items()}, letter_task=dict(
            letter1=letters[0], letter2=letters[1], orientation1=10.0, orientation2=200.0,
            letters_match=float(letters[0] == letters[1]), question="count", count_target=target, count_true=true,
            count_response=response, count_correct=float(response == true), count_rt=letter_rt)))


def letter_practice(right=(True, True, False, True), folder="practice_iteration_1"):
    """One practice pass under the count question over four pairs, two named of each kind: the count right or wrong as given."""
    out = []
    for (kind, k, rating), ok in zip((("similar", 0, 5), ("similar", 1, 6), ("different", 0, 1), ("different", 1, 2)), right):
        t = practice_trial(kind, k, rating, folder=folder)
        t["stim_timings"] = dict(letter_task=dict(letter1="L", letter2="T", letters_match=0.0, question="count", count_target="T",
                                                  count_true=1, count_response=1 if ok else 0, count_correct=float(ok), count_rt=0.5))
        out.append(t)
    return out


def letter_run(seed, same_bonus=0.0, wrong_every=0):
    """A participant's 40 main trials: the four letter pairs in turn, the target T in even blocks of 8 and L in odd
    ones, a rating that follows the pair index, `same_bonus` more under one letter on both images."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(PAIR_INDEX))
    trials = []
    for step, k in enumerate(order):
        letters, target = ("LL", "TT", "LT", "TL")[step % 4], "TL"[(step // 8) % 2]
        true = sum(x == target for x in letters)
        rating = int(np.clip(1 + k % 5 + (same_bonus if letters[0] == letters[1] else 0) + rng.integers(0, 2), 1, 6))
        wrong = wrong_every and step % wrong_every == 0
        trials.append(letter_trial(PAIR_INDEX[k], rating, letters, target, response=(true + 1) % 3 if wrong else None,
                                   rt=2.0 - 0.02 * step, letter_rt=0.5 + 0.01 * (step % 7), swap=bool(k % 2)))
    return trials


def test_a_run_s_task_is_read_from_its_first_main_trial(lab_env):
    runs, write_run = lab_env
    dot, counting = write_run("batch_a", 1), write_run("batch_a", 2, trials=letter_run(2))
    same = letter_run(3)
    for t in same:
        t["stim_timings"]["letter_task"] = dict(letter1="L", letter2="T", letters_match=0.0, identicality_correct=1.0)
    bare = [dict(t, stim_timings={k: v for k, v in t["stim_timings"].items() if k != "letter_task"}) for t in letter_run(4)]
    assert [lab.run_task(r) for r in (dot, counting, write_run("batch_a", 3, trials=same), write_run("batch_a", 4, trials=bare))] == [
        lab.DOT, lab.LETTER_COUNT, "letters-same", "no task"]
    assert set(lab.TASKS) == {lab.DOT, lab.LETTER_COUNT} and lab.TASKS[lab.DOT].rating_shifts is lab.RATING_SHIFTS


def test_runs_of_different_tasks_are_never_pooled(lab_env):
    runs, write_run = lab_env
    dots = [write_run("batch_a", s) for s in (1, 2)]
    counting = [write_run("batch_b", s, trials=letter_run(s)) for s in (3, 4)]
    with pytest.raises(ValueError, match=r"different tasks: colour-change \(2\), letters-count \(2\)"):
        lab.find_runs()
    assert lab.find_runs(task=lab.LETTER_COUNT) == counting and lab.find_runs(task=lab.DOT) == dots
    assert lab.find_runs(["batch_a"]) == dots and lab.find_runs(["batch_b"]) == counting  # one task each: no need to name it
    with pytest.raises(ValueError, match="analyses colour-change and letters-count, not 'letters-same'"):
        lab.find_runs(task="letters-same")
    with pytest.raises(FileNotFoundError, match="no completed run of the letters-count task"):
        lab.find_runs(["batch_a"], task=lab.LETTER_COUNT)
    # a participant keeps the number counted over every batch, whatever the task
    assert [lab.participant_ids()[r] for r in counting] == ["S3", "S4"]
    R, T = lab.load_runs(counting, PAIR_INDEX)
    assert list(R.columns) == ["S3", "S4"] and lab.task_of(T).name == lab.LETTER_COUNT
    with pytest.raises(ValueError, match="more than one task"):
        lab.load_runs(dots + counting, PAIR_INDEX)


def test_a_letters_count_run_loads_the_count_s_own_columns_and_scores_practice_on_it(lab_env):
    runs, write_run = lab_env
    run = write_run("batch_a", 1, trials=letter_run(1, wrong_every=4), practice=letter_practice())
    write_run("batch_a", 2, trials=letter_run(2), practice=letter_practice())
    R, T = lab.load_runs(lab.find_runs(), PAIR_INDEX)
    mine = T[T.participant == "S1"]
    assert {"clean", "letter_rt", "letters_match", "count_target", "count_true", "count_response"} <= set(T.columns)
    assert "changes" not in T and mine.clean.mean() == pytest.approx(0.75) and set(mine.count_target) == {"T", "L"}
    assert mine.groupby("block").count_target.nunique().eq(1).all()  # one target a block
    assert (mine.count_true == mine.count_response).mean() == pytest.approx(0.75) and mine.letters_match.mean() == pytest.approx(0.5)
    # practice: the share of counts right, and the rating score on the named pairs as ever
    assert lab.practice_scores(letter_practice()) == (0.75, 1.0)
    practice, gated = lab.load_practice(lab.find_runs())
    assert practice.attention.tolist() == [0.75, 0.75] and not gated.any()
    table = lab.participant_table(T, lab.load_participants(lab.find_runs()), {})
    assert table.loc["S1", "letter answers right"] == pytest.approx(0.75) and "clean dot trials" not in table
    assert table.loc["S1", "median letter rt"] == pytest.approx(mine.letter_rt.median())
    # what the same-letter shift costs the group is read off the trials under different letters
    assert (lab.TASKS[lab.LETTER_COUNT].unshifted(T) == ~T.letters_match).all()


def test_the_count_question_s_first_shift_is_the_pilot_s_same_letter_bias(lab_env):
    runs, write_run = lab_env
    for serial in (1, 2, 3):
        write_run("batch_a", serial, trials=letter_run(serial, same_bonus=2.0))
    R, T = lab.load_runs(lab.find_runs(), PAIR_INDEX)
    k = np.arange(len(PAIR_INDEX), dtype=float)
    refs = pd.DataFrame(dict(consensus=k, dispute=1 + k % 3), index=PAIR_INDEX)
    shifts = lab.rating_shifts(T, refs, n_perm=300).set_index("contrast")
    assert list(shifts.index) == list(lab.TASKS[lab.LETTER_COUNT].rating_shifts)  # no dot contrast, nor its start colours
    same = shifts.loc["the same letter on both images vs different"]
    assert same.difference > 1.2 and same.positive == 3 and same.participants == 3 and same.p < 0.05
    # the two kinds of same-letter trial differ in the count alone, not in the rating
    assert abs(shifts.loc["the target on both images vs on neither", "difference"]) < 0.8
    by = lab.rating_by_load(T)
    assert list(by.columns) == [0, 1, 2] and (by[0] > by[1]).all() and (by[2] > by[1]).all()  # the count 1 is two different letters
    cond = lab.conditional_agreement(T, refs)
    assert set(cond.condition) == {"block", "target count", "letter task", "letters", "rating speed"}
    rel = lab.trial_relations(T, refs)
    assert {"target count", "letter response time", "wrong count"} <= set(rel.x) and "colour changes" not in set(rel.x)
    assert set(lab.response_times(T).by) == {"own rating", "block", "target count"}
    assert set(lab.subset_contrasts(T, refs, n_perm=50).contrast) == {"same letters vs different", "fastest third vs the rest"}


def test_lab_run_end_to_end_smoke_on_a_letters_count_group_beside_a_colour_change_one(lab_env):
    runs, write_run = lab_env
    for serial in (1, 2, 3):
        write_run("batch_a", serial)  # the colour-change task
    line = "Experiment: practice iteration 1: attention 0.750 (min 0.75): passed"
    for serial in (4, 5, 6, 7):
        write_run("batch_b", serial, trials=letter_run(serial, same_bonus=1.0, wrong_every=5 if serial % 2 else 0),
                  practice=letter_practice(), gated=[line] if serial > 5 else False,
                  surprise=dict(q1_noticed=False, q2_noticed=False, q3_noticed=False, noticed_when_told=False, identification_correct=serial == 5))

    with pytest.raises(ValueError, match="different tasks"):
        lab_run.run()
    A = lab_run.run(task=lab.LETTER_COUNT)
    out = config.LAB_RESULTS_DIR / "pool_letters-count"  # named for its task: the pool holds two
    assert A["task"].name == lab.LETTER_COUNT and A["n"] == 4 and A["pids"] == ["S4", "S5", "S6", "S7"]
    assert {p.name for p in (out / "diagrams").iterdir()} == {f"{name}.png" for name in lab_run.FIGURES}
    table = pd.read_csv(out / "participants.csv", index_col=0)
    assert {"letter answers right", "letter answers right (block 1)", "median letter rt"} <= set(table.columns)
    assert not {"clean dot trials", "changes caught", "stray presses"} & set(table.columns)
    text = (out / "summary.md").read_text()
    assert "mean rating by how many of the two letters were the target" in text and "colour changes" not in text
    assert "the same letter on both images vs different" in text and "Gated practice (2) against the rest (2)" in text
    shifts = pd.read_csv(out / "rating_shifts.csv")
    assert shifts.contrast.iloc[0] == "the same letter on both images vs different" and shifts.contrast.iloc[-1] == lab.LETTER_SHIFT
    measures = set(A["group_contrasts"].measure)
    # (the fixture's raters answer the count equally fast, and a measure that does not vary is left out)
    assert "letter answers right" in measures and not {"clean dot trials", "changes caught"} & measures
    related = A["participant_relations"]
    assert "letter answers right" in set(related.x) | set(related.y) and "clean dot trials" not in set(related.x) | set(related.y)

    dots = lab_run.run(task=lab.DOT)
    assert dots["task"].name == lab.DOT and (config.LAB_RESULTS_DIR / "pool_colour-change" / "summary.md").exists()
    alone = lab_run.run(["batch_b"])  # one task in the selection: the folder keeps its plain name
    assert alone["n"] == 4 and (config.LAB_RESULTS_DIR / "batch_b" / "summary.md").exists()


def test_a_run_held_to_one_practice_session_is_not_gated_though_it_was_scored(lab_env):
    runs, write_run = lab_env
    line = "Experiment: practice iteration 1: attention 0.500 (min 0.75): failed, the experiment goes on"
    once, could_repeat = write_run("batch_a", 1, gated=[line]), write_run("batch_a", 2, gated=[line])
    for run, iterations in ((once, 1), (could_repeat, 3)):
        (path,) = run.glob("*_participant_#*.json")
        path.write_text(json.dumps(dict(participant_id="P", age=22, gender="Female", dominant_hand="Right", dominant_eye="Left",
                                        practice_max_iterations=iterations, notes="")))
    write_run("batch_a", 3, gated=True)  # a run before the key existed: gated, as its log says
    practice, gated = lab.load_practice(lab.find_runs())
    assert gated.to_dict() == {"S1": False, "S2": True, "S3": True}
