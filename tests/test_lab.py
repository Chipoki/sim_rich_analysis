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


def trial(pair_id, rating, rt=1.0, change_times=(), hit_rts=(), false_alarms=0, swap=False):
    """One main trial as the experiment writes it (the fields lab.py reads)."""
    a, b = pair_id.split("_")[::-1] if swap else pair_id.split("_")
    return dict(
        trial_type="experimental", similarity_rating=rating, reaction_time=rt,
        image1_path=rf"D:\itay_tomer\stimuli\Chosen_425\scene{a}.png", image2_path=rf"D:\itay_tomer\stimuli\Chosen_425\scene{b}.png",
        stim_timings=dict(
            **{phase: dict(onset=0.0, duration=d, nominal=d) for phase, d in NOMINAL.items()},
            color_change_task=dict(change_times=list(change_times), hit_rts=list(hit_rts),
                                   hits=sum(x is not None for x in hit_rts), false_alarms=false_alarms),
        ),
    )


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
                            LAB_PARTICIPANT_LABELS=runs / "participant_labels.csv", LAB_BOOTSTRAP_N=20).items():
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

    def write_run(batch, serial, ratings=None, suffix="_complete", trials=None, notes='"fine"'):
        run = runs / batch / f"P{serial}_2026-11-22_1005_to_1034{suffix}"
        run.mkdir(parents=True)
        if trials is None:
            if ratings is None:
                ratings = np.clip(truth + np.random.default_rng(serial).integers(-2, 3, size=len(truth)), 1, 6)
            order = np.random.default_rng(100 + serial).permutation(len(PAIR_INDEX))
            trials = [trial(PAIR_INDEX[k], int(ratings[k]), rt=2.0 - 0.02 * step + 0.001 * k, swap=bool(k % 2),
                            change_times=[0.1, 0.7][: step % 3], hit_rts=[0.4, None][: step % 3])
                      for step, k in enumerate(order)]
        practice = dict(trial_type="practice", image1_path="x", image2_path="y", similarity_rating=6)
        (run / f"P{serial}_x_trials_#{serial}.json").write_text(json.dumps([practice, *trials]))
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
    assert [lab.participant_of(d) for d in lab.find_runs()] == ["P1", "P3", "P5"]
    assert [lab.participant_of(d) for d in lab.find_runs(["batch_b"])] == ["P5"]
    with pytest.raises(FileNotFoundError):
        lab.find_runs(["no_such_batch"])


def test_find_runs_refuses_one_participant_id_in_two_batches(lab_env):
    runs, write_run = lab_env
    write_run("batch_a", 1), write_run("batch_b", 1)
    with pytest.raises(ValueError, match="P1"):
        lab.find_runs()


def test_pairs_are_matched_whatever_their_order_on_screen_or_in_the_run(lab_env):
    runs, write_run = lab_env
    ratings = [1 + k % 6 for k in range(len(PAIR_INDEX))]
    write_run("batch_a", 1, trials=[trial(p, r) for p, r in zip(PAIR_INDEX, ratings)])
    write_run("batch_a", 3, trials=[trial(p, r, swap=True) for p, r in zip(PAIR_INDEX, ratings)][::-1])
    table, trials = lab.load_runs(lab.find_runs(), PAIR_INDEX)
    assert list(table.columns) == ["P1", "P3"] and list(table.index) == PAIR_INDEX
    assert (table.P1 == ratings).all() and (table.P3 == ratings).all()
    assert trials[trials.participant == "P3"].block.tolist() == [k // 8 for k in range(len(PAIR_INDEX))]
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
    write_run("batch_a", 1, trials=trials)
    t = lab.load_runs(lab.find_runs(), PAIR_INDEX)[1].set_index("pair_id")
    first, second, third = (t.loc[p] for p in PAIR_INDEX[:3])
    assert (first.changes, first.hits, first.false_alarms, first.dot_rt, first.last_change) == (2, 1, 1, 0.4, "image 2")
    assert (second.changes, second.hits, second.dot_rt, second.last_change) == (1, 1, 0.5, "mask 1")
    assert third.changes == 0 and np.isnan(third.dot_rt) and pd.isna(third.last_change)


def test_participants_survive_hand_edited_notes_and_never_carry_them(lab_env):
    runs, write_run = lab_env
    write_run("batch_a", 1, notes='"Jane Roe, called the task "breezy" and left"')  # unescaped quotes: not valid JSON
    write_run("batch_a", 3)
    config.LAB_PARTICIPANT_LABELS.write_text("participant,strategy\nP1,meaning\n")
    table = lab.load_participants(lab.find_runs())
    assert table.loc["P1", ["age", "dominant_eye", "strategy"]].tolist() == [22, "Left", "meaning"]
    assert pd.isna(table.loc["P3", "strategy"]) and lab.label_columns() == ["strategy"]
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


def test_response_times_are_medians_within_each_level(planted):
    trials, _ = planted
    rts = lab.response_times(trials)
    by_block = rts[rts.by == "block"].set_index("level")
    assert by_block.loc[1, "median"] == pytest.approx(np.median([50.0, *(5.0 - 0.1 * np.arange(1, 8))])) and by_block.loc[1, "n"] == 8
    assert set(rts[rts.by == "last change in"].level) == {"none", "image 1"}


# ----------------------------------------------------------------------------- the run type

def test_lab_run_end_to_end_smoke_pooled_then_one_batch(lab_env):
    runs, write_run = lab_env
    for serial in (1, 2, 3):
        write_run("batch_a", serial)
    for serial in (4, 5, 6):
        write_run("batch_b", serial)
    config.LAB_PARTICIPANT_LABELS.write_text("participant,strategy\nP1,meaning\nP2,looks\n")

    A = lab_run.run()
    out = config.LAB_RESULTS_DIR / "pool"
    assert A["n"] == 6 and A["batches"] == ["batch_a", "batch_b"] and A["sub"] == {}  # 6 raters: no smaller than the pilot's 5
    assert {p.name for p in (out / "diagrams").iterdir()} == {f"{name}.png" for name in lab_run.FIGURES}
    assert all((out / f"{name}.csv").exists() for name in
               ("participants", "trial_relations", "conditional_agreement", "subset_contrasts", "response_times"))
    text = (out / "summary.md").read_text()
    assert "N = 6, 2 batches" in text and "batch_a" in text and "batch_b" in text
    assert -1 <= A["rbar"]["lab"] <= 1 and A["participants"].shape[0] == 6
    assert set(A["level_ci"].columns) == set(LEVELS) and len(A["contrasts"]) == 3

    B = lab_run.run(["batch_b"])
    assert B["pids"] == ["P4", "P5", "P6"] and set(B["sub"]) == set(lab.PILOT_GROUPS)  # 3 raters: groups of 3 from the pilot
    assert (config.LAB_RESULTS_DIR / "batch_b" / "diagrams" / "3_yardstick.png").exists()


def test_lab_run_needs_two_participants(lab_env):
    runs, write_run = lab_env
    write_run("batch_a", 1)
    with pytest.raises(ValueError, match="at least two"):
        lab_run.run()


def test_silenced_bars_draw_nothing_and_the_setting_ends_with_the_block():
    with progress.silenced():
        assert progress.bar(range(3)).disable
    assert not progress.bar(range(3), leave=False).disable
