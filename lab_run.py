"""
The lab run type. Run with:  python run.py --lab [--batch NAME ...]

Scores the lab experiment's participants the way the web pilot was scored,
with the pilot as yardstick. By default every batch folder under
config.LAB_RUNS_DIR is pooled; `--batch` restricts the run to the named
batch folders. Re-run it after each new batch: nothing is recomputed that is
cached (the model's embeddings and distance vectors do not depend on who was
run), so a re-run takes seconds.

Writes under config.LAB_RESULTS_DIR/<selection>/ -- `pool`, or the batch
names joined by `+`:
  summary.md                  the numbers to read first, also printed
  participants.csv            one row per participant
  trial_relations.csv         lab.trial_relations
  conditional_agreement.csv   lab.conditional_agreement
  subset_contrasts.csv        lab.subset_contrasts
  response_times.csv          lab.response_times
  practice.csv                lab.load_practice: each practice iteration's two scores
  blocks.csv                  lab.block_table
  group_contrasts.csv         lab.group_contrasts: gated practice against the rest
  rating_shifts.csv           lab.rating_shifts, and the pilot's letter shift as its last row
  participant_relations.csv   lab.participant_relations
  diagrams/*.png              one figure per function in FIGURES

compute() returns everything the figures and the summary show, as plain
pandas/numpy, so a write-up can be built from the same numbers (see
analysis/supplamentary/lab_first_look/).

Colours: the Letters / No-Letters pair plus one orange for the lab group
(validate_palette.js, light, all pairs: all checks pass). Colour marks the
group only; participants are told apart by labels, text stays in ink. Where
gated practice is set against the rest (figures 11 to 13), a filled marker
is a gated participant and a hollow one is not. Figure 13 gathers the five
panels a one-page summary needs (TAKEAWAY_PANELS: each draws on an axes it is
given, so a write-up can also place them one by one).
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from PIL import Image  # noqa: E402

import analysis  # noqa: E402
import config  # noqa: E402
import distances  # noqa: E402
import lab  # noqa: E402
import progress  # noqa: E402

WIDTH = 6.5  # inches: figures are drawn at the size they print in a text block
INK, MUTED, FAINT = "#1a1a1a", "#6b6b6b", "#b5b5b5"
C = {"letters": "#2166ac", "no_letters": "#b2182b", "lab": "#d4700a"}
NAME = {"letters": "Letters pilot", "no_letters": "No-Letters pilot", "lab": "the lab"}
ORANGES = LinearSegmentedColormap.from_list("oranges", ["#fff8f0", "#f0a860"])  # light enough for ink text
MAX_LABELLED = 12  # above this many participants, figures stop naming each one
MAX_RATINGS_LISTED = 5  # above this many, the example pairs give the mean rating, not each participant's
MARKERS = "osD^vP*X"
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.7, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.grid": True, "grid.color": "#e8e8e8",
    "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "axes.titlelocation": "left", "axes.titleweight": "bold",
}
RELATION_LABELS = {
    "trial number": "trial number", "pilot consensus": "the pilot's mean rating of the pair",
    "pilot dispute": "the pilot's disagreement on the pair", "colour changes": "colour changes on the trial",
    "dot response time": "response time to the colour change", "missed a change": "a colour change was missed",
    "own rating": "the rating given", "rating extremity": "the rating's distance from mid-scale",
    # the letters' count question
    "target count": "how many of the two letters were the target", "letter response time": "response time to the count question",
    "wrong count": "the count was wrong",
}
PARTICIPANT_X = {
    lab.DOT: {"changes caught": "colour changes caught", "median rating rt": "median rating RT (s)",
              "median dot rt": "median dot RT (s)", "rated lowest": "share rated lowest"},
    lab.LETTER_COUNT: {"letter answers right": "counts right", "median rating rt": "median rating RT (s)",
                       "median letter rt": "median count RT (s)", "rated lowest": "share rated lowest"},
}
# Per task (lab.TASKS): the per-participant measures gated practice is set against the rest on, after the ratings'
# own; what each of its conditions is called on an axis; and how the summary names its trials by load.
TASK_MEASURES = {
    lab.DOT: ["clean dot trials", "clean dot trials (block 1)", "r with consensus (block 1)", "changes caught", "stray presses",
              "median dot rt"],
    lab.LETTER_COUNT: ["letter answers right", "letter answers right (block 1)", "r with consensus (block 1)", "median letter rt"],
}
CONDITION_TITLES = {
    "block": "block", "colour changes": "colour changes on the trial", "dot task": "dot task on the trial",
    "last change in": "last colour change fell in", "rating speed": "speed of the rating (own thirds)",
    "target count": "how many of the two letters were the target", "letter task": "count question on the trial",
    "letters": "the two letters", "own rating": "the rating given",
}
LOAD_SENTENCE = {lab.DOT: ("number of colour changes", "the pairs both saw with a change"),
                 lab.LETTER_COUNT: ("how many of the two letters were the target", "the pairs both saw under different letters")}
CLEAN_TITLE = {lab.DOT: "Dot task: clean trials", lab.LETTER_COUNT: "Letter task: counts right"}


def level_label(level: str) -> str:
    return config.LEVEL_LABELS.get(level, level)


def _cross_r(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r between every column of a and every column of b."""
    a = (a - a.mean(0)) / a.std(0)
    b = (b - b.mean(0)) / b.std(0)
    return a.T @ b / len(a)


# ----------------------------------------------------------------------------- numbers

def compute(run_dirs: list[Path]) -> dict:
    """Everything the figures and the summary show, for the participants of `run_dirs` taken as one group."""
    pilot = lab.load_pilot()
    index = pilot["pair_index"]
    R, T = lab.load_runs(run_dirs, index)
    n, n_pairs = R.shape[1], len(R)
    if n < 2:
        raise ValueError("agreement between raters needs at least two participants: pool this batch with others")
    vectors = lab.model_vectors(index)  # outside the silenced block: a first run embeds for minutes and should say so
    with progress.silenced():
        return _compute(run_dirs, pilot, R, T, vectors)


def _compute(run_dirs: list[Path], pilot: dict, R: pd.DataFrame, T: pd.DataFrame, vectors: pd.DataFrame) -> dict:
    n, n_pairs = R.shape[1], len(R)
    refs = lab.pair_references(pilot, vectors)
    levels = list(config.LAB_LEVEL_DIRS)
    G = {"lab": R, **pilot["ratings"]}
    task = lab.task_of(T)
    A = dict(run_dirs=run_dirs, task=task, batches=sorted({d.parent.name for d in run_dirs}), pids=list(R.columns), n=n,
             n_pairs=n_pairs, levels=levels, ratings=R, trials=T, groups=G, refs=refs,
             sizes={g: G[g].shape[1] for g in G})

    # -- agreement between raters
    A["corr"] = R.corr()
    A["rbar"] = {g: lab.mean_pairwise_r(G[g]) for g in G}
    A["rbar_ci"] = lab.bootstrap_pairs(lambda i: lab.mean_pairwise_r(R.iloc[i]), n_pairs)
    A["rbar_rank"] = lab.mean_pairwise_r(R.rank())  # Pearson on ranks = Spearman
    A["sub"] = {g: lab.subgroup_agreement(G[g], n) for g in lab.PILOT_GROUPS if n < G[g].shape[1]}
    A["loo"], A["split"] = {}, {}
    for g in G:
        wide = lab.as_wide(G[g], g)
        A["loo"][g] = analysis.leave_one_out_reliability(wide, g)
        A["split"][g] = analysis.split_half_reliability(wide, g)
    batch_of = T.drop_duplicates("participant").set_index("participant")["batch"]
    A["batch_rbar"] = {b: lab.mean_pairwise_r(R[members]) if len(members) > 1 else np.nan
                       for b, members in ((b, list(batch_of.index[batch_of == b])) for b in A["batches"])}
    A["batch_sizes"] = batch_of.value_counts().to_dict()

    # -- each rater against the No-Letters pilot's mean rating (its own raters: leave-one-out)
    A["cons"] = {"lab": lab.per_rater_r(R, refs["consensus"]), "letters": lab.per_rater_r(G["letters"], refs["consensus"]),
                 "no_letters": A["loo"]["no_letters"]}
    # the group's mean rating against each pilot group's (and the two pilot groups' against each other, for scale)
    A["mean_vs_pilot"] = {g: np.corrcoef(R.mean(axis=1), G[g].mean(axis=1))[0, 1] for g in lab.PILOT_GROUPS}
    A["pilot_vs_pilot"] = np.corrcoef(G["letters"].mean(axis=1), G["no_letters"].mean(axis=1))[0, 1]

    # -- the model at each level
    S = refs[[f"model {lv}" for lv in levels]].set_axis(levels, axis=1)
    A["per"] = {g: pd.DataFrame({lv: lab.per_rater_r(G[g], S[lv]) for lv in levels}) for g in G}
    A["per_layer"] = {layer: pd.DataFrame({lv: lab.per_rater_r(R, distances.get_similarity(vectors, lv, layer)) for lv in levels})
                      for layer in config.LAYERS}
    A["avg_vector"] = pd.Series({lv: np.corrcoef(R.mean(axis=1), S[lv])[0, 1] for lv in levels})
    Rm, Sm = R.to_numpy(), S.to_numpy()
    A["level_ci"] = pd.DataFrame(lab.bootstrap_pairs(lambda i: _cross_r(Rm[i], Sm[i]).mean(0), n_pairs), index=["lo", "hi"], columns=levels)
    pairs = [(a, b) for k, a in enumerate(levels) for b in levels[k + 1:]]

    def differences(i):
        m = dict(zip(levels, _cross_r(Rm[i], Sm[i]).mean(0)))
        return [m[a] - m[b] for a, b in pairs]

    ci = lab.bootstrap_pairs(differences, n_pairs)
    A["contrasts"] = pd.DataFrame([dict(higher=a, lower=b, difference=d, lo=lo, hi=hi)
                                   for (a, b), d, lo, hi in zip(pairs, differences(np.arange(n_pairs)), ci[0], ci[1])])
    # the lab's pixels against the pilot's own: the id map's check, and what the two sets' masks change
    web = {lv: distances.get_similarity(pilot["model"], lv, config.DEFAULT_LAYER).loc[R.index] for lv in levels
           if (lv, config.DEFAULT_LAYER, "similarity") in pilot["model"].columns}
    A["colour"] = {g: lab.per_rater_r(G[g], refs["colour"]) for g in G}  # each rater against the pairs' colour similarity
    A["lab_web_r"] = {lv: np.corrcoef(S[lv], w)[0, 1] for lv, w in web.items()}
    A["web_pixels"] = {lv: lab.per_rater_r(G["no_letters"], w).mean() for lv, w in web.items()}

    # -- what goes with what
    A["trial_relations"] = lab.trial_relations(T, refs)
    A["conditional"] = lab.conditional_agreement(T, refs)
    A["subset_contrasts"] = lab.subset_contrasts(T, refs)
    A["response_times"] = lab.response_times(T)

    # -- practice, trial 201, the session block by block; then gated practice against the rest
    A["practice"], A["gated"] = lab.load_practice(run_dirs)
    A["blocks"] = lab.block_table(T, refs)
    A["practice_colour"] = lab.practice_pair_colours()
    first = A["practice"][A["practice"].iteration == 1].set_index("participant")
    block1 = A["blocks"][A["blocks"].block == 1].set_index("participant")
    A["participants"] = lab.participant_table(T, lab.load_participants(run_dirs), {
        "r with consensus": A["cons"]["lab"], "r with the others": A["loo"]["lab"],
        **{f"r with model ({level_label(lv)})": A["per"]["lab"][lv] for lv in levels},
        "r with colour similarity": A["colour"]["lab"],
        f"{task.correct} (block 1)": block1.clean, "r with consensus (block 1)": block1.consensus,
        "practice gated": A["gated"], "practice iterations": A["practice"].groupby("participant").size(),
        "practice attention (first)": first.attention, "practice rating (first)": first.rating,
    }).join(lab.load_surprise(run_dirs))
    # -- does anything but the pair move the rating: the dot, the trial before, the order of the two images
    A["letter_trials"] = lab.pilot_letter_trials(pilot)
    A["rating_shifts"] = lab.rating_shifts(T, refs)
    A["letter_shift"] = lab.pilot_letter_shift(A["letter_trials"], refs).iloc[0]
    A["rating_by_changes"] = lab.rating_by_load(T)  # by the number of colour changes, or by the target's count
    # agreement over only the pairs on which both raters saw the dot change (under the letters: different letters):
    # what the shift above costs the group
    A["rbar_changed"] = lab.mean_pairwise_r(T.assign(rating=T["rating"].where(task.unshifted(T))).pivot(
        index="pair_id", columns="participant", values="rating"))
    A["sequence"] = lab.sequential_dependence(T)
    A["order_agreement"] = lab.order_agreement(T)
    A["participant_relations"] = lab.participant_relations(A["participants"])
    A["group_measures"] = _group_measures(A)
    A["group_contrasts"] = lab.group_contrasts(A["group_measures"], A["gated"])
    A["group_agreement"] = lab.group_agreement(R, A["gated"])

    # -- two worked pairs: the one the group rated highest, and the one it rated lowest (ties: first in pair order)
    means = R.mean(axis=1)
    web_to_local = {web_id: local for local, web_id in lab.local_to_web().items()}
    A["examples"] = [dict(tag=tag, pair_id=pid, files=[web_to_local[i] for i in pid.split("_")], ratings=R.loc[pid],
                          ranks={lv: int((S[lv] > S.loc[pid, lv]).sum()) + 1 for lv in levels})
                     for tag, pid in (("A", means.idxmax()), ("B", means.idxmin()))]
    return A


def _group_measures(A: dict) -> pd.DataFrame:
    """The per-participant measures gated practice is set against the rest on, in the order they are shown."""
    table, per, levels, rel = A["participants"], A["per"]["lab"], A["levels"], A["trial_relations"]
    M = table[["r with consensus", "r with the others", "r with colour similarity"]].copy()
    for lv in levels:
        M[f"r with ResNet ({level_label(lv)})"] = per[lv]
    for lv in levels[1:]:
        M[f"ResNet: {level_label(levels[0])} minus {level_label(lv)}"] = per[levels[0]] - per[lv]
    M["rating RT against trial number (rho)"] = rel[(rel.y == "response time") & (rel.x == "trial number")].set_index("participant").rho
    rest = ["mean rating", "rating sd", "rated lowest", "median rating rt", *TASK_MEASURES[A["task"].name],
            "practice attention (first)", "practice rating (first)", "picked the altered image"]
    return M.join(table[[c for c in rest if c in table]].astype(float))


# ----------------------------------------------------------------------------- figures

def _spread(ys, gap: float) -> list[float]:
    """Label positions at least `gap` apart, each as near its own y as that allows (input order kept)."""
    order = np.argsort(ys)
    out = np.array(ys, float)[order]
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + gap)
    out -= (out - np.array(ys, float)[order]).mean()
    res = np.empty_like(out)
    res[order] = out
    return list(res)


def _group_phrase(A) -> str:
    return f"the lab's {A['n']}"


def _mean_line(wide: pd.DataFrame) -> pd.Series:
    """The mean over participants at each level, left out where fewer than two thirds of them have a value."""
    return wide.mean(axis=1).where(wide.notna().mean(axis=1) >= 2 / 3)


def fig_examples(A):
    s, gap, left, top, under = 0.80, 0.17, 1.20, 0.24, 0.40
    levels = A["levels"]
    s = min(s, (WIDTH - left - (len(levels) - 1) * gap) / (2 * len(levels)))
    H = top + 2 * (s + under)
    fig = plt.figure(figsize=(WIDTH, H))
    for c, lv in enumerate(levels):
        x = left + c * (2 * s + gap)
        fig.text((x + s) / WIDTH, 1 - 0.05 / H, level_label(lv) + (" (what the participants saw)" if lv == "Full" else ""),
                 ha="center", va="top", fontsize=8, fontweight="bold")
    for r, ex in enumerate(A["examples"]):
        y = H - top - (r + 1) * s - r * under
        fig.text(0.02 / WIDTH, (y + s * 0.78) / H, f"Pair {ex['tag']}", fontsize=8.5, fontweight="bold", va="center")
        if A["n"] <= MAX_RATINGS_LISTED:
            fig.text(0.02 / WIDTH, (y + s * 0.42) / H, "rated\n" + "  ".join(str(int(v)) for v in ex["ratings"]),
                     fontsize=8, va="center", linespacing=1.5)
            fig.text(0.02 / WIDTH, (y + s * 0.06) / H, "  ".join(ex["ratings"].index), fontsize=6.3, color=MUTED, va="center")
        else:
            fig.text(0.02 / WIDTH, (y + s * 0.36) / H, f"mean rating\n{ex['ratings'].mean():.1f}", fontsize=8, va="center", linespacing=1.5)
        for c, lv in enumerate(levels):
            x = left + c * (2 * s + gap)
            for k, f in enumerate(ex["files"]):
                ax = fig.add_axes([(x + k * s) / WIDTH, y / H, s / WIDTH * 0.985, s / H])
                ax.imshow(np.asarray(Image.open(config.LAB_LEVEL_DIRS[lv] / f).convert("RGB").resize((300, 300), Image.LANCZOS)))
                ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(True), sp.set_color(FAINT), sp.set_linewidth(0.5)
            fig.text((x + s) / WIDTH, (y - 0.07) / H, f"ResNet: rank {ex['ranks'][lv]} of {A['n_pairs']}", ha="center", va="top", fontsize=7.5)
    return fig


def fig_agreement(A):
    roomy = A["n"] <= 6
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.5 if roomy else 3.2), gridspec_kw=dict(
        width_ratios=[1, 1.12], wspace=0.42, left=0.07, right=0.985, top=0.86 if roomy else 0.89, bottom=0.17 if roomy else 0.14))
    pids, corr = A["pids"], A["corr"].to_numpy().copy()
    np.fill_diagonal(corr, np.nan)
    a.imshow(corr, cmap=ORANGES, vmin=0, vmax=0.45)
    if A["n"] <= MAX_LABELLED:
        for i in range(len(pids)):
            for j in range(len(pids)):
                same = i == j
                a.text(j, i, ("same\nrater" if roomy else "·") if same else f"{corr[i, j]:.2f}".replace("0.", ".").replace("-", "−"), ha="center",
                       va="center", fontsize=7 if same else 9.5 - 0.4 * max(0, len(pids) - 4), color=FAINT if same else INK)
        a.set_xticks(range(len(pids)), pids), a.set_yticks(range(len(pids)), pids)
    else:
        a.set_xticks([]), a.set_yticks([])
        a.set_xlabel(f"{len(pids)} raters; darker = higher r (white: {0:.0f}, darkest: .45+)")
    a.set_title(f"a  Each two raters: r over the {A['n_pairs']} pairs")
    upper = A["corr"].where(np.triu(np.ones(corr.shape, bool), k=1)).stack()
    x, y = upper.idxmax()
    lo, hi = config.RATING_SCALE
    g = pd.crosstab(A["ratings"][y], A["ratings"][x]).reindex(index=range(lo, hi + 1), columns=range(lo, hi + 1), fill_value=0).to_numpy()
    b.imshow(g, cmap=ORANGES, vmin=0, vmax=g.max(), origin="lower", extent=(lo - 0.5, hi + 0.5, lo - 0.5, hi + 0.5))
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            b.text(j + lo, i + lo, str(g[i, j]) if g[i, j] else "·", ha="center", va="center", fontsize=8.5, color=INK if g[i, j] else FAINT)
        b.add_patch(Rectangle((i + lo - 0.5, i + lo - 0.5), 1, 1, fill=False, ec=INK, lw=0.9))
    b.set_xticks(range(lo, hi + 1)), b.set_yticks(range(lo, hi + 1))
    b.set_xlabel(f"{x}'s rating"), b.set_ylabel(f"{y}'s rating")
    b.set_title(f"b  {x} against {y}: pairs per cell")
    for ax in (a, b):
        ax.grid(False), ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    return fig


def fig_yardstick(A):
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.29)
    bins = np.arange(-0.06, 0.861, 0.02)
    top = 1.0
    for g in lab.PILOT_GROUPS:
        if g in A["sub"]:
            ax.hist(A["sub"][g], bins, color=C[g], alpha=0.30, lw=0)
            ax.hist(A["sub"][g], bins, histtype="step", color=C[g], lw=1.1, label=f"{NAME[g].split()[0]}: groups of {A['n']}")
            top = max(top, ax.get_ylim()[1])
        else:  # the lab group is as large as the pilot's: only the whole pilot group is left to compare with
            ax.axvline(A["rbar"][g], color=C[g], lw=1.6, label=f"{NAME[g]}, all {A['sizes'][g]} raters")
    ax.axvspan(*A["rbar_ci"], color=C["lab"], alpha=0.16, lw=0)
    ax.axvline(A["rbar"]["lab"], color=C["lab"], lw=2.2, label="the lab (band: 95% over image pairs)")
    ax.set_ylim(0, top * 1.22)
    where = {g: np.median(A["sub"][g]) if g in A["sub"] else A["rbar"][g] for g in lab.PILOT_GROUPS}
    side = 1 if A["rbar"]["lab"] < where["letters"] else -1  # the Letters label goes on the side away from the lab's line
    ax.annotate(NAME["letters"], (where["letters"] + side * 0.012, top * 1.02), ha="left" if side > 0 else "right", va="bottom", fontsize=8)
    ax.annotate(NAME["no_letters"], (where["no_letters"], top * 1.02), ha="center", va="bottom", fontsize=8)
    ax.annotate(_group_phrase(A), (A["rbar"]["lab"] - side * 0.012, top * 1.02), ha="right" if side > 0 else "left", va="bottom",
                fontsize=8, fontweight="bold")
    ax.set_xlabel(f"agreement inside a group of {A['n']} raters (mean r between two of them)")
    ax.set_ylabel(f"groups of {A['n']}")
    ax.set_xlim(-0.06, 0.86)
    fig.legend(loc="lower center", ncol=3, handlelength=1.4, columnspacing=1.4, fontsize=7.2)
    return fig


def fig_raters(A):
    pids = A["pids"]
    cols = min(len(pids), 6)
    rows = -(-len(pids) // cols)
    fig = plt.figure(figsize=(WIDTH, 1.55 * rows + 1.95))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.55 * rows, 1.55], hspace=0.97 / rows,
                             left=0.085, right=0.985, top=1 - 0.2 / (1.55 * rows + 1.95), bottom=0.44 / (1.55 * rows + 1.95))
    top = outer[0].subgridspec(rows, cols, wspace=0.16, hspace=0.9)
    lo, hi = config.RATING_SCALE
    for k, p in enumerate(pids):
        ax = fig.add_subplot(top[k // cols, k % cols])
        share = A["ratings"][p].value_counts(normalize=True).reindex(range(lo, hi + 1), fill_value=0)
        ax.bar(share.index, 100 * share.values, width=0.72, color=C["lab"])
        ax.set_ylim(0, 72), ax.set_xticks(range(lo, hi + 1)), ax.grid(axis="x", visible=False)
        ax.set_title(f"a  {p}" if not k else p, pad=3)
        ax.text(0.97, 0.93, f"mean {A['ratings'][p].mean():.1f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
        if k // cols == rows - 1 or k + cols >= len(pids):
            ax.set_xlabel("rating")
        if k % cols:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(f"% of the {A['n_pairs']} pairs")
    ax = fig.add_subplot(outer[1])
    box = ax.get_position()
    ax.set_position([0.2, box.y0, box.x1 - 0.2, box.height])
    order = {"no_letters": 2, "letters": 1, "lab": 0}
    for g, y in order.items():
        v = np.sort(A["cons"][g].to_numpy())
        jitter = ((np.arange(len(v)) % 3) - 1) * 0.13 if g != "lab" or len(v) > MAX_LABELLED else np.zeros(len(v))
        ax.scatter(v, y + jitter, s=58 if g == "lab" else 26, color=C[g], edgecolor="white", linewidth=0.9, zorder=3)
    if A["n"] <= MAX_LABELLED:
        lab_r = A["cons"]["lab"]
        for p, x in zip(lab_r.index, _spread(lab_r.to_numpy(), 0.035)):
            ax.annotate(p, (lab_r[p], 0), xytext=(x, -0.42), ha="center", va="top", fontsize=7.5,
                        arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.5, shrinkA=0, shrinkB=3))
    ax.set_yticks(list(order.values()), ["No-Letters raters", "Letters raters", _group_phrase(A)])
    ax.set_ylim(-0.85, 2.5), ax.set_xlim(-0.1, 0.95), ax.grid(axis="y", visible=False)
    ax.set_xlabel("r with the No-Letters pilot's mean rating of each pair (its own raters: mean of the others)")
    ax.set_title("b  Each rater against the No-Letters consensus", pad=4)
    return fig


def fig_model(A):
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True, gridspec_kw=dict(wspace=0.08, left=0.085, right=0.985, top=0.9, bottom=0.2))
    levels = A["levels"]
    x = np.arange(len(levels))
    per = A["per"]["lab"]
    for p in per.index:
        a.plot(x, per.loc[p], "-o", color=C["lab"], lw=1.1, ms=4, mec="white", mew=0.7, alpha=0.9)
    if A["n"] <= MAX_LABELLED:
        last = per[levels[-1]]
        for p, y in zip(per.index, _spread(last.to_numpy(), 0.058)):
            a.annotate(p, (x[-1], last[p]), xytext=(x[-1] + 0.2, y), fontsize=7.5, va="center",
                       arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.5, shrinkA=0, shrinkB=3))
    mean = per.mean()
    a.errorbar(x, mean, yerr=[mean - A["level_ci"].loc["lo"], A["level_ci"].loc["hi"] - mean], fmt="-o", color=INK, lw=2, ms=5.5,
               capsize=2.5, elinewidth=1, zorder=4, label=f"mean of the {A['n']} (95% over image pairs)")
    a.plot([], [], "-o", color=C["lab"], lw=1.1, ms=4, label="one rater")
    a.set_title(f"a  {_group_phrase(A).capitalize()}"), a.set_ylabel("r of a rater's ratings with the ResNet")
    a.set_xlim(-0.3, x[-1] + 0.62), a.legend(loc="upper right", handlelength=1.6)
    for g, dx in (("no_letters", -0.2), ("letters", 0.0), ("lab", 0.2)):
        d = A["per"][g]
        for k, lv in enumerate(levels):
            b.scatter(np.full(len(d), k + dx), d[lv], s=11, color=C[g], alpha=0.32, lw=0, zorder=2)
        b.plot(x + dx, d.mean(), "-D", color=C[g], lw=1.6, ms=5, mec="white", mew=0.8, zorder=4, label=f"{NAME[g]} (n = {len(d)})")
    b.set_title("b  The same model vectors, three groups"), b.set_xlim(-0.45, x[-1] + 0.45)
    b.legend(loc="upper right", handlelength=1.6, title="mean (diamond) and each rater (dot)", title_fontsize=7.5)
    for ax in (a, b):
        ax.set_xticks(x, [level_label(lv) for lv in levels]), ax.axhline(0, color="#9a9a9a", lw=0.7), ax.grid(axis="x", visible=False)
        ax.set_xlabel("what the ResNet was shown")
    a.set_ylim(-0.14, 0.82)
    return fig


def fig_projection(A):
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.85), gridspec_kw=dict(wspace=0.27, left=0.085, right=0.985, top=0.91, bottom=0.34))
    N = np.arange(2, max(41, A["n"] + 6))
    panels = ((a, lab.expected_loo, {g: A["loo"][g].mean() for g in A["loo"]}, "a  One rater against the rest (R_LOO)"),
              (b, lab.spearman_brown, A["split"], "b  Reliability of the group mean"))
    for ax, f, observed, title in panels:
        for g in ("no_letters", "letters", "lab"):
            ax.plot(N, f(A["rbar"][g], N), color=C[g], lw=2 if g == "lab" else 1.4, label=f"{NAME[g]}: formula")
            ax.scatter([A["sizes"][g]], [observed[g]], s=46, color=C[g], edgecolor="white", linewidth=1, zorder=4)
        ax.fill_between(N, f(A["rbar_ci"][0], N), f(A["rbar_ci"][1], N), color=C["lab"], alpha=0.15, lw=0)
        ax.set_title(title), ax.set_xlabel("raters in the group"), ax.set_xlim(2, N[-1]), ax.set_ylim(0, 1)
        ax.set_xticks(sorted({A["n"], A["sizes"]["letters"], 30, 40} | ({10} if abs(A["n"] - 10) > 2 else set())))  # 9 and 10 would touch
    a.scatter([], [], s=46, color=MUTED, label="measured, at the group's own size")
    a.fill_between([], [], [], color=C["lab"], alpha=0.15, lw=0, label="the lab: 95% over image pairs")
    a.set_ylabel("r")
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=3, handlelength=1.4, columnspacing=1.6, fontsize=7.2)
    return fig


def _relation_label(x: str) -> str:
    return f"model similarity, {level_label(x[6:])}" if x.startswith("model ") else RELATION_LABELS[x]


def fig_trial_relations(A):
    rel = A["trial_relations"]
    panels = [("rating", "a  What goes with the rating"), ("response time", "b  What goes with the rating's response time")]
    xs = {y: list(dict.fromkeys(rel[rel.y == y].x)) for y, _ in panels}
    H = 0.235 * sum(len(v) for v in xs.values()) + 1.45
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, H), sharex=True, gridspec_kw=dict(
        height_ratios=[len(xs[y]) for y, _ in panels], hspace=0.07 + 0.5 / H, left=0.365, right=0.975, top=1 - 0.24 / H, bottom=0.5 / H))
    for ax, (y, title) in zip(axes, panels):
        for k, x in enumerate(xs[y]):
            row = len(xs[y]) - 1 - k
            v = rel[(rel.y == y) & (rel.x == x)].rho.dropna().to_numpy()
            if not len(v):  # a row that draws nothing would read as a zero
                ax.text(0, row, "no variation to relate", ha="center", va="center", fontsize=7, color=MUTED)
                continue
            jitter = (np.arange(len(v)) % 3 - 1) * 0.16 if len(v) > MAX_LABELLED else np.zeros(len(v))
            ax.scatter(v, row + jitter, s=24, color=C["lab"], alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
            ax.scatter([v.mean()], [row], marker="D", s=46, color=INK, edgecolor="white", linewidth=0.8, zorder=4)
        ax.axvline(0, color="#9a9a9a", lw=0.8)
        ax.set_yticks(range(len(xs[y])), [_relation_label(x) for x in reversed(xs[y])])
        ax.set_ylim(-0.6, len(xs[y]) - 0.4), ax.set_xlim(-0.8, 0.8), ax.grid(axis="y", visible=False)
        ax.set_title(title)
    axes[-1].set_xlabel("Spearman ρ over one participant's trials")
    axes[0].scatter([], [], s=24, color=C["lab"], label="one participant")
    axes[0].scatter([], [], marker="D", s=46, color=INK, label="their mean")
    axes[0].legend(loc="lower left", fontsize=7, handletextpad=0.2, borderpad=0.2)
    return fig


def fig_conditions(A):
    cond = A["conditional"]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.3), sharey=True, gridspec_kw=dict(wspace=0.1, hspace=0.62, left=0.1, right=0.985, top=0.93, bottom=0.12))
    titles = {condition: CONDITION_TITLES[condition] for condition in A["task"].conditions}  # five, the sixth panel the model's
    for ax, (condition, xlabel), letter in zip(axes.flat, titles.items(), "abcde"):
        d = cond[(cond.condition == condition) & (cond.reference == "consensus")]
        wide = d.pivot(index="level", columns="participant", values="r").reindex(list(dict.fromkeys(d.level))).dropna(how="all")
        x = np.arange(len(wide))
        ax.plot(x, wide.to_numpy(), "-o", color=C["lab"], lw=1.0, ms=3.5, mec="white", mew=0.6, alpha=0.9)
        ax.plot(x, _mean_line(wide), "-o", color=INK, lw=2, ms=5, zorder=4)
        ax.set_xticks(x, [str(v) for v in wide.index]), ax.set_xlim(-0.4, len(wide) - 0.6), ax.set_xlabel(xlabel)
        ax.set_title(f"{letter}  by {condition}"), ax.axhline(0, color="#9a9a9a", lw=0.7), ax.grid(axis="x", visible=False)
    ax = axes.flat[5]
    d = cond[(cond.condition == "rating speed") & cond.reference.str.startswith("model ")]
    for lv, style in zip(A["levels"], ("-", "--", ":", "-.")):
        m = d[d.reference == f"model {lv}"].groupby("level", sort=False).r.mean().reindex(list(lab.SPEEDS))
        ax.plot(range(len(m)), m, style, marker="o", color=INK, lw=1.6, ms=4.5, label=level_label(lv))
    ax.set_xticks(range(len(lab.SPEEDS)), lab.SPEEDS), ax.set_xlim(-0.4, len(lab.SPEEDS) - 0.6), ax.set_xlabel("speed of the rating (own thirds)")
    ax.set_title("f  ResNet, by speed"), ax.axhline(0, color="#9a9a9a", lw=0.7), ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", handlelength=2.2, fontsize=7, title="mean over raters", title_fontsize=7)
    axes[0, 0].set_ylabel("r with the pilot's consensus"), axes[1, 0].set_ylabel("r with the pilot's consensus")
    axes[1, 2].text(0.03, 0.04, "r with the ResNet", transform=axes[1, 2].transAxes, fontsize=7, color=MUTED)
    axes[0, 0].plot([], [], "-o", color=C["lab"], lw=1.0, ms=3.5, label="one rater")
    axes[0, 0].plot([], [], "-o", color=INK, lw=2, ms=5, label="mean")
    axes[0, 0].legend(loc="lower left", handlelength=1.5, fontsize=7, ncol=2)
    axes[0, 0].set_ylim(-0.25, 1.0)
    return fig


def fig_response_times(A):
    rts = A["response_times"]
    third = A["task"].response_time_by[0]
    order = {"own rating": None, "block": None, third: ["none", *lab.PHASE_NAMES] if third == "last change in" else None}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.45), sharey=True, gridspec_kw=dict(wspace=0.08, left=0.085, right=0.985, top=0.88, bottom=0.2))
    for ax, (by, levels), letter in zip(axes, order.items(), "abc"):
        cells = rts[(rts.by == by) & (rts.n >= lab.MIN_CELL_TRIALS)]  # a median of a few trials is not worth a point
        wide = cells.pivot(index="level", columns="participant", values="median")
        wide = wide.reindex(levels or sorted(wide.index)).dropna(how="all")
        x = np.arange(len(wide))
        ax.plot(x, wide.to_numpy(), "-o", color=C["lab"], lw=1.0, ms=3.5, mec="white", mew=0.6, alpha=0.9)
        ax.plot(x, _mean_line(wide), "-o", color=INK, lw=2, ms=5, zorder=4)
        ax.set_xticks(x, [str(v) for v in wide.index]), ax.set_xlim(-0.4, len(wide) - 0.6), ax.grid(axis="x", visible=False)
        ax.set_xlabel(CONDITION_TITLES[by])
        ax.set_title(f"{letter}  by {by}")
    axes[0].set_ylabel("median rating RT (s)"), axes[0].set_ylim(bottom=0)
    axes[0].plot([], [], "-o", color=C["lab"], lw=1.0, ms=3.5, label="one rater")
    axes[0].plot([], [], "-o", color=INK, lw=2, ms=5, label="mean")
    axes[0].legend(loc="lower left", handlelength=1.5, fontsize=7, ncol=2)
    return fig


def fig_participants(A):
    table = A["participants"]
    ys = ["r with consensus", f"r with model ({level_label(A['levels'][0])})"]
    labels = lab.label_columns()
    kinds = list(dict.fromkeys(table[labels[0]].fillna("not given"))) if labels else [None]
    xs = PARTICIPANT_X[A["task"].name]
    fig, axes = plt.subplots(len(ys), len(xs), figsize=(WIDTH, 3.55 if labels else 3.25), sharey="row", sharex="col",
                             gridspec_kw=dict(wspace=0.1, hspace=0.14, left=0.1, right=0.985, top=0.95, bottom=0.27 if labels else 0.15))
    for r, y in enumerate(ys):
        for c, (x, xlabel) in enumerate(xs.items()):
            ax = axes[r, c]
            for kind, marker in zip(kinds, MARKERS):
                d = table if kind is None else table[table[labels[0]].fillna("not given") == kind]
                ax.scatter(d[x], d[y], s=40, marker=marker, color=C["lab"], edgecolor="white", linewidth=0.8, zorder=3,
                           label=kind if (r, c) == (0, 0) else None)
            if len(table) <= MAX_LABELLED:
                for k, p in enumerate(table[y].sort_values().index):  # neighbours in height get labels on opposite sides
                    ax.annotate(p, (table.loc[p, x], table.loc[p, y]), xytext=(4, 4) if k % 2 else (4, -9), textcoords="offset points",
                                fontsize=6.5, color=MUTED)
            else:  # enough participants for a coefficient to mean something
                ax.text(0.04, 0.95, f"ρ = {table[x].corr(table[y], method='spearman'):+.2f}", transform=ax.transAxes, va="top", fontsize=7.5)
            ax.margins(0.22)
            ax.xaxis.set_major_locator(MaxNLocator(3, prune="both"))  # neighbouring panels' end ticks would touch
            if r == len(ys) - 1:
                ax.set_xlabel(xlabel)
            if not c:
                ax.set_ylabel(y.replace("model", "ResNet"))
    if labels:
        fig.legend(loc="lower center", ncol=min(len(kinds), 4), fontsize=7.2, handletextpad=0.3, columnspacing=1.4, title=f"{labels[0]} (experimenter's label)", title_fontsize=7.2)
    return fig


def _marker(A, p) -> dict:
    """A gated participant's marker is filled, any other's hollow."""
    return dict(mec=C["lab"], mew=1.0, mfc=C["lab"] if A["gated"][p] else "white")


def _draw_clean_course(a, A, name_participants=True):
    """One panel: the share of clean dot trials in each practice iteration, then in each block of the main task."""
    practice, blocks, gated, pids = A["practice"], A["blocks"], A["gated"], A["pids"]
    n_iter, n_blocks = int(practice.iteration.max()), int(blocks.block.max())
    for p in pids:
        pr, bl = practice[practice.participant == p], blocks[blocks.participant == p]
        line = dict(color=C["lab"], lw=1.0, ms=4.2, alpha=0.95, **_marker(A, p))
        a.plot(pr.iteration - 1, pr.attention, "-o", **line)
        a.plot(n_iter + bl.block - 1, bl.clean, "-o", **line)
        a.plot([pr.iteration.iloc[-1] - 1, n_iter], [pr.attention.iloc[-1], bl.clean.iloc[0]], ":", color=C["lab"], lw=0.8)
    for flag, style, name in ((True, "-", "mean, gated"), (False, "--", "mean, not gated")):
        members = [p for p in pids if gated[p] == flag]
        if members:
            wide = blocks[blocks.participant.isin(members)].pivot(index="block", columns="participant", values="clean")
            a.plot(n_iter + wide.index - 1, wide.mean(axis=1), style, color=INK, lw=1.8, zorder=4, label=name)
    if name_participants and A["n"] <= MAX_LABELLED:
        last = blocks[blocks.block == n_blocks].set_index("participant").clean.reindex(pids)
        for p, y in zip(pids, _spread(last.to_numpy(), 0.062)):
            a.annotate(p, (n_iter + n_blocks - 1, last[p]), xytext=(n_iter + n_blocks - 0.62, y), fontsize=6.8, va="center",
                       arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.5, shrinkA=0, shrinkB=3))
    a.plot([-0.45, n_iter - 0.5], [config.LAB_PRACTICE_MINIMUM["attention"]] * 2, color=MUTED, lw=0.9, ls=(0, (4, 2)))
    a.axvline(n_iter - 0.5, color="#9a9a9a", lw=0.7)
    a.set_xticks(range(n_iter + n_blocks), [str(k + 1) for k in range(n_iter)] + [str(k + 1) for k in range(n_blocks)])
    a.set_xlim(-0.45, n_iter + n_blocks + (0.15 if name_participants else -0.55)), a.set_ylim(0.3, 1.04)
    for x, text in (((n_iter - 1) / 2, "practice"), (n_iter + (n_blocks - 1) / 2, "main-task block")):
        a.annotate(text, (x, 0), xytext=(0, -17), xycoords=a.get_xaxis_transform(), textcoords="offset points", ha="center", va="top",
                   fontsize=plt.rcParams["axes.labelsize"], annotation_clip=False)
    a.set_ylabel("share of trials"), a.grid(axis="x", visible=False)


def _draw_pair_kinds(c, A):
    """One panel: r between two raters, by whether neither, one or both of the two had a gated practice."""
    kinds = [k for k in ("both without", "one of each", "both with") if k in A["group_agreement"]["mean"].index]
    pairs = A["group_agreement"]["pairs"]
    for x, kind in enumerate(kinds):
        r = np.sort(pairs[pairs.kind == kind].r.to_numpy())
        fill = {"both with": C["lab"], "both without": "white"}.get(kind, ORANGES(0.75))
        c.scatter(x + (np.arange(len(r)) % 5 - 2) * 0.09, r, s=20, facecolor=fill, edgecolor=C["lab"], linewidth=0.9, zorder=3)
        c.plot([x - 0.32, x + 0.32], [r.mean()] * 2, color=INK, lw=1.8, zorder=4)
    counts = pairs.kind.value_counts()
    c.set_xticks(range(len(kinds)), [{"both without": "neither\ngated", "one of each": "one\ngated", "both with": "both\ngated"}[k]
                                     + f"\n({counts[k]})" for k in kinds])
    c.set_xlim(-0.6, len(kinds) - 0.4), c.axhline(0, color="#9a9a9a", lw=0.7), c.grid(axis="x", visible=False)


def fig_practice(A):
    practice, pids = A["practice"], A["pids"]
    n_iter, n_blocks = int(practice.iteration.max()), int(A["blocks"].block.max())
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(WIDTH, 3.0), gridspec_kw=dict(
        width_ratios=[n_iter + n_blocks + 1.3, n_iter + 1.6, 4.6], wspace=0.3, left=0.075, right=0.99, top=0.89, bottom=0.31))
    _draw_clean_course(a, A)
    a.set_title(f"a  {CLEAN_TITLE[A['task'].name]}")
    for p in pids:
        pr = practice[practice.participant == p]
        b.plot(pr.iteration - 1, pr.rating, "-o", color=C["lab"], lw=1.0, ms=4.2, alpha=0.95, **_marker(A, p))
    if A["n"] <= MAX_LABELLED:
        first = practice[practice.iteration == 1].set_index("participant").rating.reindex(pids)
        for p, y in zip(pids, _spread(first.to_numpy(), 0.062)):
            b.annotate(p, (0, first[p]), xytext=(-0.62, y), fontsize=6.8, va="center", ha="right",
                       arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.5, shrinkA=0, shrinkB=3))
    if practice.rating_criterion.any():  # a minimum only in the runs that held the rating to one
        b.axhline(config.LAB_PRACTICE_MINIMUM["rating"], color=MUTED, lw=0.9, ls=(0, (4, 2)))
    b.axhline(0.5, color="#9a9a9a", lw=0.7)
    b.set_xticks(range(n_iter), [str(k + 1) for k in range(n_iter)]), b.set_xlim(-1.5, n_iter - 0.6), b.set_ylim(0, 1.04)
    b.set_xlabel("practice iteration"), b.set_title("b  Rating score"), b.grid(axis="x", visible=False)
    _draw_pair_kinds(c, A)
    c.set_title("c  Two raters' r")
    a.plot([], [], "-o", color=C["lab"], lw=1.0, ms=4.2, mfc=C["lab"], mec=C["lab"], label="gated practice")
    a.plot([], [], "-o", color=C["lab"], lw=1.0, ms=4.2, mfc="white", mec=C["lab"], label="not gated")
    a.plot([], [], color=MUTED, lw=0.9, ls=(0, (4, 2)), label="the gate's minimum")
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=5, handlelength=2.0, columnspacing=1.3, fontsize=7.0)
    return fig


def fig_groups(A):
    M, gated, contrasts = A["group_measures"], A["gated"], A["group_contrasts"].set_index("measure")
    rows = list(M.columns)
    H = 0.262 * len(rows) + 1.0
    fig, ax = plt.subplots(figsize=(WIDTH, H))
    fig.subplots_adjust(left=0.345, right=0.70, top=1 - 0.3 / H, bottom=0.72 / H)
    if contrasts.empty:  # a blank panel would read as no difference
        ax.text(0.5, 0.5, "every participant of this selection had the same kind of practice:\nnothing to compare",
                transform=ax.transAxes, ha="center", va="center", fontsize=8, color=MUTED)
    batches = A["participants"]["batch"]
    Z = (M - M.mean()) / M.std()
    for k, measure in enumerate(rows):
        y = len(rows) - 1 - k
        for flag, dy in ((False, 0.15), (True, -0.15)):
            z = Z.loc[gated[gated == flag].index, measure].dropna()
            for p, v in z.items():
                ax.scatter([v], [y + dy], s=20, marker=MARKERS[A["batches"].index(batches[p]) % len(MARKERS)], zorder=3,
                           facecolor=C["lab"] if flag else "white", edgecolor=C["lab"], linewidth=0.9)
            if len(z):
                ax.plot([z.mean()] * 2, [y + dy - 0.14, y + dy + 0.14], color=INK, lw=1.6, zorder=4)
        if measure in contrasts.index:
            row = contrasts.loc[measure]
            for x, text in ((1.04, f"{row['with']:.2f}"), (1.24, f"{row['without']:.2f}"), (1.44, f"{row['p']:.3f}")):
                ax.text(x, y, text, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7.2)
    for x, text in ((1.04, "gated"), (1.24, "not"), (1.44, "p")):
        ax.text(x, len(rows) - 0.42, text, transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=7.2, fontweight="bold")
    ax.axvline(0, color="#9a9a9a", lw=0.8)
    ax.set_yticks(range(len(rows)), list(reversed(rows)))
    ax.set_ylim(-0.6, len(rows) - 0.4), ax.set_xlim(-2.9, 2.9), ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"each measure in SDs from the mean of the {A['n']} participants")
    ax.scatter([], [], s=20, facecolor=C["lab"], edgecolor=C["lab"], label="gated practice")
    ax.scatter([], [], s=20, facecolor="white", edgecolor=C["lab"], label="not gated")
    ax.plot([], [], color=INK, lw=1.6, label="group mean")
    for b, batch in enumerate(A["batches"] if len(A["batches"]) > 1 else []):
        ax.scatter([], [], s=20, marker=MARKERS[b % len(MARKERS)], facecolor="white", edgecolor=MUTED, label=f"batch {b + 1}")
    fig.legend(loc="lower center", ncol=6, handletextpad=0.3, columnspacing=1.2, fontsize=7.0)
    return fig


def _draw_agreement_levels(ax, A):
    """One panel: mean r between two raters in the lab group (band: 95% over image pairs) and in each pilot group
    (band: the middle 90% of its groups of the lab's size)."""
    for g, y in (("no_letters", 2), ("letters", 1), ("lab", 0)):
        lo, hi = A["rbar_ci"] if g == "lab" else np.percentile(A["sub"][g], [5, 95]) if g in A["sub"] else (A["rbar"][g],) * 2
        ax.plot([lo, hi], [y, y], color=C[g], lw=7, alpha=0.3, solid_capstyle="butt")
        ax.scatter([A["rbar"][g]], [y], s=58, color=C[g], edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(f"{A['rbar'][g]:.2f}".replace("0.", "."), (A["rbar"][g], y), xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    ax.set_yticks([2, 1, 0], [NAME["no_letters"], NAME["letters"], _group_phrase(A)])
    ax.set_ylim(-0.55, 2.85), ax.set_xlim(0, 0.75), ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"mean r between two raters (pilot bands: any {A['n']} of its raters)")


def _draw_camps(ax, A):
    """One panel: each rater by how far the ratings follow the pilot's consensus and the pairs' colour similarity."""
    for g in lab.PILOT_GROUPS:
        ax.scatter(A["cons"][g], A["colour"][g], s=13, color=C[g], alpha=0.5, lw=0, label=f"{NAME[g]} rater")
    cons, colour = A["cons"]["lab"], A["colour"]["lab"]
    for p in A["pids"]:
        ax.plot([cons[p]], [colour[p]], "o", ms=6.5, zorder=3, **_marker(A, p))
    if A["n"] <= MAX_LABELLED:  # names to the right of the points, moved apart where the points crowd
        for p, y in zip(A["pids"], _spread(colour.to_numpy(), 0.032)):
            ax.annotate(p, (cons[p], colour[p]), xytext=(cons[p] + 0.055, y), fontsize=6.5, color=MUTED, va="center",
                        arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.5, shrinkA=0, shrinkB=3))
    ax.plot([], [], "o", ms=6, mfc=C["lab"], mec=C["lab"], label="lab, gated practice")
    ax.plot([], [], "o", ms=6, mfc="white", mec=C["lab"], label="lab, not gated")
    ax.axhline(0, color="#9a9a9a", lw=0.7)
    ax.set_xlabel("r with the pilot's consensus"), ax.set_ylabel("r with colour similarity")


def _draw_levels(ax, A):
    """One panel: how far a rater's ratings follow the ResNet, by what the ResNet was shown."""
    levels, per = A["levels"], A["per"]
    x = np.arange(len(levels))
    for p in per["lab"].index:
        ax.plot(x, per["lab"].loc[p], "-", color=C["lab"], lw=0.9, alpha=0.75)
    ax.plot([], [], "-", color=C["lab"], lw=0.9, label="one lab rater")
    for g in lab.PILOT_GROUPS:
        ax.plot(x, per[g].mean(), "--D", color=C[g], lw=1.2, ms=3.5, label=f"{NAME[g]} (mean)")
    mean = per["lab"].mean()
    ax.errorbar(x, mean, yerr=[mean - A["level_ci"].loc["lo"], A["level_ci"].loc["hi"] - mean], fmt="-o", color=INK, lw=2, ms=5,
                capsize=2.5, elinewidth=1, zorder=4, label=f"{_group_phrase(A)} (mean, 95%)")
    ax.set_xticks(x, [level_label(lv) for lv in levels]), ax.set_xlim(-0.3, x[-1] + 0.3), ax.axhline(0, color="#9a9a9a", lw=0.7)
    ax.set_ylabel("r with the ResNet"), ax.grid(axis="x", visible=False)


def _draw_rating_by_task(ax, A):
    """One panel: each rater's mean rating by what the second task showed on the trial -- the pilot's letters (the
    same letter on both images or not) beside the lab's task (how many times the dot changed colour, or how many
    of the two letters were the target)."""
    ticks, names = [], []
    letters = A["letter_trials"]
    if len(letters):
        means = letters.groupby(["participant", "same"], sort=False)["rating"].mean().unstack()
        means = np.column_stack([means[False], means[True]])  # different letters, the same letter
        ax.plot([0, 1.5], means.T, "-", color=C["letters"], lw=0.7, alpha=0.55)
        ax.plot([0, 1.5], means.mean(axis=0), "-o", color=INK, lw=2, ms=4.5, zorder=4)
        ax.plot([], [], "-", color=C["letters"], lw=0.9, label="one Letters pilot rater")
        ticks, names = [0, 1.5], ["different\nletters", "same\nletter"]
    by = A["rating_by_changes"]
    x = 3.5 + 1.5 * np.arange(by.shape[1])
    for p in by.index:
        ax.plot(x, by.loc[p], "-o", color=C["lab"], lw=0.8, ms=3.2, alpha=0.9, **_marker(A, p))
    ax.plot(x, by.mean(), "-o", color=INK, lw=2, ms=4.5, zorder=4, label="mean")
    ax.plot([], [], "-o", color=C["lab"], lw=0.8, ms=3.2, mfc=C["lab"], mec=C["lab"], label="one lab rater")
    load = A["task"].load_names  # "no change" ..., or "target on neither" ...: the last word on a line of its own
    ax.set_xticks([*ticks, *x], [*names, *("\n".join(load.get(k, str(k)).rsplit(" ", 1)) for k in by.columns)])
    ax.set_xlim(-0.5, x[-1] + 0.5), ax.set_ylabel("mean rating"), ax.grid(axis="x", visible=False)


# The one-page summary's panels: name -> a function drawing one panel on the axes it is given.
TAKEAWAY_PANELS = {"agreement": _draw_agreement_levels, "pairs": _draw_pair_kinds, "camps": _draw_camps, "shift": _draw_rating_by_task,
                   "dot": lambda ax, A: _draw_clean_course(ax, A, name_participants=False), "levels": _draw_levels}


def fig_rating_shifts(A):
    shifts = pd.concat([A["letter_shift"].to_frame().T, A["rating_shifts"]], ignore_index=True)
    shifts = shifts[shifts.participants > 0].reset_index(drop=True)
    H = 0.37 * len(shifts) + 1.05
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, H), gridspec_kw=dict(width_ratios=[2.25, 1.3], wspace=1.12, left=0.07, right=0.925,
                                                                          top=1 - 0.3 / H, bottom=0.72 / H))
    _draw_rating_by_task(a, A)
    a.set_title("a  Mean rating, by the second task")
    a.legend(loc="upper right", fontsize=6.5, handlelength=1.6)
    limit = 1.6  # the letters' range runs far past every other row's: it is cut here and its end written out
    for k, row in shifts.iterrows():
        y = len(shifts) - 1 - k
        colour = C["letters"] if row.contrast == lab.LETTER_SHIFT else C["lab"]
        b.plot([row.lowest, min(row.highest, limit)], [y, y], color=colour, lw=3.5, alpha=0.35, solid_capstyle="butt")
        if row.highest > limit:
            b.annotate(f"to {row.highest:+.1f}", (limit, y), xytext=(-2, 5), textcoords="offset points", ha="right", va="bottom", fontsize=6.5, color=MUTED)
        b.scatter([row.difference], [y], marker="D", s=34, color=INK, edgecolor="white", linewidth=0.7, zorder=4)
        b.text(1.04, y, "<0.001" if row.p < 0.0005 else f"{row.p:.3f}", transform=b.get_yaxis_transform(), va="center", ha="left", fontsize=7.2)
    b.text(1.04, len(shifts) - 0.5, "p", transform=b.get_yaxis_transform(), va="center", ha="left", fontsize=7.2, fontweight="bold")
    b.axvline(0, color="#9a9a9a", lw=0.8)
    b.set_yticks(range(len(shifts)), [textwrap.fill(c, 30) for c in reversed(shifts.contrast.tolist())])
    b.set_ylim(-0.6, len(shifts) - 0.4), b.set_xlim(-0.8, limit), b.grid(axis="y", visible=False)
    b.set_xlabel("first kind minus second,\nin rating points")
    b.set_title("b  Shifts in the rating")
    return fig


def fig_takeaways(A):
    titles = {"agreement": "a  How much two raters agree", "camps": "b  What each rater follows", "pairs": "c  Two raters' r, by practice",
              "shift": "d  Mean rating, by the second task", "dot": f"e  {CLEAN_TITLE[A['task'].name]}",
              "levels": "f  Ratings against the ResNet"}
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 7.6), gridspec_kw=dict(left=0.16, right=0.985, top=0.965, bottom=0.2, hspace=0.66, wspace=0.34))
    handles = {}
    for ax, name in zip(axes.flat, titles):
        TAKEAWAY_PANELS[name](ax, A)
        ax.set_title(titles[name])
        handles.update(zip(*reversed(ax.get_legend_handles_labels())))
    handles["the gate's minimum"] = plt.Line2D([], [], color=MUTED, lw=0.9, ls=(0, (4, 2)))
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3, fontsize=7.0, handlelength=2.2, columnspacing=1.6)
    return fig


FIGURES = {
    "1_examples": fig_examples, "2_agreement": fig_agreement, "3_yardstick": fig_yardstick, "4_raters": fig_raters,
    "5_model": fig_model, "6_projection": fig_projection, "7_trial_relations": fig_trial_relations,
    "8_conditions": fig_conditions, "9_response_times": fig_response_times, "10_participants": fig_participants,
    "11_practice": fig_practice, "12_groups": fig_groups, "13_takeaways": fig_takeaways, "14_rating_shifts": fig_rating_shifts,
}


def draw_all(A: dict, out_dir: Path, extension: str = "png") -> list[Path]:
    """Every figure of FIGURES into `out_dir`, never shown on screen."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context(STYLE):
        for name, draw in FIGURES.items():
            fig = draw(A)
            paths.append(out_dir / f"{name}.{extension}")
            fig.savefig(paths[-1], dpi=config.FIGURE_DPI)
            plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- summary

def consistent_relations(A: dict, floor: float = 0.1) -> pd.DataFrame:
    """Per-trial relations with the same sign in every participant and a mean |rho| of at least `floor`.

    A screen for what to look at next, not a test: nothing here is corrected for the number of relations tried.
    """
    g = A["trial_relations"].groupby(["y", "x"], sort=False).rho
    table = pd.DataFrame({"mean rho": g.mean(), "lowest": g.min(), "highest": g.max(), "participants": g.count()})
    same_sign = (np.sign(table["lowest"]) == np.sign(table["highest"])) & (table["participants"] == A["n"])
    return table[same_sign & (table["mean rho"].abs() >= floor)].sort_values("mean rho", key=np.abs, ascending=False)


def summary(A: dict, selection: str) -> str:
    def r(x):
        return f"{x:.2f}"

    n, levels, per, size = A["n"], A["levels"], A["per"], A["sizes"]["letters"]
    out = [f"# Lab ratings: {selection} (N = {n}, {len(A['batches'])} batch{'es' if len(A['batches']) > 1 else ''})", "",
           "batch                                    N   mean r between two raters"]
    out += [f"{b:<40} {A['batch_sizes'][b]:<3} {r(A['batch_rbar'][b]) if A['batch_sizes'][b] > 1 else 'one rater'}" for b in A["batches"]]

    out += ["", "## Agreement between raters",
            f"- mean r between two raters: {r(A['rbar']['lab'])} (95% over image pairs {r(A['rbar_ci'][0])} to {r(A['rbar_ci'][1])}; "
            f"on ranks {r(A['rbar_rank'])}). Letters pilot {r(A['rbar']['letters'])}, No-Letters pilot {r(A['rbar']['no_letters'])}."]
    out += [f"- groups of {n} from the {NAME[g]} that agree less: {100 * (A['sub'][g] < A['rbar']['lab']).mean():.0f}% of {len(A['sub'][g]):,}"
            for g in A["sub"]]
    out += [f"- R_LOO {r(A['loo']['lab'].mean())}, split-half {r(A['split']['lab'])} at N = {n} "
            f"(pilot at N = {size}: Letters {r(A['loo']['letters'].mean())} / {r(A['split']['letters'])}, "
            f"No-Letters {r(A['loo']['no_letters'].mean())} / {r(A['split']['no_letters'])})",
            f"- by formula at N = {size}: R_LOO {r(float(lab.expected_loo(A['rbar']['lab'], size)))}, "
            f"reliability of the mean {r(float(lab.spearman_brown(A['rbar']['lab'], size)))}",
            f"- the group's mean rating against the pilot's: r = {r(A['mean_vs_pilot']['no_letters'])} (No-Letters), "
            f"{r(A['mean_vs_pilot']['letters'])} (Letters); the two pilot groups against each other: {r(A['pilot_vs_pilot'])}"]

    out += ["", f"## The ResNet ({config.DEFAULT_LAYER}): mean r of a rater with the model",
            f"{'level':<14}{'lab':>6}  {'95% over pairs':<16}{'Letters':>8}{'No-Letters':>12}{'lab mean rating':>17}"]
    out += [f"{level_label(lv):<14}{r(per['lab'][lv].mean()):>6}  {r(A['level_ci'][lv]['lo']) + ' to ' + r(A['level_ci'][lv]['hi']):<16}"
            f"{r(per['letters'][lv].mean()):>8}{r(per['no_letters'][lv].mean()):>12}{r(A['avg_vector'][lv]):>17}" for lv in levels]
    out += [f"- {level_label(c.higher)} minus {level_label(c.lower)}: {c.difference:+.2f} ({c.lo:+.2f} to {c.hi:+.2f})"
            for c in A["contrasts"].itertuples()]

    practice, gated, minimum = A["practice"], A["gated"], config.LAB_PRACTICE_MINIMUM
    held = practice.groupby("participant").rating_criterion.any()
    out += ["", f"## Practice (attention / rating per iteration; a gated practice repeats below {minimum['attention']} attention, "
            f"and one marked * also below {minimum['rating']} rating)"]
    out += [f"{p:<5}{A['participants'].loc[p, 'serial']:<5}{('gated*' if held[p] else 'gated') if gated[p] else 'not gated':<11}"
            + "   ".join(f"{row.attention:.2f} / {row.rating:.2f}" for row in practice[practice.participant == p].itertuples())
            for p in A["pids"]]
    if "picked the altered image" in A["participants"]:
        t201 = A["participants"][["said noticed", "picked the altered image"]].dropna()
        out += ["", "## Trial 201", f"- said they noticed something: {int(t201['said noticed'].sum())} of {len(t201)}; picked the altered "
                f"image out of four: {int(t201['picked the altered image'].sum())} of {len(t201)}"]
    if len(A["group_contrasts"]):
        ga, n_with = A["group_agreement"], int(gated.sum())
        kinds = ga["pairs"].kind.value_counts()
        out += ["", f"## Gated practice ({n_with}) against the rest ({n - n_with}): exact p over every relabelling, uncorrected",
                "- mean r between two raters: " + ", ".join(f"{kind} {r(ga['mean'][kind])} ({kinds[kind]} pairs)" for kind in ga["mean"].index)
                + f"; both with against both without: p = {ga['p']:.2f}; same group against one of each: p = {ga['p_between']:.3f}",
                f"- mean r with the pairs' colour similarity: gated {r(A['colour']['lab'][gated].mean())}, the rest "
                f"{r(A['colour']['lab'][~gated].mean())}; Letters pilot {r(A['colour']['letters'].mean())}, No-Letters pilot "
                f"{r(A['colour']['no_letters'].mean())}",
                A["group_contrasts"].round(3).to_string(index=False)]

    shifts = pd.concat([A["letter_shift"].to_frame().T, A["rating_shifts"]], ignore_index=True)
    sequence, order = A["sequence"], A["order_agreement"]
    out += ["", "## Does anything but the pair move the rating? (first kind minus second, rating points; p by shuffling within "
                "each participant, uncorrected)",
            shifts.drop(columns=["lowest", "highest"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"),
            f"- mean rating by {LOAD_SENTENCE[A['task'].name][0]}: "
            + ", ".join(f"{k}: {v:.2f}" for k, v in A["rating_by_changes"].mean().items())
            + f"; mean r between two raters over {LOAD_SENTENCE[A['task'].name][1]}: {r(A['rbar_changed'])}",
            f"- a rating against the one before it: mean r {sequence['mean']:+.2f} ({sequence['per'].min():+.2f} to "
            f"{sequence['per'].max():+.2f} over participants), p = {sequence['p']:.2f}",
            f"- two raters on the pairs they saw in the same order minus in opposite orders: r {order['mean']:+.2f} "
            f"(positive for {order['positive']} of {order['pairs']} rater pairs), p = {order['p']:.2f}"]
    if len(A["participant_relations"]):
        out += ["", "## Across participants (Spearman rho; p by shuffling, uncorrected)",
                A["participant_relations"].to_string(index=False, float_format=lambda x: f"{x:.2f}")]

    hidden = ("trials", "rating sd", "batch", "serial", "age", "gender", "dominant_hand", "dominant_eye", "practice gated",
              "practice attention (first)", "practice rating (first)", f"{A['task'].correct} (block 1)", "r with consensus (block 1)",
              "said noticed", "noticed when told", "picked the altered image")
    consistent = consistent_relations(A)
    out += ["", "## Participants (every column: participants.csv)",
            A["participants"].drop(columns=[c for c in hidden if c in A["participants"]]).round(2).to_string(),
            "", "## Per-trial relations with one sign in every participant (a screen, uncorrected)",
            consistent.round(2).to_string() if len(consistent) else "none",
            "", "## Agreement inside a subset of trials minus outside it (r; p by shuffling the subset, uncorrected)",
            A["subset_contrasts"].round(3).to_string(index=False), ""]
    return "\n".join(out)


def run(batches: list[str] | None = None, task: str | None = None) -> dict:
    """The run type's entry point: compute, write the summary, tables and figures, print the summary.

    `task` (lab.TASKS) picks the participants who did that second task; it is needed only when the selected batches
    hold more than one, and then the results folder carries its name, so that one task's results never overwrite
    another's."""
    selection = "+".join(batches) if batches else "pool"
    runs = lab.find_runs(batches, task)
    if len(set(lab.run_tasks(batches).values())) > 1:
        selection += f"_{task}"
    out = config.LAB_RESULTS_DIR / selection
    out.mkdir(parents=True, exist_ok=True)
    A = compute(runs)
    A["participants"].to_csv(out / "participants.csv")
    A["rating_shifts"].to_csv(out / "rating_shifts.csv", index=False)
    A["letter_shift"].to_frame().T.to_csv(out / "rating_shifts.csv", index=False, header=False, mode="a")
    for name in ("trial_relations", "conditional", "subset_contrasts", "response_times", "practice", "blocks", "group_contrasts",
                 "participant_relations"):
        A[name].to_csv(out / f"{'conditional_agreement' if name == 'conditional' else name}.csv", index=False)
    text = summary(A, selection)
    (out / "summary.md").write_text(text)
    figures = draw_all(A, out / "diagrams")
    progress.write(text)
    progress.write(f"Done. {len(figures)} figures in {out / 'diagrams'} | tables and summary.md in {out}")
    return A
