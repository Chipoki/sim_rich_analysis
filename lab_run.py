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
  diagrams/*.png              one figure per function in FIGURES

compute() returns everything the figures and the summary show, as plain
pandas/numpy, so a write-up can be built from the same numbers (see
analysis/supplamentary/lab_first_look/).

Colours: the Letters / No-Letters pair plus one orange for the lab group
(validate_palette.js, light, all pairs: all checks pass). Colour marks the
group only; participants are told apart by labels, text stays in ink.
"""

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
MAX_LABELLED = 8  # above this many participants, figures stop naming each one
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
}
PARTICIPANT_X = {"changes caught": "colour changes caught", "median rating rt": "median rating RT (s)",
                 "median dot rt": "median dot RT (s)", "rated lowest": "share rated lowest"}


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
    A = dict(run_dirs=run_dirs, batches=sorted({d.parent.name for d in run_dirs}), pids=list(R.columns), n=n,
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
    A["lab_web_r"] = {lv: np.corrcoef(S[lv], w)[0, 1] for lv, w in web.items()}
    A["web_pixels"] = {lv: lab.per_rater_r(G["no_letters"], w).mean() for lv, w in web.items()}

    # -- what goes with what
    A["trial_relations"] = lab.trial_relations(T, refs)
    A["conditional"] = lab.conditional_agreement(T, refs)
    A["subset_contrasts"] = lab.subset_contrasts(T, refs)
    A["response_times"] = lab.response_times(T)
    A["participants"] = lab.participant_table(T, lab.load_participants(run_dirs), {
        "r with consensus": A["cons"]["lab"], "r with the others": A["loo"]["lab"],
        **{f"r with model ({level_label(lv)})": A["per"]["lab"][lv] for lv in levels},
    })

    # -- two worked pairs: the one the group rated highest, and the one it rated lowest (ties: first in pair order)
    means = R.mean(axis=1)
    web_to_local = {web_id: local for local, web_id in lab.local_to_web().items()}
    A["examples"] = [dict(tag=tag, pair_id=pid, files=[web_to_local[i] for i in pid.split("_")], ratings=R.loc[pid],
                          ranks={lv: int((S[lv] > S.loc[pid, lv]).sum()) + 1 for lv in levels})
                     for tag, pid in (("A", means.idxmax()), ("B", means.idxmin()))]
    return A


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
        if A["n"] <= MAX_LABELLED:
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
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.5), gridspec_kw=dict(width_ratios=[1, 1.12], wspace=0.42,
                                                                            left=0.07, right=0.985, top=0.86, bottom=0.17))
    pids, corr = A["pids"], A["corr"].to_numpy().copy()
    np.fill_diagonal(corr, np.nan)
    a.imshow(corr, cmap=ORANGES, vmin=0, vmax=0.45)
    if A["n"] <= MAX_LABELLED:
        for i in range(len(pids)):
            for j in range(len(pids)):
                same = i == j
                a.text(j, i, "same\nrater" if same else f"{corr[i, j]:.2f}".replace("0.", ".").replace("-", "−"), ha="center",
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
    for g, dx in (("letters", -0.012), ("no_letters", 0.0)):
        where = np.median(A["sub"][g]) if g in A["sub"] else A["rbar"][g]
        ax.annotate(NAME[g], (where + dx, top * 1.02), ha="right" if g == "letters" else "center", va="bottom", fontsize=8)
    ax.annotate(_group_phrase(A), (A["rbar"]["lab"] + 0.012, top * 1.02), ha="left", va="bottom", fontsize=8, fontweight="bold")
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
    outer = fig.add_gridspec(2, 1, height_ratios=[1.55 * rows, 1.55], hspace=0.72 / max(rows, 1) + 0.25,
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
        ax.set_xticks(sorted({A["n"], 10, A["sizes"]["letters"], 30, 40}))
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
    titles = {"block": "block", "colour changes": "colour changes on the trial", "dot task": "dot task on the trial",
              "last change in": "last colour change fell in", "rating speed": "speed of the rating (own thirds)"}
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
    order = {"own rating": None, "block": None, "last change in": ["none", *lab.PHASE_NAMES]}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.45), sharey=True, gridspec_kw=dict(wspace=0.08, left=0.085, right=0.985, top=0.88, bottom=0.2))
    for ax, (by, levels), letter in zip(axes, order.items(), "abc"):
        cells = rts[(rts.by == by) & (rts.n >= lab.MIN_CELL_TRIALS)]  # a median of a few trials is not worth a point
        wide = cells.pivot(index="level", columns="participant", values="median")
        wide = wide.reindex(levels or sorted(wide.index)).dropna(how="all")
        x = np.arange(len(wide))
        ax.plot(x, wide.to_numpy(), "-o", color=C["lab"], lw=1.0, ms=3.5, mec="white", mew=0.6, alpha=0.9)
        ax.plot(x, _mean_line(wide), "-o", color=INK, lw=2, ms=5, zorder=4)
        ax.set_xticks(x, [str(v) for v in wide.index]), ax.set_xlim(-0.4, len(wide) - 0.6), ax.grid(axis="x", visible=False)
        ax.set_xlabel({"own rating": "the rating given", "block": "block", "last change in": "last colour change fell in"}[by])
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
    fig, axes = plt.subplots(len(ys), len(PARTICIPANT_X), figsize=(WIDTH, 3.55 if labels else 3.25), sharey="row", sharex="col",
                             gridspec_kw=dict(wspace=0.1, hspace=0.14, left=0.1, right=0.985, top=0.95, bottom=0.27 if labels else 0.15))
    for r, y in enumerate(ys):
        for c, (x, xlabel) in enumerate(PARTICIPANT_X.items()):
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


FIGURES = {
    "1_examples": fig_examples, "2_agreement": fig_agreement, "3_yardstick": fig_yardstick, "4_raters": fig_raters,
    "5_model": fig_model, "6_projection": fig_projection, "7_trial_relations": fig_trial_relations,
    "8_conditions": fig_conditions, "9_response_times": fig_response_times, "10_participants": fig_participants,
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

    hidden = ("trials", "rating sd", "batch", "age", "gender", "dominant_hand", "dominant_eye")
    consistent = consistent_relations(A)
    out += ["", "## Participants (every column: participants.csv)",
            A["participants"].drop(columns=[c for c in hidden if c in A["participants"]]).round(2).to_string(),
            "", "## Per-trial relations with one sign in every participant (a screen, uncorrected)",
            consistent.round(2).to_string() if len(consistent) else "none",
            "", "## Agreement inside a subset of trials minus outside it (r; p by shuffling the subset, uncorrected)",
            A["subset_contrasts"].round(3).to_string(index=False), ""]
    return "\n".join(out)


def run(batches: list[str] | None = None) -> dict:
    """The run type's entry point: compute, write the summary, tables and figures, print the summary."""
    selection = "+".join(batches) if batches else "pool"
    out = config.LAB_RESULTS_DIR / selection
    out.mkdir(parents=True, exist_ok=True)
    A = compute(lab.find_runs(batches))
    A["participants"].to_csv(out / "participants.csv")
    for name in ("trial_relations", "conditional", "subset_contrasts", "response_times"):
        A[name].to_csv(out / f"{'conditional_agreement' if name == 'conditional' else name}.csv", index=False)
    text = summary(A, selection)
    (out / "summary.md").write_text(text)
    figures = draw_all(A, out / "diagrams")
    progress.write(text)
    progress.write(f"Done. {len(figures)} figures in {out / 'diagrams'} | tables and summary.md in {out}")
    return A
