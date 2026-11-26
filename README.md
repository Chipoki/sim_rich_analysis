# Peripheral Similarity — Web Replication Analysis

Replicates the degradation-level / layer / AVG-vector / desaturation analyses
from `SimRich Pres 29.12.25` on the new web-experiment data (Letters &
No-Letters conditions), CPU-only.

## 0. Expected directory layout

This project assumes it sits **next to** your raw-data folder and writes
everything to a **third**, sibling folder:

```
<some parent folder>/
├── src_data/                      # your raw data (edit config.py if named differently)
│   ├── All20Letters.txt
│   ├── All20NoLetters.txt
│   ├── resnet50_places365.pth.tar
│   ├── stimuli_full_fld/
│   ├── NavehVer_05Stim/
│   ├── NavehVer_3Stim/
│   ├── NavehVer_Tunnel_WhiteBG_JPG/   (NavehVer_Tunnel_PNG also available)
│   ├── Naveh_Desaturated/
│   └── Naveh_BigFoveaDesaturated/
├── web_analysis/                  # <- this project
│   └── ... (this code)
└── web_analysis_results/          # created automatically by run.py
    ├── data/
    │   ├── sim_vectors.pkl
    │   ├── dist_vectors.pkl            # primary backend (config.BACKEND)
    │   ├── dist_vectors_dinov2.pkl     # one per config.EXTRA_BACKENDS entry
    │   ├── analysis_results.pkl
    │   └── embeddings/
    │       ├── resnet50/<level>_<layer>.pkl   # avgpool / layer4 / classifier
    │       └── dinov2/<level>_<layer>.pkl     # pooled "dinov2" + grid-search blockNN_{cls,mean} configs
    └── diagrams/
        ├── resnet50/*/*.png            # ResNet50-Places365 figures (config.BACKEND)
        ├── dinov2/*/*.png              # same figure set, DinoV2 embeddings
        ├── integrated/*.png            # cross-model comparisons (e.g. resnet50 vs dinov2)
        └── experimental_blocks/letters_accuracy_by_block.png  # purely behavioral, generated once
```

Every backend (config.BACKEND and each of config.EXTRA_BACKENDS) gets its own
`data/embeddings/<name>/` and `diagrams/<name>/` sub-tree, named via
`config.BACKEND_DIR_NAMES` (`resnet50_places365` -> `resnet50`,
`dinov2` -> `dinov2`) -- no backend is special-cased to an unprefixed path.

If your raw-data folder isn't literally named `src_data`, update
`SRC_DATA_DIR` at the top of `config.py` — that's the only path you should
ever need to touch.

## 1. Environment setup (CPU-only, Ubuntu 24.04 / i7-13700H / 16 GB RAM)

`environment.yml` pulls `pytorch-cpu` from `conda-forge` — a build of
PyTorch with no CUDA dependency at all, so there's nothing GPU-related to
misconfigure and no multi-GB CUDA toolkit to download.

```bash
conda env create -f environment.yml
conda activate sim_rich_analysis_env
```

**Watching the live solve/install progress** (instead of a static
"Solving environment…"):

- `conda env create -f environment.yml -v` — verbose mode; shows each
  package as conda's solver considers and downloads it. Use `-vv` or
  `-vvv` for even more detail.
- `conda config --set verbosity 3` — sets max verbosity globally for all
  future conda commands, so you don't need `-v` each time.
- The libmamba solver (conda's default since ~2023) is already
  responsible for most of the speed-up over the old solver; if you're on
  an older conda, install it explicitly first:
  `conda install -n base conda-libmamba-solver && conda config --set solver libmamba`.
- Fastest option: use **mamba** instead of conda outright —
  `mamba env create -f environment.yml` — it shows a live, per-package
  progress bar by default and resolves the environment much faster.

## 2. Run the pipeline

```bash
python run.py
```

First run computes and caches embeddings for every stimulus level/layer
(the slow part — CPU ResNet50 forward passes over ~400 images × 6 levels);
every subsequent run reads straight from
`web_analysis_results/data/embeddings/<backend name>/` (e.g. `resnet50/`,
`dinov2/` — see `config.BACKEND_DIR_NAMES`) and is fast. Delete a specific
`<level>_<layer>.pkl` under the relevant backend's sub-dir to force
recomputation of just that one.

Each analysis figure is saved (not displayed interactively) to
`web_analysis_results/diagrams/`, so a full run completes unattended without
popping up GUI windows.

### 2.1 The lab run type

```bash
python run.py --lab                                       # every batch, pooled
python run.py --lab --batch participants_1_to_4_221126    # only the named batch folder(s)
python run.py --lab --task letters-count                 # only the participants who did that second task
python run.py --lab --without S3                          # also scores the group without the named participant(s)
```

Participants who did different second tasks are not one group. A run's task is read from its first main trial
(`lab.run_task`): `colour-change` (the dot; the nine participants so far) or `letters-count` (the letters, asked
how many of a pair's two were the block's target letter; prepared 2026-11-25, no run yet). A selection that holds
both is refused until `--task` names one, and its results then go to `<selection>_<task>/`. What differs between
the two is in one place, `lab.TASKS`: the per-trial variables, the conditions, the rating shifts (for the letters,
first of all the pilot's own bias, the same letter on both images against different letters) and the task's
per-participant measures.

`--without S<k> ...` asks "how much of the group's numbers is this participant?". The whole group is scored as
always, and the same computation is run again for the others (`lab_run.compute`'s `A["rest"]`): every figure
that shows a number of the whole group draws that number without them beside it, dashed, the summary sets the
two side by side, and the results go to `<selection>_without-<ids>/`, so the selection's own are not
overwritten. The tables (`*.csv`) stay the whole group's. It is a what-if, not an exclusion rule.

Scores the lab experiment's participants the way the web pilot was scored,
with the pilot as yardstick (so run the default `python run.py` once first:
it reads that run's caches). The lab shows the pilot's 200 fixed pairs, so
lab ratings are keyed by the pilot's own `pair_id`s. Re-run it after every
new batch: the model side is cached, and a run takes seconds.

Inputs, all named in `config.py`'s lab section and all outside this repo:

- `LAB_RUNS_DIR`: one sub-folder per batch, each holding its participants'
  run folders (`P<serial>_..._complete`; test runs carry a suffix and are
  left out). Adding a batch is copying its folder there. The lab PC's serial
  starts over between batches, so participants are numbered `S1`, `S2`, ...
  over every batch, in the order they ran (`lab.participant_ids`): a
  participant keeps that id whichever batches a run selects.
- `LAB_LEVEL_DIRS`: the lab's own pixels per level (Full, Scramble 3.0,
  Tunnel; add Scramble 0.5 there once it is rendered and the next run embeds
  it). The lab's degraded images use a wider aperture than the web's, so its
  degraded-level numbers are not comparable with the web run's.
- `LAB_PARTICIPANT_LABELS` (optional): a CSV the experimenter keeps, one row
  per participant, named by its `batch` folder and `serial` (`P<serial>`),
  then label columns such as `strategy`.
- `LAB_PRACTICE_MINIMUM`, `LAB_PRACTICE_PAIRS_DIR`,
  `LAB_PRACTICE_ITERATION_PREFIX`: the experiment's practice minimums, the
  folder holding the practice pairs named similar / different, and the name
  of the folder each practice iteration shows its own pairs from.

It writes `lab_analysis_results/<selection>/` (`pool`, or the batch names
joined by `+`): `summary.md` (also printed), `participants.csv`,
`trial_relations.csv`, `conditional_agreement.csv`, `subset_contrasts.csv`,
`response_times.csv`, `practice.csv`, `blocks.csv`, `group_contrasts.csv`,
`rating_shifts.csv`, `participant_relations.csv` and `diagrams/*.png`
(figure 13 gathers the panels of a one-page summary, figure 15 is the noise ceiling). Caches shared by every selection sit in
`lab_analysis_results/data/`.

What it reports:

- **Agreement between raters** as the mean r between two raters, which does
  not depend on group size (R_LOO and split-half do, so they cannot be
  compared between a growing lab pool and the pilot's 19), against every
  same-sized group of raters the pilot holds, and carried to any group size
  by two formulas.
- **The model at each level**, per rater, with intervals from resampling the
  image pairs (raters held fixed), and the pilot groups on the same vectors.
- **The noise ceiling**, lower and upper (`lab.noise_ceiling`): a rater
  against the mean of the others (R_LOO) and against the mean of all, the
  rater included. Both depend on the group's size, so the pilot's are scored
  again in groups of the lab group's size (`lab.subgroup_ceilings`, on the
  very subgroups the agreement is scored on), and the model's mean r is given
  as a share of the lower ceiling of the raters it was scored on, with an
  interval over image pairs for the lab. A noisy group's lower ceiling is
  low, which makes its share high: compare shares between same-sized groups
  only, and read them beside the upper ceiling. `lab.expected_upper` is the
  upper ceiling's formula, next to `lab.expected_loo`.
- **Relations**, a fixed set: per trial, the rating and its response time
  against trial number, the pilot's ratings of the pair, the model's
  similarity at each level and the colour-change task; agreement recomputed
  within blocks, numbers of colour changes, dot-task outcomes and rating
  speeds, with a shuffle test of two of those splits (`lab.SUBSET_CONTRASTS`:
  trials without a colour change, and the fastest third of ratings, each
  against the rest); and per participant, agreement against dot-task performance,
  response times, scale use and the experimenter's labels. Nothing there is
  corrected for the number of relations tried: it is a screen for what to
  look at, not a set of tests. A participant whose ratings do not vary
  inside a subset (one who gives nearly every pair the same rating) is left
  out of that subset's contrast, and the table counts who entered.
- **Practice and its gate.** Each practice iteration's two scores
  (`lab.practice_scores`): attention, as the experiment computes it, and the
  rating score the runs of 2026-11-23 were also held to, now only reported
  and only where the iteration shows the named pairs (the first, since
  2026-11-25, when every iteration got pairs of its own:
  `lab.practice_iterations`). Whether the run's practice was *gated*,
  scored and repeated while a score was short, and whether the rating was
  one of its criteria (`rating_criterion`), are read from the run's log.
  Gated participants are then set against the
  rest: every participant-level measure with an exact p over all ways to
  relabel the participants (`lab.group_contrasts`), and agreement between
  two raters by whether both, neither or one of them is gated
  (`lab.group_agreement`). Uncorrected, and a comparison of versions of the
  experiment, not a randomized one.
- **Colour.** Each rater against the pairs' colour similarity (the
  correlation of the two images' RGB histograms, `lab.colour_similarity`): a
  crude check of the "I went by colour" strategy participants report.
- **Trial 201**: who said they noticed something and who picked the altered
  image. Typed answers and the experimenter's notes are never read in.
- **What moves the rating besides the pair** (`lab.rating_shifts`). The
  pilot's letters leaked into the rating: a pair was rated about a point
  higher when its two images carried the same letter. The run asks the same
  of the dot (no colour change against one or two; the dot the same colour
  on both images; a change during an image; a missed change; a stray press;
  the dot's first colour; the order of the two images), as a difference in
  mean rating with a within-participant shuffle p, and scores the pilot's
  letter effect with the same function as the yardstick
  (`lab.pilot_letter_shift`, read from `RESULT_FILES["letters"]`). A rank
  correlation with the number of changes misses this: the pattern is not a
  trend. Also whether a rating leans on the one before it
  (`lab.sequential_dependence`), whether two raters agree more on pairs
  they saw in the same order (`lab.order_agreement`), and a short list of
  relations across participants (`lab.participant_relations`).

## 3. Design notes

- **Parsing** (`parsing.py`): verified directly against both real result
  files. A logical trial is 2–3 consecutive JSON objects sharing a
  `pair_id`; `trial_index` is per-phase and is never used for joining. All
  40 participants (both conditions) rate the *same* 200 pairs — human and
  model vectors are aligned by `pair_id`, not trial order or position.
- **Embeddings** (`embeddings.py`): `ResNetPlacesExtractor` is the default
  backend, using your local `resnet50_places365.pth.tar`. It exposes the 3
  layers examined in the original analysis:
  - `avgpool` (default) — 2048-d global-average-pooled features
  - `layer4` — the flattened last conv block (100,352-d) — this is the
    *un-pooled* spatial map, distinct from `avgpool`
  - `classifier` — 365-d Places365 logits

  These interpretations of "AvgPool / Layer_4 / Classifier" are my best
  reading of the 3-layer setup referenced in the PDF. Your raw-data
  folders already contain precomputed `<Layer>_embeddings.csv` files under
  each stimulus folder's `ResNet_places/` — if you want to sanity-check
  dimensionality (e.g. confirm `Layer_4` should really be flattened rather
  than pooled), diff a row of those CSVs against `embeddings.py`'s output
  and adjust the layer definitions if they don't match. Note a few of
  those CSVs are incomplete on disk (e.g. `Naveh_BigFoveaDesaturated` only
  has `*_pairs.csv`, no `*_embeddings.csv`) — this project never reads
  them, it recomputes embeddings fresh, so that's fine either way.

  `DinoV2Extractor` is implemented behind the same interface (per the
  original prompt's aspirational ask) but is **not** the default backend;
  it needs internet access on first run to fetch weights via `torch.hub`
  from the public, unauthenticated `dl.fbaipublicfiles.com` CDN.

  `config.EXTRA_BACKENDS` (default `["dinov2"]`) additionally runs the
  *entire* per-condition figure set (degradation levels, best-layer, AVG
  vector, desaturation, noise ceiling, blocks, cross-condition comparisons)
  a second time using DinoV2 embeddings instead of ResNet-Places50's,
  alongside (not instead of) the primary backend's figures -- see
  `run.py`'s module docstring. Every backend, `config.BACKEND` included,
  gets its own `diagrams/<name>/` sub-tree (`config.BACKEND_DIR_NAMES`,
  e.g. `resnet50/`, `dinov2/`) -- no backend is special-cased to an
  unprefixed path. Set `config.EXTRA_BACKENDS = []` to disable and only run
  `config.BACKEND`.

  Each backend's own `degradation/` sub-tree also gets a
  `letters_vs_no_letters_ratio_<layer>.png` line plot per examined layer,
  alongside the existing `letters_vs_no_letters_<layer>.png` bar figure --
  the *ratio* of mean correlation between adjacent degradation levels
  (`Full:Deg05`, `Deg05:Deg3`, `Deg3:Tunnel`), one line per condition, rather
  than each level's absolute value -- see `plotting.ratio_line_plot()`.

  Cross-*model* comparisons (as opposed to the above, which is cross-
  *condition* within one model) live under `diagrams/integrated/` instead of
  any single backend's sub-tree, since they need more than one backend's
  results at once -- currently just each model's best-layer correlation per
  degradation level, `config.BACKEND` vs each of `config.EXTRA_BACKENDS`, one
  figure per condition; see `run._run_integrated_figures()`.

  **Which DinoV2 configs the pipeline reports** is resolved by
  `embeddings.dinov2_layers()`, in priority order:

  1. **`config.DINOV2_PINNED_CONFIGS` (the shipped default).** A fixed,
     a-priori set: three depths spaced evenly across the 12-block ViT-S
     (blocks 4 / 8 / 12), each read out both ways DinoV2 is normally used --
     CLS token and mean-pooled patch tokens. Depth and pooling are crossed so
     the two factors stay separable in the figures.
     `config.DINOV2_PINNED_DEFAULT_CONFIG` (`block12_cls`, i.e. what
     `DinoV2Extractor` returns on its own) names the single config that
     stands in for DinoV2 wherever one representative layer is plotted --
     **not** the first entry, which is the shallowest block.
  2. **The grid search**, if pinning is cleared to `()`. See below.
  3. The single pooled `SINGLE_LAYER_BACKEND_LABELS`/`_COLORS` fallback.

  *Why pinned by default:* `supplemental/dinov2_layer_grid_search/` is a
  standalone script (not part of `run.py`) that scores all 12 blocks x
  {cls, mean} against the human similarity judgments and ranks them by
  `avg_vector_r`. Letting it choose means the reported configs are selected
  on the same ratings the pipeline then reports, so their r is optimistically
  biased. Pinning removes that selection step, putting DinoV2 on the same
  footing as ResNet-Places50, whose `config.LAYERS` are likewise fixed and
  never data-selected. The grid search remains available as exploratory
  analysis -- it writes its own outputs and no longer drives `run.py`.

  ⚠ *One asymmetry to keep in mind when reading the best-layer figures:* the
  pinned set has 6 configs against ResNet-Places50's 3 examined layers, and
  `best_layer_per_participant` takes a per-participant max, so DinoV2 draws
  its maximum from a larger pool. That inflates DinoV2's best-layer bars
  slightly relative to ResNet's. It is a far smaller effect than the
  grid-search selection it replaced (best-of-6-fixed vs best-of-3, rather
  than top-3-of-24-fitted), but it is not zero -- the per-config figures,
  which involve no maximum, are the clean comparison.

- **Distances** (`distances.py`): cosine distance throughout. These CNN
  layers are ReLU-activated (non-negative), so cosine distance stays in
  `[0, 1]` and `similarity = 1 - distance` is a valid bounded similarity.
- **Exclusions**: the two conditions exclude at **different stages**, on
  purpose.
  - *Letters* (`analysis.py` / `config.py`): `letter_discrimination_correct`
    accuracy < 80%, computed automatically from the parsed data — verified to
    match the PDF's N=19 exactly (one participant at 66.5% accuracy). Excluded
    **downstream**, so the participant stays in the tidy DataFrame: one figure
    deliberately plots them to justify the threshold.
  - *No-Letters* (`config.py` / `parsing.py`): the PDF's "claimed technical
    problems" exclusion is a self-report with **no corresponding field in the
    JSON**, but the participant was recovered from the raw file's presentation
    telemetry — that participant is the only one of the 20 with any
    missing frame-timing block (165 of 212 trials logged none, vs 212/212 for
    everyone else). Independently corroborated behaviorally: their leave-one-out
    reliability is **0.06** against a next-lowest of **0.57**, a gap wider than
    the entire spread of the remaining 19. Since this names a specific id rather
    than scoring a rule, it is applied at **acquisition**
    (`parsing.parse_file`) — the participant never enters the tidy DataFrame,
    so no derived figure or cached artifact can reintroduce them. The id itself
    is a Prolific participant identifier and is **not in this repository**: copy
    `local_config.example.py` to `local_config.py` (untracked) and fill it in,
    or leave `NO_LETTERS_MANUAL_EXCLUDE_PARTICIPANT_ID = None` to analyze all
    N=20. `config.py` warns at import if the local file is missing, since the
    silent fallback would otherwise reproduce the N=20 numbers.
- **Reliability**: `leave_one_out_reliability()` doubles as both the
  PDF's headlined R_LOO and the noise ceiling plotted alongside model
  bars. `split_half_reliability()` is Spearman-Brown corrected over
  `config.SPLIT_HALF_N_ITER` (1000) random participant-split iterations.
  Both were validated against the PDF's headline numbers using the real
  rating data: computed R_LOO ≈ 0.39 (Letters) / **0.76** (No-Letters) vs.
  the PDF's ≈0.44 / ≈0.78 — in the right ballpark and same direction
  (No-Letters much more reliable). Resolving the No-Letters exclusion above
  moved that figure from 0.73 (N=20) to 0.76 (N=19), i.e. most of the way to
  the PDF's ≈0.78; the residual gap is attributable to this being a different
  (web) data collection than the original PDF's. Every "vs noise ceiling"
  figure carries a symbolic footnote
  naming both r values' source (LOO / split-half + its iteration count),
  so the methodology behind the plotted numbers is never left implicit.
- **CPU / memory**: batching is small (`config.BATCH_SIZE = 8`) and
  images are resized to 224×224 before the model ever sees them, so peak
  memory during embedding extraction stays well under 16 GB. Embeddings
  are cached to disk per (level, layer) so nothing is ever recomputed
  across runs.

## 4. Files

| File | Responsibility |
|---|---|
| `config.py` | paths, folder maps, layer names, exclusions, all constants |
| `parsing.py` | raw JSON → tidy long-format DataFrame |
| `sim_vectors.py` | tidy DataFrame → per-participant human similarity vectors |
| `embeddings.py` | CPU image embedding extraction + disk caching (ResNet-Places50 / DINOv2) |
| `distances.py` | embeddings → per-pair cosine distance/similarity vectors |
| `analysis.py` | correlations, best-layer selection, AVG-vector R, split-half, LOO/noise ceiling, paired stats |
| `plotting.py` | bar + per-participant-dot figures |
| `progress.py` | pinned multi-line tqdm progress dashboard shared by every stage/sub-stage |
| `run.py` | thin end-to-end orchestration |
| `lab.py` | the lab experiment's runs → ratings/trials/participants tables, its model vectors, size-free agreement statistics, relations |
| `lab_run.py` | the lab run type (`python run.py --lab`): numbers, figures, summary |
| `tests/` | pytest suite (synthetic fixtures; one opt-in slow real-data test) — see §5 |

## 5. Tests

```bash
pytest              # fast suite: synthetic fixtures only, offline, ~seconds
pytest -m slow       # + the real end-to-end run against real src_data/weights (opt-in, slow)
```

Every module has a matching `tests/test_<module>.py`. `tests/conftest.py` provides the shared synthetic
fixtures (a tiny trial-JSON factory matching the real jsPsych shape, tiny generated images, and a
`StubExtractor` that fakes `EmbeddingExtractor`'s model so caching/batching logic is tested without real
torch compute). `tests/test_run.py` is a full-pipeline smoke test on a small synthetic dataset.
`tests/test_lab.py` covers the lab run type the same way (synthetic run folders, tiny images, the stub
extractor, a made-up pilot cache), ending in its own end-to-end smoke test.
`tests/test_slow_e2e.py` runs the real, unmocked pipeline and is skipped whenever real `src_data`/weights
aren't present; `pytest.ini` also deselects it by default so plain `pytest` never triggers it.
