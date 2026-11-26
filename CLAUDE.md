# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CPU-only analysis pipeline that replicates the degradation-level / layer / AVG-vector / desaturation
analyses from a peripheral-vision similarity-judgment study (`SimRich Pres 29.12.25`) on new web-experiment
data (Letters & No-Letters conditions). It correlates human pairwise similarity ratings against CNN
(ResNet50-Places365) embedding distances for images at various degradation levels.

## Setup and running

```bash
conda env create -f environment.yml
conda activate sim_rich_analysis_env
python run.py
```

There is no linter or build step in this repo — it's a single-purpose analysis pipeline run end-to-end
via `python run.py`. There is a `tests/` pytest suite (see below); there is no other build step.

`python run.py --lab [--batch NAME ...] [--without S3 ...]` is a second run type: the lab experiment's participants
(README §2.1, `lab.py` / `lab_run.py` below). It reads the default run's caches, so the default run comes first.

### Tests

```bash
pytest              # fast suite: synthetic fixtures only, offline, ~seconds
pytest -m slow       # + the real end-to-end run against real src_data/weights (opt-in, slow)
```

`pytest.ini` deselects the `slow` marker by default (`addopts = -m "not slow"`), so plain `pytest` never
triggers a real CPU ResNet50 run. `tests/test_slow_e2e.py`'s single slow test is additionally
`skipif`-gated on `config.SRC_DATA_DIR` / `RESNET_WEIGHTS_PATH` existing, so it's a no-op wherever the
real raw data isn't present. Every other test file mirrors a pipeline module 1:1 (`test_parsing.py`,
`test_embeddings.py`, ...) and uses only synthetic fixtures from `tests/conftest.py` — a tiny generated
trial-JSON factory, tiny generated images, and a `StubExtractor` that fakes `EmbeddingExtractor`'s
model/forward pass so caching/batching logic is exercised without real torch compute.

### Expected directory layout

This project must sit as a **sibling** of a `src_data/` folder (raw data: result JSON files, stimulus image
folders, `resnet50_places365.pth.tar`) and writes all outputs to a third sibling folder, `web_analysis_results/`
(auto-created). If the raw-data folder has a different name, change `SRC_DATA_DIR` in `config.py` — that's
the only path that should ever need editing. Full layout is documented in README.md §0.

### Caching behavior

First run is slow (CPU ResNet50 forward passes over ~400 images × 6 levels); embeddings are cached to
`web_analysis_results/data/embeddings/<backend name>/<level>_<layer>.pkl` (one sub-dir per backend, e.g.
`resnet50/`, `dinov2/` — see `config.BACKEND_DIR_NAMES`) and every subsequent run reads from cache.
Delete a specific `<level>_<layer>.pkl` under the relevant backend's sub-dir to force recomputation of
just that one. `sim_vectors.pkl`, `dist_vectors.pkl`, and `analysis_results.pkl` cache the other pipeline
stages similarly.

## Architecture

The pipeline is a strict linear stage sequence, each stage consuming the previous stage's output and each
stage's module owning its own disk cache (pickle files under `web_analysis_results/data/`). `config.py` is
the single source of truth for every path, folder-name mapping, layer name, and exclusion constant — no
other module hardcodes these.

```
parsing.py  →  sim_vectors.py         →  analysis.py  →  plotting.py
              ↗
embeddings.py → distances.py         ↗
```

- **`parsing.py`** — raw JSON (one line per participant, a JSON array of trial objects) → tidy long-format
  DataFrame. A logical trial is 2–3 consecutive JSON objects (`psychophysics`, optional `letter_question`,
  `similarity_question`) sharing a `pair_id`. **`trial_index` is per-phase and must never be used as a join
  key** — `pair_id` is the only safe key, and it's stable across all 40 participants in both conditions
  (everyone rates the same 200 pairs, just in randomized order).

- **`sim_vectors.py`** — pivots the tidy DataFrame into a wide table: index = `pair_id` (sorted, canonical
  order used by every vector downstream), columns = `(condition, participant_id)` MultiIndex, values =
  1–6 similarity rating. This alignment-by-`pair_id` (not trial order or position) is what makes human
  vectors comparable to each other and to model vectors.

- **`embeddings.py`** — CPU image embedding extraction behind an `EmbeddingExtractor` ABC with pluggable
  backends (`ResNetPlacesExtractor` default, `DinoV2Extractor` alternate — set via `config.BACKEND`).
  `ResNetPlacesExtractor` exposes 3 layers per the original analysis: `avgpool` (2048-d, default),
  `layer4` (100,352-d flattened, un-pooled), `classifier` (365-d Places365 logits) — extracted in one
  forward pass via a forward hook on `layer4`. Handles its own batching (`config.BATCH_SIZE`) and per-
  `(level, layer)` disk caching; the model is lazily loaded so instantiating an extractor never touches torch.

  **Which DinoV2 configs are active is decided by `embeddings.dinov2_layers()` — the single chokepoint**
  (`run.py`'s three `_*_for_backend` helpers and `get_extractor` all route through it). Priority:
  `config.DINOV2_PINNED_CONFIGS` (an a-priori set — blocks 4/8/12 × {cls, mean} — shipped as the default),
  else the supplemental grid search's ranking, else the single pooled embedding. Pinning exists to keep the
  reported configs from being selected on the very ratings the pipeline reports; don't reintroduce
  grid-driven selection into `run.py` without saying so in the write-up. Two traps: (1)
  `_default_layer_for_backend` must return `config.DINOV2_PINNED_DEFAULT_CONFIG` when pinned, **not**
  `next(iter(...))` — the pinned tuple is depth-ordered, so its first entry is the shallowest block;
  (2) `tests/conftest.py`'s `StubExtractor.LAYER_DIMS` derives from `DINOV2_PINNED_CONFIGS`, so adding a
  pinned config that the stub can't produce breaks the end-to-end smoke test rather than any unit test.

- **`distances.py`** — per-image embeddings → per-pair cosine distance/similarity vectors, aligned 1:1
  with the human vectors by `pair_id`. Cosine distance is used throughout because these layers are
  ReLU-activated (non-negative), keeping distance in `[0, 1]` so `similarity = 1 - distance` is a valid
  bounded similarity.

- **`analysis.py`** — per-participant Pearson correlations (human vs. model vector), best-layer-per-
  participant selection, AVG-vector (group-mean) correlation, split-half reliability (Spearman-Brown
  corrected), leave-one-out reliability (doubles as both the headline `R_LOO` stat and the noise ceiling
  plotted alongside model bars), and paired Wilcoxon tests between adjacent degradation levels.
  `get_excluded_participants()` computes the Letters exclusion automatically from
  `letter_discrimination_correct` accuracy (< `LETTER_ACCURACY_THRESHOLD`), applied **downstream** so that
  participant still reaches the tidy DataFrame — one figure plots them deliberately, to justify the
  threshold. The No-Letters exclusion instead names a single id
  (`config.NO_LETTERS_MANUAL_EXCLUDE_PARTICIPANT_ID`) and is applied at **acquisition**, in
  `parsing.excluded_participant_ids()`. So for No-Letters this function is a *reporting* no-op: the id is
  already gone from the data, and it stays in the returned set only so `run.py`'s `"2. Exclusions: ..."`
  line still names who was dropped. Don't "fix" that redundancy by removing it.

- **`plotting.py`** — bar + per-participant-dot figures (`bar_with_dots`, `bar_with_noise_ceiling`), cross-
  participant scatter plots (`scatter_relationship`), and a condition line plot tracking a ratio metric
  across level transitions (`ratio_line_plot`); every figure is saved to `web_analysis_results/diagrams/`
  (never displayed interactively -- no `plt.show()` -- so a full run completes unattended without popping
  up GUI windows).

- **`run.py`** — thin end-to-end orchestration calling the above in order, for both conditions
  (`letters`, `no_letters`), producing one figure per analysis in spec §4. Every backend (`config.BACKEND`
  and each of `config.EXTRA_BACKENDS`) gets its own `diagrams/<name>/` sub-tree (`config.BACKEND_DIR_NAMES`);
  cross-model comparisons that need more than one backend's results at once (e.g. ResNet50-Places365 vs
  DinoV2's best-layer correlation) live under `diagrams/integrated/` instead — see
  `run._run_integrated_figures()`.

- **`lab.py`**, **`lab_run.py`** — the lab run type, beside the stage sequence above rather than in it. The lab
  experiment shows the web pilot's 200 fixed pairs, so `lab.load_runs` turns run folders (one per participant, in
  batch folders under `config.LAB_RUNS_DIR`) into ratings keyed by the pilot's `pair_id`s — matched as unordered
  pairs, since the lab randomizes which image comes first — and `analysis.py`'s functions then apply unchanged
  (`lab.as_wide`). `lab.model_vectors` runs the lab's own pixels (`config.LAB_LEVEL_DIRS`) through
  `embeddings.py`/`distances.py`. **The embedding cache is keyed by level name alone**, and the lab's levels share
  the web set's names: `extract_level(..., cache_root=, images=)` exists for that, and a lab call without its own
  `cache_root` would silently be served the web set's embeddings
  (`test_model_vectors_keep_a_cache_of_their_own_and_rebuild_for_a_new_level`). The pool grows batch by batch, so
  the headline agreement statistic is the mean r between two raters (`lab.mean_pairwise_r`), which does not depend
  on group size; R_LOO and split-half do, and are compared with the pilot's only through same-sized pilot subgroups
  (`lab.subgroup_agreement`) or the formulas (`lab.expected_loo`, `lab.expected_upper`, `lab.spearman_brown`).
  **The noise ceiling has two bounds** (`lab.noise_ceiling`, per rater): lower, the rater against the mean of the
  others (equal to `analysis.leave_one_out_reliability`), and upper, against the mean of all. The pilot's are
  compared at the lab group's size only (`lab.subgroup_ceilings`; `lab._subgroup_members` draws the subgroups for
  it and for `subgroup_agreement`, so the two describe the same subgroups row for row), and the model's fit is a
  share of the lower ceiling of the raters it was scored on (`A["share"]`, `A["sub_ceiling"]`). A noisy group has
  a low lower ceiling and so a high share: never quote a share without the same-sized pilot groups beside it.
  `lab_run.compute(run_dirs, without=[...])` also scores the group without the named participants: `A["rest"]` is
  the same computation for the others (`None` when nobody is named), and a figure draws it, dashed (`DASH`,
  `REST_LINE`, `_tick`), only when it is there. **A run that takes nobody out must draw exactly what it drew**:
  the first-look companion's one-page summary draws its panels with `rest=None` and has no room for more.
  `lab_run.compute`
  returns every number as plain pandas/numpy; the figures (`lab_run.FIGURES`) and `summary` only draw from it, and
  the write-up in `../supplamentary/lab_first_look/` is built from the same call. Participant-level inputs and
  outputs (run folders, the optional labels CSV, `participants.csv`) live outside this repo, and the participant
  files' free-text `notes`, which can name the participant, are never read into a table. **A participant is
  `S<k>`, not the folder's `P<serial>`**: the lab PC's serial starts over between batches (two batches both hold
  a P1), so `lab.participant_ids` numbers every completed run under `LAB_RUNS_DIR` by start time, and the labels
  CSV names a run by `batch` and `serial`. `lab.load_practice` scores each practice iteration
  and reads from the run's log whether practice was *gated* (repeated until passed) and on what (attention; in the
  runs of 2026-11-23 the rating too: `rating_criterion`). **An iteration is read from the folder of its images**
  (`practice_iteration_<k>`, each with pairs of its own since 2026-11-25; `lab.practice_iterations`), and only in
  the earlier runs, which repeated one set of pairs, from the trial's place over the number of different pairs;
  the rating score needs the named pairs, so it is NaN after the first iteration; `lab.group_contrasts`
  and `lab.group_agreement` then set the gated against the rest with exact p-values over every relabelling
  (4 against 5: 126 ways, so no p below 1/126). `lab.colour_similarity` (RGB-histogram r) is one more per-pair
  reference. A participant who rates nearly every pair the same has no correlation inside a small subset:
  `lab.subset_contrasts` leaves them out and counts `participants`, where it once returned NaN with p = 0.
  **The dot replaced the pilot's letters to stop the second task leaking into the rating**, so every lab run
  reports `lab.rating_shifts` (mean rating of one kind of trial minus another's: `lab.RATING_SHIFTS`, e.g. no
  colour change against some) beside the pilot's own letter effect scored by the same function
  (`lab.pilot_letter_shift`). Test a shift as a difference in means, not as a rank correlation with the number of
  changes: the pattern found (no change highest, one lowest, two between) has no trend for ρ to see. The
  write-ups in `../supplamentary/lab_first_look/` are the companion and a one-page summary built from the same
  values; the summary's panels are `lab_run.TAKEAWAY_PANELS`, each drawn on an axes it is given.
  **Everything that depends on which second task a run had is in `lab.TASKS`**, keyed by `lab.DOT`
  (`colour-change`) and `lab.LETTER_COUNT` (`letters-count`: the letters asked how many of a pair's two were the
  block's target; prepared 2026-11-25, no run yet): the tables the module-level `RELATION_X`, `CONDITIONS`,
  `SUBSET_CONTRASTS`, `RATING_SHIFTS` and `PARTICIPANT_RELATIONS` are for the dot, the columns they need
  (`add_columns`), the per-participant measures and the column of what the task showed on a trial (`load`). A
  trials table carries its `task` (`lab.task_of`), `clean` means the task answered right under either, and
  `lab.find_runs` refuses a selection of runs with different tasks until one is named (`--task`; results then go
  to `<selection>_<task>/`). `lab_run`'s task-specific labels are `TASK_MEASURES`, `CONDITION_TITLES`,
  `LOAD_SENTENCE`, `CLEAN_TITLE` and `PARTICIPANT_X`. The dot's names and numbers are unchanged by this: the
  nine participants' `values.json` is byte-identical before and after. The first-look companion's
  `make_lab_assets.py` pins itself to the colour-change participants (`find_runs(task=lab.DOT)`).

- **`config.py`** — all paths, `LEVEL_FOLDERS` (degradation/desaturation level → stimulus subfolder name),
  `LAYERS`, `DEGRADATION_LEVELS` / `DESATURATION_LEVELS` groupings, exclusion thresholds, reliability/
  plotting constants. Read this first when tracing where a level, layer, or exclusion rule is defined.

- **`progress.py`** — the tqdm bar factories every stage/sub-stage loop uses, so the run reports progress
  as one pinned, multi-line dashboard instead of scrolling prints. Fixed nesting by `position`: depth 0 is
  the whole run (4 stages), depth 1 a stage's primary loop, depth 2 the finer loop nested inside that.
  `progress.write(...)` (a `tqdm.write` wrapper) is used instead of `print(...)` for any milestone message
  emitted while bars are active, since a plain `print` would corrupt the pinned display.

## Known caveats worth knowing before changing analysis logic

- The "AvgPool / Layer_4 / Classifier" interpretation of the original MATLAB analysis's 3 layers is a
  best-effort reading. Raw-data folders contain precomputed `<Layer>_embeddings.csv` under each stimulus
  folder's `ResNet_places/` for cross-checking dimensionality if the layer definitions are ever suspect —
  this pipeline never reads those CSVs itself, it recomputes embeddings fresh.
- Computed reliability (`R_LOO` ≈ 0.39 Letters / 0.76 No-Letters) is in the right ballpark vs. the PDF's
  ≈0.44 / ≈0.78. The No-Letters figure was 0.73 while that condition was still analyzed at N=20; resolving
  its exclusion (the participant whose own LOO r was 0.06 against a next-lowest 0.57) moved it
  to 0.76. The residual gap is attributable to this being a different (web) data collection than the PDF's.
- A PNG variant of the tunnel condition exists on disk (`NavehVer_Tunnel_PNG`); `LEVEL_FOLDERS["Tunnel"]`
  currently points at the white-background JPG variant instead.

## AI Coding Guardrails

1. General behavior
   - Act as a disciplined, context-aware AI pair programmer, not an autonomous refactorer.
   - Treat the existing codebase and documentation as the source of truth, and optimize for diffs that a human reviewer can fully understand and trust.

2. Working in an existing codebase
   - Surgical changes only: Edit exactly what is necessary to satisfy the current request. Do not refactor, rename, reformat, or “improve” unrelated code, comments, or structure unless explicitly asked.
   - Reuse before adding: Prefer using existing functions, classes, modules, patterns, and conventions. Introduce new abstractions only when clearly required for a robust, maintainable solution.
   - Respect project style: Match the current naming, error handling, logging, and architectural patterns. Avoid introducing conflicting paradigms.
   - Docs alignment: If a README or other top-level docs exist, update them surgically whenever your changes make them incomplete or inaccurate, including clarifying previously undocumented behavior or constraints, while keeping all prose minimal and focused.

3. New code and design principles
   - Modular & simple: Follow modular and OOP-leaning design with clear separation of concerns, but avoid over-engineering. Favor simple, explicit solutions, tight interfaces, and minimal configuration surface.
   - Minimal but meaningful comments:
       - Comment only where logic, edge cases, or integration behavior are non-obvious.
       - Do not narrate trivial code or restate what is already clear from names and structure.
   - Documentation discipline: Keep README and other docs concise and task-focused. Explain what future developers need to know to safely extend, debug, or integrate, not every detail.

4. Testing and correctness
   - Correctness first: Always prioritize correctness, safety, and reproducibility over speed of delivery or cleverness.
   - Test behavior, not implementation: When adding or changing behavior, write or extend tests that cover observable behavior and important edge cases, rather than internal implementation details.
   - Always add/extend tests for new behavior:
     - For every new feature, bug fix, or non-trivial refactor, add new tests or extend existing ones to cover the change.
     - Prefer small, focused tests that clearly express the expected behavior and guard against regression.
   - Performance-heavy and pipeline-style code
       - Smoke tests instead of full runs by default: For performance-heavy code (e.g., long-running pipelines, large fMRI/ML jobs, full-model training loops, large ETL), prefer smoke tests and lightweight integration tests over full end-to-end runs on every change.
       - Design smoke tests to be fast and representative:
           - Use small, representative datasets, synthetic data, or reduced configurations that approximate real-world structure without full cost.
           - Aim for tests that verify “does this run end-to-end without crashing, with sane outputs and resource usage?” rather than full metric-level validation.
       - Bounded expectations:
           - For stochastic or retrained models, assert that outputs fall within reasonable bounds (e.g., ranges, shapes, basic invariants) rather than exact values.
           - Avoid tests that require full convergence, full training epochs, or production-scale throughput unless explicitly requested.
       - Full end-to-end tests as special cases:
           - Reserve full end-to-end runs (large datasets, full training, full pipeline execution) for explicitly labeled scenarios (e.g., release gating, performance benchmarking, or dedicated CI jobs), not for routine local changes.
           - When proposing such tests, clearly mark them as “expensive” and, if possible, provide a cheaper smoke-test variant that can be run more frequently.

5. Communication and assumptions
    - Clarify before acting: When the request is underspecified or ambiguous, ask targeted questions instead of guessing.
    - Expose tradeoffs: When there are meaningful tradeoffs (speed vs. memory, readability vs. abstraction, smoke vs. full end-to-end testing), briefly state them and choose the option that best matches the existing project philosophy.
    - Diff-optimized changes: Assume your work will be reviewed as diffs. Aim for small, coherent commits with clear intent and no hidden behavior changes.

6. Session legacy node (Obsidian)
    - Location: The persistent history of our collaboration lives in the knowledge vault, **not in this repo**:
      the node **named `Sim-Rich Analysis`** under `~/Documents/obsidian_knowledge/95 AI Session Legacy/`.
      Resolve its folder from the routing table in that directory's `_Legacy MOC.md` rather than from a
      path written here — the corpus re-arranges its sub-categories as nodes are added, and this line
      went stale on 2026-09-08 when `MSc Research/Sim-Rich/` was split out.
      Corpus index: `95 AI Session Legacy/_Legacy MOC.md`. Full corpus rules: `~/.claude/CLAUDE.md` §6–§7.
      Sibling node: `Sim-Rich Experiment` (the experiment-side repo) — read it too when a
      change spans both.
    - Session start behavior: At the start of each new session:
        - Read the node to recover context.
        - Use it to understand past decisions, known pitfalls, and existing patterns before proposing changes.
        - The node does not exist yet. Create it on the first logged session — frontmatter, a one-paragraph
          "what this project is", a link back to `[[_Legacy MOC]]` — and register it in the MOC before doing
          any other work.
    - Log entry structure: For each meaningful increment, append a concise entry under a
      `## Session N — <name>` heading (newest at the bottom) that includes:
        - Request: What was asked.
        - Changes: Files, functions, interfaces, tests, and docs that were modified or added.
        - Trials and errors: Any significant dead ends, surprising behavior, or debugging paths.
        - Retrospective: In hindsight, the root cause of any issue plus the final solution and any reusable lessons or patterns learned.
    - Keep the node's `updated:` frontmatter field current on every write.
    - Retired: `./.claude/project_legacy_log`. Never write project history into this repo.

7. Log hygiene and corpus curation
    - High-signal overview: Treat the node as an accumulated overview of technical and conceptual evolution—not a scratchpad.
    - Collapse solved noise: When an issue is solved after a long debugging or exploratory process, replace verbose, step-by-step notes with a short, high-value summary that a future reader can understand in under a minute.
    - Avoid duplication: Do not repeat the same information across multiple entries; reference prior sessions with an Obsidian heading link (`[[Sim-Rich Analysis#Session 1 — …]]`) instead of restating them.
    - Keep entries skimmable: Use short paragraphs or bullets; avoid long narrative blocks.
    - Curate the tree: Whenever a node is added anywhere in `95 AI Session Legacy/`, re-examine the whole corpus and re-arrange it if it is no longer the crispest division of the material — merge stray sub-categories, split heterogeneous ones, rename any that no longer describe their contents, and name sessions for what they accomplished. Update `_Legacy MOC.md` to match.

8. Session message rule
    - Message counting: Count only user messages in the current conversation (exclude assistant replies).
    - On every third message: Before executing the request, update the legacy node with a session-level summary of all the new messages that have been added: what was covered, what changed, and outstanding questions or TODOs.
    - treat every 6th message as the first prompt of a new logical session, while preserving all existing files and context.
    - Only then proceed to execute the 6th request under the same guardrails.

9. Git operations
   - Never initiate any git operation (status, diff, add, commit, push, pull, branch, checkout, stash, reset, etc.) unless the user explicitly asks for it in the current request.
   - Simply work on the currently checked-out branch, assuming no one else works on it at the same time — do not create branches, switch branches, or stash/restore state on your own initiative.

10. Global priorities
    - Favor explicitness over guesswork.
    - Favor tight feedback loops over broad autonomous changes.
    - Favor fast smoke tests for frequent validation and reserve full end-to-end runs for explicit, high-value scenarios.
    - Favor code and tests that a careful human reviewer would trust after reading only the diffs, the tests, and the legacy node.
