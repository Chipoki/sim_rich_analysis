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
    │   ├── dist_vectors.pkl
    │   ├── analysis_results.pkl
    │   └── embeddings/<level>_<layer>.pkl
    └── diagrams/*.png
```

If your raw-data folder isn't literally named `src_data`, update
`SRC_DATA_DIR` at the top of `config.py` — that's the only path you should
ever need to touch.

## 1. Environment setup (CPU-only, Ubuntu 24.04 / i7-13700H / 16 GB RAM)

`environment.yml` pulls `pytorch-cpu` from `conda-forge` — a build of
PyTorch with no CUDA dependency at all, so there's nothing GPU-related to
misconfigure and no multi-GB CUDA toolkit to download.

```bash
conda env create -f environment.yml
conda activate web-analysis
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
every subsequent run reads straight from `web_analysis_results/data/embeddings/`
and is fast. Delete a specific `<level>_<layer>.pkl` to force recomputation
of just that one.

Each analysis figure is displayed and saved to
`web_analysis_results/diagrams/`.

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
  original prompt's aspirational ask) but is **not** the default; it needs
  internet access on first run to fetch weights via `torch.hub`.

- **Distances** (`distances.py`): cosine distance throughout. These CNN
  layers are ReLU-activated (non-negative), so cosine distance stays in
  `[0, 1]` and `similarity = 1 - distance` is a valid bounded similarity.
- **Exclusions** (`analysis.py` / `config.py`): the Letters exclusion
  (`letter_discrimination_correct` accuracy < 80%) is computed
  automatically from the data — verified to match the PDF's N=19 exactly
  (one participant at 66.5% accuracy). The No-Letters exclusion ("claimed
  technical problems") is a self-report with **no corresponding field in
  the JSON** — set `NO_LETTERS_MANUAL_EXCLUDE_PARTICIPANT_ID` in
  `config.py` if you know which participant that was; it defaults to
  `None` (N=20, all included).
- **Reliability**: `leave_one_out_reliability()` doubles as both the
  PDF's headlined R_LOO and the noise ceiling plotted alongside model
  bars. `split_half_reliability()` is Spearman-Brown corrected. Both were
  validated against the PDF's headline numbers using the real rating data
  during development: computed R_LOO ≈ 0.39 (Letters) / 0.73 (No-Letters)
  vs. the PDF's ≈0.44 / ≈0.78 — in the right ballpark and same direction
  (No-Letters much more reliable), with the remaining gap plausibly
  explained by the still-unresolved No-Letters exclusion above and by
  this being a different (web) data collection than the original PDF's.
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
`tests/test_slow_e2e.py` runs the real, unmocked pipeline and is skipped whenever real `src_data`/weights
aren't present; `pytest.ini` also deselects it by default so plain `pytest` never triggers it.
