"""
Central configuration for the peripheral-similarity analysis project.

Everything path- or constant-related lives here so the rest of the codebase
never hardcodes a folder name, layer name, or exclusion rule.

EDIT ME: `SRC_DATA_DIR` below assumes this project folder (`web_analysis/`)
sits as a *sibling* of your raw-data folder. If your raw-data folder isn't
named "web_analysis_src_data", change SRC_DATA_DIR accordingly.
"""

import warnings
from pathlib import Path

# --------------------------------------------------------------------------
# Top-level directories
# --------------------------------------------------------------------------

# This project's own folder.
PROJECT_ROOT = Path(__file__).resolve().parent

# The raw-data folder (where All20Letters.txt, All20NoLetters.txt, the
# stimuli sub-folders, and resnet50_places365.pth.tar live). Sibling of
# PROJECT_ROOT, per the requested layout. Rename here if yours differs.
SRC_DATA_DIR = PROJECT_ROOT.parent / "web_analysis_src_data"

# Everything this pipeline *writes* (cached embeddings, tidy dataframes,
# figures) lives here -- a sibling of both the project and the raw data,
# never inside PROJECT_ROOT.
RESULTS_DIR = PROJECT_ROOT.parent / "web_analysis_results"
DATA_DIR = RESULTS_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DIAGRAMS_DIR = RESULTS_DIR / "diagrams"
# Cross-model comparison figures (e.g. ResNet50-Places365 vs DinoV2) -- the
# only figures that need more than one backend's results at once -- live
# here instead of under any single backend's own diagrams/<name>/ sub-tree.
# See run._run_integrated_figures().
INTEGRATED_DIAGRAMS_DIR = DIAGRAMS_DIR / "integrated"

for _d in (RESULTS_DIR, DATA_DIR, EMBEDDINGS_DIR, DIAGRAMS_DIR, INTEGRATED_DIAGRAMS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Raw result files (one line per participant, JSON array of trial objects)
# --------------------------------------------------------------------------

RESULT_FILES = {
    "letters": SRC_DATA_DIR / "All20Letters.txt",
    "no_letters": SRC_DATA_DIR / "All20NoLetters.txt",
}

N_MAIN_TRIALS = 200
RATING_SCALE = (1, 6)

# --------------------------------------------------------------------------
# Degradation-level -> stimulus-folder map (folder names verified against
# the actual data tree)
# --------------------------------------------------------------------------

LEVEL_FOLDERS = {
    "Full": "stimuli_full_fld",
    "Deg05": "NavehVer_05Stim",
    "Deg3": "NavehVer_3Stim",
    "Tunnel": "NavehVer_Tunnel_WhiteBG_JPG",
    "Desaturated": "Naveh_Desaturated",
    "DesaturatedBigFov": "Naveh_BigFoveaDesaturated",
}

# A PNG variant of the tunnel condition also exists on disk
# (NavehVer_Tunnel_PNG). Swap LEVEL_FOLDERS["Tunnel"] to that if you want
# the PNG stimuli instead of the white-background JPGs.

# Analysis groupings (§4 of the spec)
DEGRADATION_LEVELS = ["Full", "Deg05", "Deg3", "Tunnel"]
DESATURATION_LEVELS = ["Full", "DesaturatedBigFov", "Desaturated"]

# All levels actually needed anywhere in the pipeline (union, de-duplicated,
# order preserved) -- this is what embeddings.py / distances.py will loop
# over so nothing is computed twice.
ALL_LEVELS = list(dict.fromkeys(DEGRADATION_LEVELS + DESATURATION_LEVELS))

IMAGE_ID_WIDTH = 3  # image_093.jpg -> zero-padded to 3 digits
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")  # tried in this order per level

# --------------------------------------------------------------------------
# Embedding backend
# --------------------------------------------------------------------------

BACKEND = "resnet50_places365"  # or "dinov2"

# Backends *additionally* run end-to-end (embeddings -> analysis -> the same
# figure set produced for BACKEND) alongside the primary one, each writing
# its own figures under a diagrams/<backend dir name>/ sub-tree so they never
# overwrite each other's outputs -- including BACKEND itself, which gets its
# own sub-tree too (see BACKEND_DIR_NAMES below), not a special-cased
# unprefixed path.
EXTRA_BACKENDS = ["dinov2"]

# Internal backend key -> short on-disk directory name, used to namespace
# both diagrams/<name>/ (run.py) and data/embeddings/<name>/ (embeddings.py)
# per model. Every backend in BACKEND/EXTRA_BACKENDS must have an entry here.
BACKEND_DIR_NAMES = {
    "resnet50_places365": "resnet50",
    "dinov2": "dinov2",
}

RESNET_WEIGHTS_PATH = SRC_DATA_DIR / "resnet50_places365.pth.tar"
RESNET_NUM_CLASSES = 365

# The 3 layers examined in the original MATLAB analysis (PDF: "layer with
# highest correlation, out of 3 examined"). Internal name -> human label
# (the human label matches the precomputed CSV filenames under each
# stimulus folder's ResNet_places/ sub-folder, useful for cross-checking).
LAYERS = {
    "avgpool": "AvgPool",   # 2048-d, global-average-pooled penultimate features
    "layer4": "Layer_4",    # flattened last conv block (2048x7x7 -> 100352-d)
    "classifier": "Classifier",  # 365-d final-layer logits
}
DEFAULT_LAYER = "avgpool"

# Colorblind-safe qualitative palette (ColorBrewer "Dark2"), one color per
# layer -- used to color-code which layer won a per-participant "best layer"
# comparison (plotting.py).
LAYER_COLORS = {
    "avgpool": "#1b9e77",
    "layer4": "#d95f02",
    "classifier": "#7570b3",
}

DINOV2_MODEL_NAME = "dinov2_vits14"  # used only if BACKEND/EXTRA_BACKENDS includes "dinov2"

# Every backend other than resnet50_places365 exposes a single pooled
# embedding (no per-layer breakdown), keyed by the backend's own name so it
# can be run through the same per-layer plumbing as resnet50_places365's
# LAYERS/LAYER_COLORS (a size-1 "layers" dict instead of size-3). Internal
# key -> human label / hex color, mirroring LAYERS / LAYER_COLORS above.
SINGLE_LAYER_BACKEND_LABELS = {"dinov2": "DinoV2"}
SINGLE_LAYER_BACKEND_COLORS = {"dinov2": "#e7298a"}

# --------------------------------------------------------------------------
# DINOv2 grid-search integration (supplemental/dinov2_layer_grid_search)
# --------------------------------------------------------------------------

# Where run_grid_search.py dumps its scored (block x pooling) results. If
# this file exists when the main pipeline runs "dinov2" as a backend
# (config.BACKEND or config.EXTRA_BACKENDS), run.py/embeddings.py
# automatically substitute the single pooled DINOv2 embedding for the top
# DINOV2_GRID_TOP_N grid configs ranked by avg_vector_r -- see
# embeddings.dinov2_grid_layers(). Falls back to the single pooled "dinov2"
# embedding (SINGLE_LAYER_BACKEND_LABELS/COLORS above) if this file is
# missing, i.e. the grid search hasn't been run yet.
DINOV2_GRID_SEARCH_DIR = RESULTS_DIR / "supplemental" / "dinov2_layer_grid_search"
DINOV2_GRID_SEARCH_DATA_DIR = DINOV2_GRID_SEARCH_DIR / "data"
DINOV2_GRID_SEARCH_DIAGRAMS_DIR = DINOV2_GRID_SEARCH_DIR / "diagrams"
DINOV2_GRID_SEARCH_RESULTS_PATH = DINOV2_GRID_SEARCH_DATA_DIR / "grid_results.pkl"

DINOV2_GRID_N_BLOCKS = 12
DINOV2_GRID_POOLINGS = ("cls", "mean")
# Matches len(LAYERS), so the "best config" plot stays a 3-way color
# comparison for dinov2 just like ResNet50's 3 examined layers.
DINOV2_GRID_TOP_N = 3

# --------------------------------------------------------------------------
# DINOv2 a-priori configs (take precedence over the grid search above)
# --------------------------------------------------------------------------

# Which DINOv2 (block x pooling) configs the *main* pipeline reports. When
# this is non-empty it overrides the grid search entirely -- run.py and
# embeddings.get_extractor() use exactly these, and grid_results.pkl is not
# consulted at all.
#
# Why pin instead of grid-search: the grid ranks 24 configs by their
# correlation with the very human ratings the pipeline then reports, so
# whichever configs win are selected on the evaluation data and their
# reported r is optimistically biased. Choosing a fixed, theory-driven set up
# front removes that selection step, so DINOv2's numbers mean the same thing
# ResNet50's do (config.LAYERS is likewise fixed, never data-selected).
#
# The set: three depths spaced evenly across the 12-block ViT-S (early /
# middle / final third), each read out both ways DINOv2 is normally used --
# the CLS token and mean-pooled patch tokens. Depth and pooling are crossed
# so the two factors stay separable when reading the figures.
#
# Set to () to fall back to the grid search (see DINOV2_GRID_SEARCH_* above).
DINOV2_PINNED_CONFIGS = (
    "block04_cls", "block04_mean",
    "block08_cls", "block08_mean",
    "block12_cls", "block12_mean",
)

# The single config standing in for DINOv2 wherever one representative layer
# is plotted (figure titles' "layer: ...", the by-block breakdown) -- the
# analogue of DEFAULT_LAYER for resnet50_places365. Final-block CLS, i.e.
# what DinoV2Extractor returns on its own, rather than DINOV2_PINNED_CONFIGS'
# first entry, which is the *shallowest* config and a poor stand-in for the
# model. Must be one of DINOV2_PINNED_CONFIGS.
DINOV2_PINNED_DEFAULT_CONFIG = "block12_cls"

# Colorblind-safe palette (ColorBrewer Dark2, 8-class) assigned to
# grid-selected configs in ranked order -- extends LAYER_COLORS' 3 colors in
# case DINOV2_GRID_TOP_N is ever raised past 3.
DINOV2_GRID_TOP_N_COLORS = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666",
]

# CPU-only batching -- kept small to stay well inside 16 GB RAM.
BATCH_SIZE = 8
IMAGE_RESIZE = 256
IMAGE_CROP = 224

# --------------------------------------------------------------------------
# Distances / similarity
# --------------------------------------------------------------------------

# Cosine distance is used throughout because it is naturally bounded to
# [0, 1] for the non-negative activations these layers produce, so
# similarity = 1 - distance is a valid similarity in [0, 1].
DISTANCE_METRIC = "cosine"

# --------------------------------------------------------------------------
# Exclusions (PDF: Letters N=19, No-Letters N=19)
# --------------------------------------------------------------------------

# Letters: computed automatically from letter_discrimination_correct
# (verified against the real data -- exactly one participant falls below
# this threshold, at ~66.5% accuracy).
LETTER_ACCURACY_THRESHOLD = 0.80

# No-Letters: the PDF's exclusion ("single participant claimed technical
# problems") is a self-report with no corresponding field in the JSON, so it
# can't be read off the data directly -- but it *was* recovered from the raw
# file's presentation telemetry. This participant is the only one of the 20
# with any missing frame-timing block at all: 165 of its 212 stimulus trials
# logged no measured_frame_time_ms / measured_refresh_rate_hz / *_actual_ms
# (every other participant logged 212/212), including one whole 40-trial
# block, and the 47 trials that did log dipped to 22.2 Hz against its own
# steady 60 Hz baseline. Stimuli were still displayed -- the presentation
# sequence ran its full ~1689 ms, same as everyone else -- so this is a
# logging failure rather than a blank screen; what makes it an exclusion
# rather than merely unverified data is that the untimed trials also drew
# ~2-3x faster ratings than the timed ones *within the same trial bins*
# (456 vs 1300 ms overall), so they aren't behaviorally interchangeable.
#
# Unlike LETTER_ACCURACY_THRESHOLD above -- a rule the data is scored against
# -- this is a single named id, so it is applied at *acquisition*
# (parsing.parse_file) and the participant never enters the tidy DataFrame.
# Set to None to analyze all N=20.
#
# The id is a Prolific participant identifier, and naming it here alongside
# "this person's data was unusable" is participant-level information, so it is
# kept out of version control: it lives in an untracked local_config.py (see
# local_config.example.py, and .gitignore). Without that file the exclusion
# cannot be applied and the condition would silently revert to N=20 -- hence
# the warning rather than a quiet None.
try:
    from local_config import NO_LETTERS_MANUAL_EXCLUDE_PARTICIPANT_ID
except ImportError:
    NO_LETTERS_MANUAL_EXCLUDE_PARTICIPANT_ID = None
    warnings.warn(
        "local_config.py not found -- the No-Letters technical-problems "
        "exclusion will NOT be applied, so that condition is analyzed at N=20 "
        "rather than the reported N=19. Copy local_config.example.py to "
        "local_config.py and fill in the id to restore it.",
        stacklevel=2,
    )

# --------------------------------------------------------------------------
# Reliability analyses
# --------------------------------------------------------------------------

SPLIT_HALF_N_ITER = 1000
SPLIT_HALF_SEED = 0

# --------------------------------------------------------------------------
# Lab experiment (run type `python run.py --lab`; lab.py, lab_run.py)
# --------------------------------------------------------------------------

# The lab experiment's own tree (thesis/experiment), beside this repo's parent.
LAB_EXPERIMENT_DIR = PROJECT_ROOT.parent.parent / "experiment"

# One sub-folder per batch of participants, each holding its runs' folders
# (P<serial>_<date>_<times>_complete). The pool is every batch.
LAB_RUNS_DIR = LAB_EXPERIMENT_DIR / "collected_data" / "wet_runs"

# Optional, kept by the experimenter: one row per participant (`participant`,
# then any label columns, e.g. `strategy`). The first label column sets the
# marker shape in the per-participant figure. Participant-level, so it lives
# with the data, never in this repo.
LAB_PARTICIPANT_LABELS = LAB_RUNS_DIR / "participant_labels.csv"

# Web image id -> the lab's file name for that image (the same name at every
# level). The lab shows the web pilot's 200 fixed pairs, so its ratings are
# keyed by the pilot's own pair_ids.
LAB_ID_MAP = LAB_EXPERIMENT_DIR / "sim_rich_experiment" / "stimuli_web_id_to_local_map.json"

# Level -> folder of the lab's own pixels, in degradation order. Scramble 0.5
# ("Deg05", between Full and Deg3) is not rendered yet: add its folder here
# once stimuli/main400_levels/ holds it.
LAB_LEVEL_DIRS = {
    "Full": LAB_EXPERIMENT_DIR / "stimuli" / "Chosen_425",
    "Deg3": LAB_EXPERIMENT_DIR / "stimuli" / "main400_levels" / "scramble_3.0",
    "Tunnel": LAB_EXPERIMENT_DIR / "stimuli" / "main400_levels" / "tunnel",
}
LEVEL_LABELS = {"Full": "Full", "Deg05": "Scramble 0.5", "Deg3": "Scramble 3.0", "Tunnel": "Tunnel"}

LAB_TRIALS_PER_BLOCK = 40

# Written by the lab run type; created on first use. `data/` holds the caches
# shared by every selection of batches (embeddings, distance vectors).
LAB_RESULTS_DIR = PROJECT_ROOT.parent / "lab_analysis_results"
LAB_DATA_DIR = LAB_RESULTS_DIR / "data"

LAB_BOOTSTRAP_N = 2000
LAB_BOOTSTRAP_SEED = 0
LAB_PERMUTATIONS = 5000
# A pilot group holds more groups of k raters than are worth scoring once k
# grows (19 choose 9 = 92,378); beyond this many, a random draw of them.
LAB_MAX_SUBGROUPS = 20000

# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

FIGURE_DPI = 150
