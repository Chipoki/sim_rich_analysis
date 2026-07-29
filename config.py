"""
Central configuration for the peripheral-similarity analysis project.

Everything path- or constant-related lives here so the rest of the codebase
never hardcodes a folder name, layer name, or exclusion rule.

EDIT ME: `SRC_DATA_DIR` below assumes this project folder (`web_analysis/`)
sits as a *sibling* of your raw-data folder. If your raw-data folder isn't
named "web_analysis_src_data", change SRC_DATA_DIR accordingly.
"""

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

for _d in (RESULTS_DIR, DATA_DIR, EMBEDDINGS_DIR, DIAGRAMS_DIR):
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

DINOV2_MODEL_NAME = "dinov2_vits14"  # used only if BACKEND == "dinov2"

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
# problems") is a self-report with no corresponding field in the JSON, so
# it can't be recovered from the data alone. Fill in the participant_id
# below if you know which one it was; leave as None to analyze all N=20.
NO_LETTERS_MANUAL_EXCLUDE_PARTICIPANT_ID = None

# --------------------------------------------------------------------------
# Reliability analyses
# --------------------------------------------------------------------------

SPLIT_HALF_N_ITER = 1000
SPLIT_HALF_SEED = 0

# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

FIGURE_DPI = 150
