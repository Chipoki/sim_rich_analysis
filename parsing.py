"""
Turn the raw jsPsych-style result files into one tidy pandas DataFrame,
long format: one row per participant x main-phase trial.

Verified against the actual data (both files, all 20+20 participants):
  - Each line is a JSON array of trial objects for one participant.
  - A logical trial is `psychophysics`, optionally followed by
    `letter_question` (letters file only), then `similarity_question` --
    always consecutive in the stream and always sharing the same `pair_id`.
    `trial_index` is per-phase and NOT a safe join key; we never use it
    for joining.
  - Within a participant, main-phase `pair_id` values are unique (200 of
    them), and the same 200 pair_ids are used by every participant in both
    conditions -- so pair_id is a safe, stable key to align vectors across
    participants and across conditions later on.
"""

import json
from pathlib import Path

import pandas as pd

import progress

TIMING_FIELDS = [
    "fixation_actual_ms", "fixation_actual_frames",
    "image1_actual_ms", "image1_actual_frames",
    "mask1_actual_ms", "mask1_actual_frames",
    "image2_actual_ms", "image2_actual_frames",
    "mask2_actual_ms", "mask2_actual_frames",
]


def _parse_participant(trials: list[dict], condition: str) -> tuple[dict, list[dict]]:
    """Parse one participant's trial-object array into (meta, rows)."""
    session_metadata = next(t for t in trials if t["trial_type"] == "session_metadata")
    participant_info = next(t for t in trials if t["trial_type"] == "participant_info")

    meta = {
        "participant_id": participant_info["participant_id"],
        "run_id": session_metadata.get("run_id"),
        "condition": condition,
        "age": participant_info.get("participant_age"),
        "gender": participant_info.get("participant_gender"),
        "handedness": participant_info.get("participant_handedness"),
    }

    rows = []
    n = len(trials)
    i = 0
    while i < n:
        t = trials[i]
        if t.get("trial_type") == "psychophysics" and t.get("phase") == "main":
            row = {
                "participant_id": meta["participant_id"],
                "run_id": meta["run_id"],
                "condition": condition,
                "pair_id": t["pair_id"],
                "image1_id": t["image1_id"],
                "image2_id": t["image2_id"],
                "block_index": t["block_index"],
                "trial_index_within_block": t["trial_index_within_block"],
            }
            for f in TIMING_FIELDS:
                row[f] = t.get(f)

            j = i + 1
            # Optional letter_question (letters condition only)
            if j < n and trials[j].get("trial_type") == "letter_question" and trials[j].get("pair_id") == t["pair_id"]:
                lq = trials[j]
                row["letter1_shown"] = t.get("letter1_shown")
                row["letter2_shown"] = t.get("letter2_shown")
                row["letter1_rotation_deg"] = t.get("letter1_rotation_deg")
                row["letter2_rotation_deg"] = t.get("letter2_rotation_deg")
                row["letter_discrimination_correct"] = lq.get("letter_discrimination_correct")
                row["letter_rt"] = lq.get("rt")
                j += 1
            else:
                row["letter1_shown"] = None
                row["letter2_shown"] = None
                row["letter1_rotation_deg"] = None
                row["letter2_rotation_deg"] = None
                row["letter_discrimination_correct"] = None
                row["letter_rt"] = None

            # similarity_question must follow (same pair_id)
            if j < n and trials[j].get("trial_type") == "similarity_question" and trials[j].get("pair_id") == t["pair_id"]:
                sq = trials[j]
                row["participant_similarity_rating"] = sq.get("participant_similarity_rating")
                row["similarity_rt"] = sq.get("rt")
                j += 1
            else:
                raise ValueError(
                    f"Expected similarity_question after pair_id={t['pair_id']} "
                    f"for participant {meta['participant_id']}, got "
                    f"{trials[j].get('trial_type') if j < n else 'end of stream'}"
                )

            rows.append(row)
            i = j
        else:
            i += 1

    return meta, rows


def parse_file(path: Path, condition: str) -> list[dict]:
    """Parse a whole result file (one participant per line) into tidy rows."""
    rows = []
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in progress.bar(lines, desc=f"{condition} participants", depth=progress.STEP):
        trials = json.loads(line)
        _meta, participant_rows = _parse_participant(trials, condition)
        rows.extend(participant_rows)
    return rows


def load_all(result_files: dict[str, Path]) -> pd.DataFrame:
    """Load both conditions into one tidy long-format DataFrame."""
    all_rows = []
    for condition, path in progress.bar(
        result_files.items(), total=len(result_files), desc="Parsing files", depth=progress.STAGE
    ):
        all_rows.extend(parse_file(path, condition))
    df = pd.DataFrame(all_rows)
    df["letter_discrimination_correct"] = df["letter_discrimination_correct"].astype("boolean")
    return df


def letter_accuracy_by_participant(df: pd.DataFrame) -> pd.Series:
    """Per-participant L/T discrimination accuracy (letters condition only)."""
    letters = df[df["condition"] == "letters"]
    return letters.groupby("participant_id")["letter_discrimination_correct"].mean()


def letter_accuracy_by_participant_and_block(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-participant, per-block L/T discrimination accuracy (letters condition
    only). Index = participant_id, columns = block_index (0..4), values =
    that participant's accuracy within that experimental block -- a vigilance/
    fatigue diagnostic at finer grain than the single overall accuracy figure.
    """
    letters = df[df["condition"] == "letters"]
    return (
        letters.groupby(["participant_id", "block_index"])["letter_discrimination_correct"]
        .mean()
        .unstack("block_index")
    )
