"""
Build per-participant human similarity vectors, aligned by `pair_id`.

Verified against the real data: all 40 participants (both conditions) rate
the *same* 200 pairs, just in a different (per-participant randomized)
trial order. So vectors must be aligned by pair_id -- not by trial
position -- to be comparable across participants or against a model
vector.
"""

import pickle
from pathlib import Path

import pandas as pd


def build_human_similarity_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide table of human similarity ratings: index = pair_id (sorted, stable
    order shared by every participant/model vector downstream), columns =
    (condition, participant_id) MultiIndex, values = 1-6 rating.
    """
    wide = df.pivot_table(
        index="pair_id",
        columns=["condition", "participant_id"],
        values="participant_similarity_rating",
        aggfunc="first",
    )
    wide = wide.sort_index()
    return wide


def build_block_assignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide table of each participant's experimental block index per pair_id --
    same shape as build_human_similarity_vectors' output (index = pair_id,
    columns = (condition, participant_id) MultiIndex), but the value is which
    of the 5 presentation blocks that pair fell into for that participant.

    Block membership is independently randomized per participant (verified
    against the real data), so -- unlike, say, the level a pair belongs to --
    this is genuinely a per-(pair_id, participant) lookup, not a constant.
    """
    wide = df.pivot_table(
        index="pair_id",
        columns=["condition", "participant_id"],
        values="block_index",
        aggfunc="first",
    )
    return wide.sort_index()


def get_vector(wide: pd.DataFrame, condition: str, participant_id: str) -> pd.Series:
    """Convenience accessor for one participant's ordered rating vector."""
    return wide[(condition, participant_id)]


def get_group(wide: pd.DataFrame, condition: str) -> pd.DataFrame:
    """All participants' vectors for one condition, pair_id x participant_id."""
    return wide[condition]


def save(wide: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(wide, f)


def load(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        return pickle.load(f)
