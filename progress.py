"""
Centralized tqdm bar factories, so every stage of the pipeline reports
progress the same way: a pinned, multi-line dashboard rather than scrolling
print/bar spam.

Nesting is by fixed `position`, capped at 3 depths:
  - PIPELINE (0): the whole run, one tick per major stage. `leave=True` so it
    persists on screen for the entire run.
  - STAGE (1): the current stage's primary loop (files being parsed, levels
    being embedded, conditions being analyzed, ...).
  - STEP (2): the finer loop nested inside that (participants within a file,
    images within a level, split-half's iterations, ...).

Because pinned bars corrupt if a plain `print()` interleaves with them, any
milestone messages emitted while a bar is active should go through
`tqdm.write(...)` instead of `print(...)`.
"""

from typing import Iterable, Optional

from tqdm import tqdm

PIPELINE, STAGE, STEP = 0, 1, 2

_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"


def pipeline_bar(stage_names: list[str]) -> tqdm:
    """The depth-0 bar tracking overall progress through the pipeline's major stages."""
    return tqdm(
        total=len(stage_names),
        position=PIPELINE,
        leave=True,
        desc="Pipeline",
        bar_format=_BAR_FORMAT,
    )


def bar(
    iterable: Optional[Iterable] = None,
    *,
    total: Optional[int] = None,
    desc: str = "",
    depth: int = STAGE,
    leave: bool = False,
) -> tqdm:
    """A depth-1 or depth-2 sub-bar, pinned at a fixed screen line via `position`."""
    return tqdm(
        iterable,
        total=total,
        position=depth,
        leave=leave,
        desc=desc,
        bar_format=_BAR_FORMAT,
    )


def write(message: str) -> None:
    """Print a milestone message without corrupting active pinned bars."""
    tqdm.write(message)
