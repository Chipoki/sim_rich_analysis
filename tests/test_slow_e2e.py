"""
Opt-in full end-to-end test: runs the real, unmocked pipeline (real
src_data images, real resnet50_places365 weights) via `python run.py`'s
equivalent. Reserved for explicit, dedicated verification per CLAUDE.md's
guardrail against full-scale runs on every change -- pytest.ini's `addopts`
deselects the `slow` marker by default, so run it deliberately with
`pytest -m slow`, not as part of the routine fast suite.

Additionally skipped (even under `-m slow`) when the real sibling
data/weights aren't present, so it's a no-op in any environment lacking the
real raw-data folder.
"""

import pytest

import config
import run

_missing_reasons = []
if not config.SRC_DATA_DIR.exists():
    _missing_reasons.append(f"SRC_DATA_DIR not found: {config.SRC_DATA_DIR}")
if not config.RESNET_WEIGHTS_PATH.exists():
    _missing_reasons.append(f"RESNET_WEIGHTS_PATH not found: {config.RESNET_WEIGHTS_PATH}")


@pytest.mark.slow
@pytest.mark.skipif(bool(_missing_reasons), reason="; ".join(_missing_reasons))
def test_full_pipeline_against_real_data():
    run.run()
