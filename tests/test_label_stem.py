"""Label/image file stem shortening shared by stage1 and grid search."""

from pathlib import Path

import safetrajai.grid_search.core as grid_search_core
import safetrajai.pipeline.sequence_tracker as sequence_tracker
from safetrajai.evaluation.dataset_layout import short_label_stem

PLACE_NAME = " 中華三路 民生二路(東北角)_"


def test_strips_the_place_name_of_the_evaluation_camera() -> None:
    seq_name = "C000025 中華三路 民生二路(東北角)_20251209145630_15s_1"
    assert short_label_stem(seq_name) == "C000025_20251209145630_15s_1"


def test_leaves_other_sequence_names_unchanged() -> None:
    for seq_name in ("C000054 民族一路 天祥二路_20250627071742", "plain_name", ""):
        assert short_label_stem(seq_name) == seq_name


def test_replaces_every_occurrence_like_str_replace() -> None:
    seq_name = f"a{PLACE_NAME}b{PLACE_NAME}c"
    assert short_label_stem(seq_name) == "a_b_c"


def test_callers_use_the_shared_helper() -> None:
    for module in (sequence_tracker, grid_search_core):
        assert PLACE_NAME not in Path(module.__file__).read_text(encoding="utf-8")
