"""Pure helpers of the person review page: verdict encoding, box corrections, crop geometry."""

import importlib.util

import pytest

from safetrajai.paths import PROJECT_ROOT

_spec = importlib.util.spec_from_file_location(
    "serve_person_review", PROJECT_ROOT / "scripts" / "cvat_review" / "serve_person_review.py")
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)


def test_partial_verdict_keeps_the_class_and_legacy_partial_means_person() -> None:
    assert review.split_verdict("partial:scooter") == ("partial", "scooter")
    assert review.split_verdict("partial") == ("partial", "person")
    assert review.split_verdict("scooter") == ("normal", "scooter")
    assert review.split_verdict("not_person") == ("none", "person")
    assert review.split_verdict("unsure") == ("unsure", "person")


def test_only_known_verdicts_are_accepted() -> None:
    assert "partial:small_truck" in review.VERDICTS
    assert "partial:dog" not in review.VERDICTS
    assert "bus" in review.VERDICTS and "normal" not in review.VERDICTS


def test_corrections_are_validated_against_box_count_and_geometry() -> None:
    assert review.parse_corrections({"1": [10, 20, 30.5, 60]}, 2) == {"1": [10.0, 20.0, 30.5, 60.0]}
    for bad in ({"2": [0, 0, 5, 5]}, {"0": [5, 5, 5, 9]}, {"0": [0, 0, 5]}, {"0": [9, 9, 3, 3]}):
        with pytest.raises(ValueError):
            review.parse_corrections(bad, 2)


def test_crop_geometry_pads_boxes_clamps_to_image_and_scales_up_small_crops() -> None:
    left, top, right, bottom, scale = review.crop_geometry([[100, 100, 140, 200, 0.9]], 1280, 720)
    assert (left, top, right, bottom) == (50, 50, 190, 250)
    assert scale == pytest.approx(420 / 200)
    left, top, _, _, _ = review.crop_geometry([[0, 0, 40, 100, 0.9]], 1280, 720)
    assert (left, top) == (0, 0)
