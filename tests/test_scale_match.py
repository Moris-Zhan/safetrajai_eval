"""Unit tests for satellite image similarity estimation used by the scale check."""

import cv2
import numpy as np
import pytest

from safetrajai.collision.homography.scale_match import estimate_similarity, scale_within_tolerance


def structured_image(size: int = 1280, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 90, np.uint8)
    for _ in range(400):
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        x, y = (int(v) for v in rng.integers(0, size, 2))
        w, h = (int(v) for v in rng.integers(10, 80, 2))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        else:
            cv2.circle(img, (x, y), w // 2, color, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def transformed(img: np.ndarray, scale: float, shift: tuple) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, scale)
    M[:, 2] += shift
    return cv2.warpAffine(img, M, (w, h), borderValue=(90, 90, 90))


@pytest.mark.parametrize("scale", [1.00, 1.05])
def test_estimated_scale_is_accurate(scale: float) -> None:
    ref = structured_image()
    est = estimate_similarity(ref, transformed(ref, scale, (25.0, -40.0)))
    assert est.scale == pytest.approx(scale, rel=0.005)
    assert est.inliers >= 20


def test_scale_tolerance_decides_pass() -> None:
    ref = structured_image()
    assert scale_within_tolerance(estimate_similarity(ref, transformed(ref, 1.00, (60.0, 30.0))).scale)
    assert not scale_within_tolerance(estimate_similarity(ref, transformed(ref, 1.05, (0.0, 0.0))).scale)


def test_translation_is_reported_not_judged() -> None:
    ref = structured_image()
    est = estimate_similarity(ref, transformed(ref, 1.00, (60.0, 30.0)))
    assert est.tx_px == pytest.approx(60.0, abs=2.0) and est.ty_px == pytest.approx(30.0, abs=2.0)
    assert scale_within_tolerance(est.scale)


@pytest.mark.parametrize(
    ("measured", "image_scale", "expected"),
    [
        (3.6, None, "passed"),
        (3.6, 1.01, "passed"),
        (3.6, 1.05, "failed"),
        (4.2, None, "failed"),
        (2.9, 1.00, "failed"),
    ],
)
def test_decide_scale_check(measured: float, image_scale, expected: str) -> None:
    from safetrajai.collision.homography.scale_match import decide_scale_check, known_length_ok

    assert decide_scale_check(known_length_ok(measured, 3.5), image_scale) == expected


def test_failed_image_match_fails_scale_check() -> None:
    from safetrajai.collision.homography.scale_match import decide_scale_check

    assert decide_scale_check(True, None, image_match_failed=True) == "failed"


def test_unrelated_images_raise() -> None:
    from safetrajai.collision.homography.scale_match import ScaleMatchError

    flat = np.full((1280, 1280, 3), 120, np.uint8)
    with pytest.raises(ScaleMatchError):
        estimate_similarity(structured_image(), flat)
