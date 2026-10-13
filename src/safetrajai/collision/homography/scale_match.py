"""
Similarity (scale + rotation + translation) between an existing base map and a freshly
downloaded verification image of the same camera (spec `camera-homography`, design §8).

Only the scale is judged (1 ± 2%): the translation is reported but not judged, because the
OSM intersection centre may legitimately differ from the GPS used for the original download.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

SCALE_TOLERANCE = 0.02
MIN_GOOD_MATCHES = 20
RATIO_TEST = 0.75


class ScaleMatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SimilarityEstimate:
    scale: float
    rotation_deg: float
    tx_px: float
    ty_px: float
    matches: int
    inliers: int


def estimate_similarity(existing_bgr: np.ndarray, verify_bgr: np.ndarray, n_features: int = 5000) -> SimilarityEstimate:
    """Similarity mapping existing-map pixels to verification-image pixels."""
    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(existing_bgr, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(verify_bgr, cv2.COLOR_BGR2GRAY), None)
    if des1 is None or des2 is None or len(kp1) < MIN_GOOD_MATCHES or len(kp2) < MIN_GOOD_MATCHES:
        raise ScaleMatchError("特徵點不足，無法比對")

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]
    if len(good) < MIN_GOOD_MATCHES:
        raise ScaleMatchError(f"可靠的匹配只有 {len(good)} 組（至少需要 {MIN_GOOD_MATCHES}）")

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    cv2.setRNGSeed(0)
    M, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000, confidence=0.999,
    )
    inliers = int(mask.sum()) if mask is not None else 0
    if M is None or inliers < MIN_GOOD_MATCHES:
        raise ScaleMatchError(f"RANSAC 內點只有 {inliers} 組，無法估計相似變換")

    return SimilarityEstimate(
        scale=float(math.hypot(M[0, 0], M[1, 0])),
        rotation_deg=float(math.degrees(math.atan2(M[1, 0], M[0, 0]))),
        tx_px=float(M[0, 2]),
        ty_px=float(M[1, 2]),
        matches=len(good),
        inliers=inliers,
    )


def scale_within_tolerance(scale: float, tolerance: float = SCALE_TOLERANCE) -> bool:
    return abs(scale - 1.0) <= tolerance


KNOWN_LENGTH_TOLERANCE = 0.15


def known_length_ok(measured_m: float, reference_m: float, tolerance: float = KNOWN_LENGTH_TOLERANCE) -> bool:
    return abs(measured_m - reference_m) <= tolerance * reference_m


def decide_scale_check(length_ok: bool, image_scale: float | None, image_match_failed: bool = False) -> str:
    """`passed` only if the known-length check passes and, when an image match was run,
    its scale is within tolerance; `image_scale=None` means the image match was skipped.
    A match that was attempted but could not be estimated counts as a failure."""
    if not length_ok or image_match_failed:
        return "failed"
    if image_scale is not None and not scale_within_tolerance(image_scale):
        return "failed"
    return "passed"
