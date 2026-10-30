"""Blind-area geometry on the satellite pixel grid: body rectangle, shadow behind it, rasterised area.

Hand-computed case: viewpoint (50, 200), a 20 x 20 body centred at (150, 200). The shadow edges pass through
the near corners (140, 190) and (140, 210), so at x the shadow is 20 * (x - 50) / 90 wide. On a 400 x 400
canvas its area from x = 140 is (20 / 90) * (350^2 - 90^2) / 2 = 12711, and the body itself (400) is not counted.
"""

import numpy as np
import pytest

from safetrajai.collision.conflict.blind_area import blind_area_px, body_rectangle, polygon_mask, shadow_polygon

SHAPE = (400, 400)
VIEWPOINT = (50.0, 200.0)
FULL = np.ones(SHAPE, dtype=bool)
SINGLE_SHADOW = (20 / 90) * (350**2 - 90**2) / 2 - 400   # 12311.1
TOLERANCE = 0.04      # cv2.fillPoly also fills boundary pixels: about half a pixel too much along every edge


def body(center_x: float = 150.0, length: float = 20.0, width: float = 20.0, heading: float = 0.0):
    return body_rectangle((center_x, 200.0), heading, length, width)


def test_body_rectangle_follows_the_heading() -> None:
    along_x = body_rectangle((150, 200), 0.0, 20, 10)
    along_y = body_rectangle((150, 200), np.pi / 2, 20, 10)

    assert (along_x[:, 0].min(), along_x[:, 0].max(), along_x[:, 1].min(), along_x[:, 1].max()) == pytest.approx((140, 160, 195, 205))
    assert (along_y[:, 0].min(), along_y[:, 0].max(), along_y[:, 1].min(), along_y[:, 1].max()) == pytest.approx((145, 155, 190, 210))


def test_shadow_starts_at_the_near_corners_and_runs_away_from_the_viewpoint() -> None:
    shadow = shadow_polygon(VIEWPOINT, body(), reach=1000.0)

    assert shadow is not None
    near = {(140.0, 190.0), (140.0, 210.0)}
    assert near <= {tuple(round(v, 6) for v in point) for point in shadow}
    assert shadow[:, 0].min() == pytest.approx(140.0)
    assert shadow[:, 0].max() > 1000.0


@pytest.mark.parametrize("viewpoint", [(150.0, 200.0), (140.0, 200.0), (160.0, 210.0)])   # inside, on an edge, on a corner
def test_a_viewpoint_inside_or_on_the_body_gives_no_shadow(viewpoint) -> None:
    assert shadow_polygon(viewpoint, body(), reach=1000.0) is None


def test_the_shadow_area_matches_the_hand_computed_value_without_the_body() -> None:
    area, skipped = blind_area_px(VIEWPOINT, [body()], FULL)

    assert area == pytest.approx(SINGLE_SHADOW, rel=TOLERANCE)
    assert skipped == 0


def test_only_the_part_inside_the_roi_counts() -> None:
    roi = polygon_mask(np.array([[200, 0], [300, 0], [300, 399], [200, 399]], dtype=float), SHAPE)
    expected = (20 / 90) * (250**2 - 150**2) / 2        # 4444.4

    area, _ = blind_area_px(VIEWPOINT, [body()], roi)

    assert area == pytest.approx(expected, rel=TOLERANCE)


def test_roi_mask_area_matches_the_polygon() -> None:
    mask = polygon_mask(np.array([[100, 100], [200, 100], [200, 150], [100, 150]], dtype=float), SHAPE)

    assert mask.sum() == pytest.approx(100 * 50, rel=TOLERANCE)


def test_overlapping_shadows_are_counted_once_and_a_body_in_a_shadow_stays_blind() -> None:
    behind = body(center_x=250.0)       # sits inside the first shadow, so its own shadow adds nothing

    area, _ = blind_area_px(VIEWPOINT, [body(), behind], FULL)

    assert area == pytest.approx(SINGLE_SHADOW, rel=TOLERANCE)


def test_a_second_body_off_the_first_shadow_adds_its_own_area() -> None:
    aside = body_rectangle((150.0, 320.0), 0.0, 20, 20)     # far enough in y that the shadows do not touch

    first, _ = blind_area_px(VIEWPOINT, [body()], FULL)
    second, _ = blind_area_px(VIEWPOINT, [aside], FULL)
    both, _ = blind_area_px(VIEWPOINT, [body(), aside], FULL)

    assert second > 1000
    assert both == pytest.approx(first + second, rel=0.001)


def test_a_body_with_the_viewpoint_inside_is_skipped_and_counted() -> None:
    area, skipped = blind_area_px(VIEWPOINT, [body(), body_rectangle((50, 200), 0.0, 20, 20)], FULL)

    assert area == pytest.approx(SINGLE_SHADOW, rel=TOLERANCE)
    assert skipped == 1


def test_no_bodies_means_no_blind_area() -> None:
    assert blind_area_px(VIEWPOINT, [], FULL) == (0.0, 0)


def test_a_roi_that_is_only_the_body_has_no_blind_area() -> None:
    corners = body()
    roi = polygon_mask(corners, SHAPE)

    area, _ = blind_area_px(VIEWPOINT, [corners], roi)

    assert roi.sum() > 0
    assert area == 0.0


def test_a_roi_on_the_viewpoint_side_of_the_body_has_no_blind_area() -> None:
    roi = polygon_mask(np.array([[0, 0], [130, 0], [130, 399], [0, 399]], dtype=float), SHAPE)

    area, _ = blind_area_px(VIEWPOINT, [body()], roi)

    assert area == 0.0


def test_a_roi_beside_the_shadow_has_no_blind_area() -> None:
    roi = polygon_mask(np.array([[200, 260], [399, 260], [399, 399], [200, 399]], dtype=float), SHAPE)   # shadow is 80 px wide at x=400

    area, _ = blind_area_px(VIEWPOINT, [body()], roi)

    assert area == 0.0
