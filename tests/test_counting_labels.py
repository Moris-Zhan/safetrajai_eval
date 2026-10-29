"""forward/backward labels derived from the line geometry (and the homography) instead of typed in.

The forward travel direction of a line p1 -> p2 with d = p2 - p1 is (-dy, dx) in image coordinates
(y down): a horizontal line drawn left to right has "down" as forward. With a homography the
direction is moved to the satellite plane (north up) and named by compass.
"""

import numpy as np
import pytest

from safetrajai.collision.counting.labels import ResolvedLabels, resolve_labels
from safetrajai.collision.counting.lines import CountingLine


def line(p1, p2, **labels) -> CountingLine:
    return CountingLine("gate", p1, p2, **labels)


@pytest.mark.parametrize("p1, p2, forward, backward", [
    ((0, 10), (10, 10), "down", "up"),                  # horizontal, left to right
    ((10, 10), (0, 10), "up", "down"),                  # horizontal, right to left
    ((10, 0), (10, 10), "left", "right"),               # vertical, top to bottom
    ((10, 10), (10, 0), "right", "left"),               # vertical, bottom to top
    ((0, 0), (10, 10), "down_left", "up_right"),        # diagonal
    ((0, 10), (10, 0), "down_right", "up_left"),
])
def test_screen_labels_without_a_homography(p1, p2, forward, backward) -> None:
    assert resolve_labels(line(p1, p2)) == ResolvedLabels(forward, backward, "screen", "screen")


@pytest.mark.parametrize("p1, p2, forward, backward", [
    ((0, 10), (10, 10), "southbound", "northbound"),
    ((10, 10), (0, 10), "northbound", "southbound"),
    ((10, 0), (10, 10), "westbound", "eastbound"),
    ((10, 10), (10, 0), "eastbound", "westbound"),
    ((0, 0), (10, 10), "southwestbound", "northeastbound"),
    ((0, 10), (10, 0), "southeastbound", "northwestbound"),
])
def test_compass_labels_with_an_identity_homography(p1, p2, forward, backward) -> None:
    assert resolve_labels(line(p1, p2), np.eye(3)) == ResolvedLabels(forward, backward, "compass", "compass")


def test_the_homography_turns_the_direction() -> None:
    rotate_quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # (x, y) -> (-y, x)
    mirror = np.diag([-1.0, 1.0, 1.0])                                                   # (x, y) -> (-x, y)

    assert resolve_labels(line((0, 10), (10, 10)), rotate_quarter).forward == "westbound"   # down -> west
    assert resolve_labels(line((10, 0), (10, 10)), mirror).forward == "eastbound"           # left -> east


def test_a_perspective_homography_uses_the_local_direction() -> None:
    perspective = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.001, 1.0]])

    assert resolve_labels(line((100, 100), (200, 100)), perspective).forward == "southbound"


def test_typed_labels_win_and_each_direction_is_decided_on_its_own() -> None:
    both = resolve_labels(line((0, 10), (10, 10), forward_label="in", backward_label="out"), np.eye(3))
    only_forward = resolve_labels(line((0, 10), (10, 10), forward_label="in"), np.eye(3))

    assert both == ResolvedLabels("in", "out", "manual", "manual")
    assert only_forward == ResolvedLabels("in", "northbound", "manual", "compass")
