"""Counting-line crossings of synthetic tracks (hand-computed cases).

The line runs (0, 10) -> (10, 10): d = (10, 0), side = 10 * (py - 10), so points above the line
(py < 10, screen up) are on its left and points at or below it (py >= 10) on its right.
Moving down therefore crosses left -> right, i.e. "forward".
"""

import numpy as np

from safetrajai.collision.counting.crossing import CountingTrack, find_crossings, side_of
from safetrajai.collision.counting.lines import CountingLine

LINE = CountingLine("gate", (0.0, 10.0), (10.0, 10.0), "down", "up")


def track(points, frames=None, class_name="car", track_id=1) -> CountingTrack:
    frames = frames if frames is not None else list(range(1, len(points) + 1))
    return CountingTrack(class_name, track_id, np.array(frames), np.array(points, dtype=float))


def summary(crossings):
    return [(c.line, c.direction, c.class_name, c.track_id, c.frame) for c in crossings]


def test_side_is_negative_on_the_left_of_the_drawing_direction() -> None:
    assert side_of(LINE, (5.0, 5.0)) < 0
    assert side_of(LINE, (5.0, 15.0)) > 0
    assert side_of(LINE, (5.0, 10.0)) == 0


def test_moving_down_is_forward_and_up_is_backward() -> None:
    down = track([(5, 5), (5, 15)], track_id=1)
    up = track([(5, 15), (5, 5)], track_id=2)

    assert summary(find_crossings([down, up], [LINE])) == [
        ("gate", "forward", "car", 1, 2), ("gate", "backward", "car", 2, 2)]


def test_a_path_that_passes_beside_the_segment_is_not_a_crossing() -> None:
    assert find_crossings([track([(15, 5), (15, 15)])], [LINE]) == ()
    assert find_crossings([track([(-3, 5), (-3, 15)])], [LINE]) == ()


def test_staying_on_one_side_never_crosses() -> None:
    assert find_crossings([track([(5, 1), (5, 4), (5, 9)])], [LINE]) == ()
    assert find_crossings([track([(5, 11), (5, 20)])], [LINE]) == ()
    assert find_crossings([track([(5, 5)])], [LINE]) == ()


def test_jitter_over_the_line_counts_each_direction_once() -> None:
    jitter = track([(5, 5), (5, 15), (5, 5), (5, 15), (5, 5)])

    assert summary(find_crossings([jitter], [LINE])) == [
        ("gate", "forward", "car", 1, 2), ("gate", "backward", "car", 1, 3)]


def test_touching_the_line_counts_once_when_the_track_carries_on() -> None:
    assert summary(find_crossings([track([(5, 5), (5, 10), (5, 15)])], [LINE])) == [("gate", "forward", "car", 1, 2)]


def test_track_ids_are_unique_only_within_a_class() -> None:
    car = track([(5, 5), (5, 15)], class_name="car", track_id=7)
    scooter = track([(5, 5), (5, 15)], class_name="scooter", track_id=7)

    assert summary(find_crossings([car, scooter], [LINE])) == [
        ("gate", "forward", "car", 7, 2), ("gate", "forward", "scooter", 7, 2)]


def test_every_line_is_evaluated_on_its_own() -> None:
    other = CountingLine("side", (0.0, 20.0), (10.0, 20.0))

    result = find_crossings([track([(5, 5), (5, 15), (5, 25)])], [LINE, other])

    assert summary(result) == [("gate", "forward", "car", 1, 2), ("side", "forward", "car", 1, 3)]


def test_frame_gaps_are_not_treated_specially() -> None:
    assert summary(find_crossings([track([(5, 5), (5, 15)], frames=[3, 40])], [LINE])) == [
        ("gate", "forward", "car", 1, 40)]


def test_crossings_are_ordered_by_frame_then_line_then_class_then_id() -> None:
    late = track([(5, 5), (5, 15)], frames=[10, 11], track_id=1)
    early = track([(5, 5), (5, 15)], frames=[1, 2], track_id=2)

    assert [c.frame for c in find_crossings([late, early], [LINE])] == [2, 11]
