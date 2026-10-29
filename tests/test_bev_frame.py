"""Per-frame drawing of the bird's-eye video (hand-placed tracks on a blank base map)."""

import numpy as np

from safetrajai.collision.conflict.bev import CLASS_COLORS_BGR, BevTrack, bev_scale, draw_bev_frame

BLANK = np.zeros((300, 300, 3), dtype=np.uint8)


def track(class_name="car", track_id=1, frames=(1, 2, 3), xy=((120, 200), (160, 200), (200, 200)), speed=(np.nan,) * 3):
    return BevTrack(class_name, track_id, np.array(frames), np.array(xy, dtype=float), np.array(speed, dtype=float))


def draw(tracks, frame, scale=1.0, tail_frames=10, image=BLANK):
    return draw_bev_frame(image, scale, frame, tracks, tail_frames)


def test_every_class_has_its_own_colour() -> None:
    assert len(set(CLASS_COLORS_BGR.values())) == len(CLASS_COLORS_BGR) == 6


def test_scale_never_enlarges_and_fits_the_longest_side() -> None:
    assert bev_scale((1280, 1280), 960) == 0.75
    assert bev_scale((1280, 640), 640) == 0.5
    assert bev_scale((400, 300), 960) == 1.0


def test_current_position_trail_and_class_colour() -> None:
    image = draw([track()], frame=3)

    assert tuple(image[200, 200]) == CLASS_COLORS_BGR["car"]          # current position
    assert tuple(image[200, 180]) == CLASS_COLORS_BGR["car"]          # on the trail between frame 2 and 3
    assert tuple(image[200, 140]) == CLASS_COLORS_BGR["car"]          # on the trail between frame 1 and 2


def test_the_tail_only_reaches_back_tail_frames() -> None:
    image = draw([track()], frame=3, tail_frames=1)

    assert tuple(image[200, 180]) == CLASS_COLORS_BGR["car"]          # frame 2 -> 3 is inside the tail
    assert tuple(image[200, 140]) == (0, 0, 0)                        # frame 1 -> 2 is outside


def test_a_track_is_not_drawn_outside_its_frames() -> None:
    quiet = draw([], frame=9)

    assert np.array_equal(draw([track()], frame=9), quiet)
    assert np.array_equal(draw([track(frames=(5, 6, 7))], frame=3), draw([], frame=3))


def test_coordinates_are_scaled_with_the_image() -> None:
    small = np.zeros((150, 150, 3), dtype=np.uint8)

    image = draw([track(xy=((100, 200), (150, 200), (240, 160)))], frame=3, scale=0.5, image=small)

    assert tuple(image[80, 120]) == CLASS_COLORS_BGR["car"]
    assert tuple(image[100, 60]) == CLASS_COLORS_BGR["car"]          # trail segment (50,100)-(75,100) after scaling


def test_speed_is_written_only_when_known() -> None:
    known = draw([track(speed=(np.nan, np.nan, 36.0))], frame=3)
    unknown = draw([track()], frame=3)

    assert not np.array_equal(known, unknown)
    assert np.array_equal(known[:, :190], unknown[:, :190])           # text sits right of the marker, nothing else differs


def test_input_image_is_not_modified_and_the_legend_stays_in_the_corner() -> None:
    before = BLANK.copy()

    image = draw([], frame=1)

    assert np.array_equal(BLANK, before)
    assert image[:100, :100].any()                                    # legend, top-left
    assert not image[100:, 100:].any()


def _two_cars():
    upper = track(track_id=1)                                                 # y = 200
    lower = track(track_id=2, xy=((120, 250), (160, 250), (200, 250)))        # y = 250
    return [upper, lower]


def test_event_link_is_drawn_only_between_its_start_and_end_frame() -> None:
    from safetrajai.collision.conflict.bev import BevEvent

    cars = _two_cars()
    event = BevEvent(start_frame=2, end_frame=3, i=0, j=1)

    with_link = draw_bev_frame(BLANK, 1.0, 3, cars, 10, events=[event])
    before = draw_bev_frame(BLANK, 1.0, 1, cars, 10, events=[event])

    assert tuple(with_link[225, 200]) == (0, 0, 255)                          # between the two positions at frame 3
    assert np.array_equal(before, draw_bev_frame(BLANK, 1.0, 1, cars, 10))    # frame 1 is before the event
    assert np.array_equal(draw_bev_frame(BLANK, 1.0, 3, cars, 10, events=[BevEvent(1, 2, 0, 1)]),
                          draw_bev_frame(BLANK, 1.0, 3, cars, 10))             # frame 3 is after it


def _counting():
    from safetrajai.collision.conflict.bev import BevCounting, BevCountingLine
    from safetrajai.collision.counting.crossing import Crossing

    line = BevCountingLine("gate", (100.0, 100.0), (250.0, 100.0), "down", "up")
    crossings = (Crossing("gate", "forward", "car", 1, 2), Crossing("gate", "forward", "car", 2, 3),
                 Crossing("gate", "backward", "car", 3, 3))
    return BevCounting((line,), crossings)


def test_cumulative_counts_follow_the_frames() -> None:
    from safetrajai.collision.conflict.bev import cumulative_counts

    counting = _counting()

    assert cumulative_counts(counting, 1) == {("gate", "forward"): 0, ("gate", "backward"): 0}
    assert cumulative_counts(counting, 2) == {("gate", "forward"): 1, ("gate", "backward"): 0}
    assert cumulative_counts(counting, 3) == {("gate", "forward"): 2, ("gate", "backward"): 1}


def test_counting_lines_and_panel_are_drawn_only_with_counting() -> None:
    plain = draw_bev_frame(BLANK, 1.0, 1, [], 10)
    first = draw_bev_frame(BLANK, 1.0, 1, [], 10, counting=_counting())
    later = draw_bev_frame(BLANK, 1.0, 3, [], 10, counting=_counting())

    assert tuple(first[100, 200]) == (0, 255, 255)                            # the line itself, yellow
    assert not np.array_equal(first[:60, 150:], plain[:60, 150:])             # the panel, top-right
    assert not np.array_equal(first[:60, 150:], later[:60, 150:])             # its numbers changed
    assert np.array_equal(first[100:, :], later[100:, :])                      # nothing else moved
    assert np.array_equal(draw_bev_frame(BLANK, 1.0, 1, [], 10, counting=None), plain)


def test_counting_line_is_scaled_with_the_image() -> None:
    small = np.zeros((150, 150, 3), dtype=np.uint8)

    image = draw_bev_frame(small, 0.5, 1, [], 10, counting=_counting())

    assert tuple(image[50, 100]) == (0, 255, 255)
