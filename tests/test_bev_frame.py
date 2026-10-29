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
