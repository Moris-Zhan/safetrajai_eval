"""conflict_analysis.bev_video: off by default, `true` or a block with max_side_px and tail_s."""

import pytest

from safetrajai.collision.conflict.config import BevVideoConfig, ConflictConfig, load_conflict_config


def test_bev_video_is_off_by_default() -> None:
    assert load_conflict_config(None).bev_video == BevVideoConfig()
    assert load_conflict_config({"enabled": True}).bev_video == BevVideoConfig(enabled=False, max_side_px=960, tail_s=3.0)
    assert ConflictConfig().bev_video.enabled is False


def test_true_turns_it_on_with_default_parameters() -> None:
    assert load_conflict_config({"enabled": True, "bev_video": True}).bev_video == BevVideoConfig(enabled=True)


def test_a_block_sets_the_parameters() -> None:
    cfg = load_conflict_config({"bev_video": {"enabled": True, "max_side_px": 1280, "tail_s": 5}})

    assert cfg.bev_video == BevVideoConfig(enabled=True, max_side_px=1280, tail_s=5.0)


@pytest.mark.parametrize("value, message", [
    ("yes", "bev_video"),
    ({"enabled": True, "fps": 10}, "fps"),
    ({"enabled": "yes"}, "enabled"),
    ({"max_side_px": 0}, "max_side_px"),
    ({"max_side_px": 96.5}, "max_side_px"),
    ({"max_side_px": True}, "max_side_px"),
    ({"tail_s": 0}, "tail_s"),
    ({"tail_s": -1}, "tail_s"),
])
def test_bad_values_are_rejected_naming_the_field(value, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_conflict_config({"bev_video": value})


def test_summary_params_do_not_change_when_the_option_is_present() -> None:
    from safetrajai.collision.conflict import _params

    assert "bev_video" not in _params(load_conflict_config({"enabled": True}))
    assert "bev_video" not in _params(load_conflict_config({"enabled": True, "bev_video": True}))
