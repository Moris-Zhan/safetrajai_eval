"""conflict_analysis.blind_area and object_width_m: off by default, validated, kept out of summary params."""

import pytest

from safetrajai.collision.conflict.config import BlindAreaConfig, ConflictConfig, load_conflict_config

DEFAULT_WIDTHS = {"person": 0.5, "car": 1.8, "scooter": 0.8, "small_truck": 2.0, "bus": 2.5, "truck": 2.5}


def test_blind_area_is_off_by_default() -> None:
    assert load_conflict_config(None).blind_area == BlindAreaConfig()
    assert load_conflict_config({"enabled": True}).blind_area.enabled is False
    assert ConflictConfig().blind_area.enabled is False


def test_default_occluders_are_the_four_large_classes() -> None:
    assert BlindAreaConfig().occluder_classes == ("car", "small_truck", "bus", "truck")


def test_true_turns_it_on_and_a_block_sets_the_occluders() -> None:
    assert load_conflict_config({"blind_area": True}).blind_area == BlindAreaConfig(enabled=True)

    cfg = load_conflict_config({"blind_area": {"enabled": True, "occluder_classes": ["bus", "truck"]}})

    assert cfg.blind_area == BlindAreaConfig(enabled=True, occluder_classes=("bus", "truck"))


@pytest.mark.parametrize("value, message", [
    ("yes", "blind_area"),
    ({"enabled": True, "height_m": 1.5}, "height_m"),
    ({"enabled": "yes"}, "enabled"),
    ({"occluder_classes": ["car", "tank"]}, "tank"),
    ({"occluder_classes": []}, "occluder_classes"),
    ({"occluder_classes": "car"}, "occluder_classes"),
])
def test_bad_blind_area_values_are_rejected_naming_the_field(value, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_conflict_config({"blind_area": value})


def test_object_width_defaults_cover_all_six_classes() -> None:
    widths = ConflictConfig().object_width_m

    assert dict(widths) == DEFAULT_WIDTHS
    assert widths["car"] == 1.8 and widths["small_truck"] == 2.0 and widths["bus"] == 2.5 and widths["truck"] == 2.5


def test_object_width_must_name_every_class_with_positive_numbers() -> None:
    with pytest.raises(ValueError, match="object_width_m"):
        load_conflict_config({"object_width_m": {"car": 1.8}})
    with pytest.raises(ValueError, match="object_width_m"):
        load_conflict_config({"object_width_m": {**DEFAULT_WIDTHS, "bus": 0}})

    assert load_conflict_config({"object_width_m": {**DEFAULT_WIDTHS, "bus": 3}}).object_width_m["bus"] == 3.0


def test_summary_params_do_not_change() -> None:
    from safetrajai.collision.conflict import _params

    for block in ({"enabled": True}, {"enabled": True, "blind_area": True}):
        params = _params(load_conflict_config(block))
        assert "blind_area" not in params and "object_width_m" not in params
