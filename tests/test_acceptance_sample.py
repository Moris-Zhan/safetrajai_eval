"""Appearance-stratified acceptance sample: representative choice and the weighted disagreement rate."""

import importlib.util

from safetrajai.paths import PROJECT_ROOT


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_acceptance_sample", "scripts/cvat_review/build_acceptance_sample.py")
audit = _load("serve_audit_for_test", "scripts/cvat_review/serve_audit.py")


def test_each_cluster_gets_one_question_weighted_by_cluster_size() -> None:
    items = build.build_items("car", [10, 11, 12, 13, 14], [0, 0, 1, 1, 1], answered=set(), seed=0)
    assert [(i["cluster"], i["weight"]) for i in items] == [(0, 2), (1, 3)]
    assert items[0]["track_id"] in (10, 11) and items[1]["track_id"] in (12, 13, 14)
    assert items[0]["item_id"] == f"acceptance-car-{items[0]['track_id']}"


def test_already_answered_member_is_preferred_as_representative() -> None:
    items = build.build_items("person", [1, 2, 3, 4], [0, 0, 0, 1], answered={3}, seed=7)
    assert items[0]["track_id"] == 3


def test_summary_rate_is_weighted_by_cluster_size() -> None:
    items = [
        {"item_id": "a", "kind": "acceptance_car", "claimed_final": "car", "weight": 90},
        {"item_id": "b", "kind": "acceptance_car", "claimed_final": "car", "weight": 10},
        {"item_id": "c", "kind": "acceptance_car", "claimed_final": "car", "weight": 50},
    ]
    summary = audit.summarize(items, {"a": "car", "b": "scooter", "c": "unsure"})
    car = summary["acceptance_car"]
    assert car["unsure"] == 1 and car["disagreed"] == 1
    assert abs(car["rate"] - 0.1) < 1e-9  # 10 of (90 + 10) decided weight; unsure is excluded
