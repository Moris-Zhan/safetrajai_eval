"""Audit failure rates (spec reid-dataset: 抽查不合格率決定自動排除結果能否進入資料集版本)."""

import importlib.util

from safetrajai.paths import PROJECT_ROOT


def load():
    spec = importlib.util.spec_from_file_location("summarize_audit", PROJECT_ROOT / "scripts/reid_data/summarize_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(camera: str, cls: str, n: int, purpose: str = "camera_audit", prefix: str = "") -> list[dict]:
    return [{"class": cls, "identity": f"{prefix or camera}_{cls}{i}", "camera": camera, "purpose": purpose,
             "reviewed": "False"} for i in range(n)]


def removed(identities: list[str], cls: str, image: str = "") -> list[dict]:
    return [{"class": cls, "identity": i, "image": image, "action": "remove"} for i in identities]


def test_single_crop_problems_do_not_count_as_failures() -> None:
    audit = sample("C000011", "car", 15)
    confirmed = removed([f"C000011_car{i}" for i in range(4)], "car", image="000001.jpg")

    rates = {(r["scope"], r["class"]): r for r in load().failure_rates(audit, confirmed, major_cameras=set())}
    assert rates[("C000011", "all")]["failures"] == 0 and not rates[("C000011", "all")]["over_threshold"]


def test_minor_camera_over_threshold_and_bus_person_ignored() -> None:
    audit = sample("C000011", "car", 15) + sample("C000054", "bus", 20, "bus_audit") + sample("C000054", "person", 5, "person_review")
    confirmed = (removed([f"C000011_car{i}" for i in range(3)], "car")
                 + removed([f"C000054_bus{i}" for i in range(10)], "bus")
                 + removed([f"C000054_person{i}" for i in range(5)], "person"))

    rates = load().failure_rates(audit, confirmed, major_cameras=set())
    assert [(r["scope"], r["class"], r["failures"], r["sampled"], r["over_threshold"]) for r in rates] == [
        ("C000011", "all", 3, 15, True)]


def test_major_camera_reports_car_and_scooter_separately() -> None:
    audit = sample("C000054", "car", 33) + sample("C000054", "scooter", 25) + sample("C000054", "truck", 2)
    confirmed = removed([f"C000054_scooter{i}" for i in range(4)], "scooter") + removed(["C000054_truck0"], "truck")

    rates = {(r["scope"], r["class"]): r for r in load().failure_rates(audit, confirmed, major_cameras={"C000054"})}
    assert rates[("C000054", "all")]["failures"] == 5 and not rates[("C000054", "all")]["over_threshold"]
    assert rates[("C000054", "scooter")]["over_threshold"] and not rates[("C000054", "car")]["over_threshold"]
    assert ("C000054", "truck") not in rates
