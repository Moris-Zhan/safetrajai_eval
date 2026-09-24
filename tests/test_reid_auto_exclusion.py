"""Auto exclusion rules (spec reid-dataset, change auto-reid-exclusions design D1-D3)."""

from safetrajai.reid.auto_exclusion import AUTO_EXCLUSION_FIELDS, build_auto_exclusions


def index_rows(cls: str, identity: str, n: int) -> list[dict]:
    return [{"class": cls, "identity": identity, "crop_path": f"{cls}/{identity}/{k:06d}.jpg"} for k in range(n)]


def motion(cls: str, identity: str, n: int, static: bool = False, jump: bool = False) -> dict:
    return {"class": cls, "identity": identity, "n_crops": str(n), "static": str(static), "motion_jump": str(jump)}


def crops(cls: str, identity: str, n: int, predicted: str | None = None, outliers: tuple[int, ...] = ()) -> list[dict]:
    return [{"crop_path": f"{cls}/{identity}/{k:06d}.jpg", "class": cls, "identity": identity,
             "predicted_class": predicted or cls, "is_outlier": str(k in outliers)} for k in range(n)]


def run(spec: list[tuple], min_crops: int = 4) -> list[dict]:
    idx, mot, crp = [], [], []
    for cls, identity, n, kwargs in spec:
        idx += index_rows(cls, identity, n)
        mot.append(motion(cls, identity, n, kwargs.get("static", False), kwargs.get("jump", False)))
        crp += crops(cls, identity, n, kwargs.get("predicted"), kwargs.get("outliers", ()))
    return build_auto_exclusions(idx, mot, crp, min_crops)


def test_static_identity_is_excluded_whole() -> None:
    rows = run([("car", "sign", 6, {"static": True}), ("car", "clean", 6, {})])

    assert rows == [{"class": "car", "identity": "sign", "image": "", "action": "remove", "rule": "static"}]
    assert tuple(rows[0]) == AUTO_EXCLUSION_FIELDS


def test_appearance_outlier_excludes_only_that_crop() -> None:
    rows = run([("car", "a", 20, {"outliers": (3,)})])

    assert rows == [{"class": "car", "identity": "a", "image": "000003.jpg", "action": "remove", "rule": "appearance_outlier"}]


def test_scooter_predicted_as_person_is_not_excluded() -> None:
    assert run([("scooter", "rider", 8, {"predicted": "person"})]) == []


def test_four_wheel_versus_two_wheel_mismatch_is_excluded_whole() -> None:
    rows = run([("car", "a", 8, {"predicted": "scooter"}), ("car", "b", 8, {"predicted": "small_truck"})])

    assert rows == [{"class": "car", "identity": "a", "image": "", "action": "remove", "rule": "class_group"}]


def test_identities_below_min_crops_are_ignored() -> None:
    assert run([("car", "short", 3, {"static": True, "outliers": (0,)})]) == []


def test_multiple_rules_merge_into_one_row_and_subsume_crop_rows() -> None:
    rows = run([("car", "a", 6, {"static": True, "jump": True, "predicted": "scooter", "outliers": (1,)})])

    assert rows == [{"class": "car", "identity": "a", "image": "", "action": "remove",
                     "rule": "static+motion_jump+class_group"}]


def test_output_is_sorted_and_deterministic() -> None:
    spec = [("scooter", "z", 5, {"jump": True}), ("car", "b", 5, {"outliers": (4, 0)}), ("car", "a", 5, {"static": True})]

    first, second = run(spec), run(list(reversed(spec)))
    assert first == second
    assert [(r["class"], r["identity"], r["image"]) for r in first] == [
        ("car", "a", ""), ("car", "b", "000000.jpg"), ("car", "b", "000004.jpg"), ("scooter", "z", "")]


def test_apply_script_is_deterministic_and_touches_nothing_else(tmp_path) -> None:
    import csv
    import subprocess
    import sys

    import cv2
    import numpy as np

    from safetrajai.paths import PROJECT_ROOT
    from safetrajai.reid.dataset_index import INDEX_FIELDS, index_path, write_index

    crop_root, reports = tmp_path / "crops", tmp_path / "reports"
    idx = []
    for identity, n in (("sign", 5), ("ok", 5)):
        for k in range(n):
            rel = f"car/{identity}/{k:06d}.jpg"
            (crop_root / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_root / rel), np.zeros((40, 30, 3), np.uint8))
            idx.append(dict(zip(INDEX_FIELDS, [rel, identity, "car", "C000054", "s", "20250623", k, k * 0.5, 0, 0, 30, 40, 0.9])))
    write_index(index_path(crop_root, "s"), idx)
    reports.mkdir()
    for name, rows in (("motion_flags.csv", [motion("car", "sign", 5, static=True), motion("car", "ok", 5)]),
                       ("contamination_crops.csv", crops("car", "sign", 5) + crops("car", "ok", 5, outliers=(2,)))):
        with (reports / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manual = tmp_path / "exclusions.csv"
    manual.write_text("class,identity,image,action\ncar,ok,000001.jpg,remove\n", encoding="utf-8")
    out = tmp_path / "auto_exclusions.csv"
    before = sorted(p.as_posix() for p in crop_root.rglob("*"))
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/reid_data/apply_auto_exclusions.py"),
           "--crop-root", str(crop_root), "--reports-dir", str(reports), "--out", str(out)]

    subprocess.run(cmd, check=True, capture_output=True)
    assert not out.exists()
    subprocess.run(cmd + ["--apply"], check=True, capture_output=True)
    first = out.read_bytes()
    subprocess.run(cmd + ["--apply"], check=True, capture_output=True)

    assert out.read_bytes() == first
    with out.open(encoding="utf-8") as f:
        assert [(r["identity"], r["image"], r["rule"]) for r in csv.DictReader(f)] == [
            ("ok", "000002.jpg", "appearance_outlier"), ("sign", "", "static")]
    assert manual.read_text(encoding="utf-8") == "class,identity,image,action\ncar,ok,000001.jpg,remove\n"
    assert sorted(p.as_posix() for p in crop_root.rglob("*")) == before
