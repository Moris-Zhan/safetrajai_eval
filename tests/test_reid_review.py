"""Review page selection and exclusion list merging (spec reid-dataset, design D7, ADR 0006)."""

import csv
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from safetrajai.paths import PROJECT_ROOT
from safetrajai.reid.dataset_index import INDEX_FIELDS, index_path, write_index

SCRIPTS = PROJECT_ROOT / "scripts" / "reid_data"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_crops(root: Path, spec: list[tuple[str, str, int]], camera: str = "C000054") -> list[dict]:
    rows = []
    for cls, ident, n in spec:
        for k in range(n):
            rel = f"{cls}/{ident}/{k:06d}.jpg"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / rel), np.zeros((40, 30, 3), np.uint8))
            rows.append(dict(zip(INDEX_FIELDS, [rel, ident, cls, camera, "s", "20250623", k, k * 0.5, 0, 0, 30, 40, 0.9])))
    write_index(index_path(root, f"s_{camera}"), rows)
    return rows


def write_flags(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def consistency_row(cls: str, identity: str, n: int, flagged: bool = False) -> dict:
    return {"class": cls, "identity": identity, "n_crops": n, "agreement_rate": 1.0,
            "coarse_agreement_rate": 0.2 if flagged else 1.0, "flagged": flagged}


def motion_row(cls: str, identity: str, n: int, static: bool = False, jump: bool = False) -> dict:
    return {"class": cls, "identity": identity, "n_crops": n, "duration_s": 10.0, "displacement": 0.1,
            "max_velocity_change": 0.0, "static": static, "motion_jump": jump, "flagged": static or jump}


def run_review(tmp_path: Path, monkeypatch, extra_args: list[str] = ()) -> str:
    out = tmp_path / "review"
    module = load("generate_identity_review")
    monkeypatch.setattr("sys.argv", ["review", "--crop-root", str(tmp_path / "crops"),
                                     "--reports-dir", str(tmp_path / "reports"), "--out-dir", str(out), *extra_args])
    module.main()
    return "".join(p.read_text(encoding="utf-8") for p in sorted(out.glob("*.html")))


def test_review_covers_flagged_and_all_bus_with_capped_thumbnails(tmp_path: Path, monkeypatch) -> None:
    crops = tmp_path / "crops"
    build_crops(crops, [("car", "flagged", 20), ("car", "clean", 5), ("bus", "b1", 5)])
    reports = tmp_path / "reports"
    write_flags(reports / "class_consistency.csv", [
        consistency_row("car", "flagged", 20, flagged=True), consistency_row("car", "clean", 5), consistency_row("bus", "b1", 5)])
    write_flags(reports / "appearance_outliers.csv", [
        {"class": "car", "identity": "clean", "n_crops": 5, "n_outliers": 0, "outlier_fraction": 0, "flagged": False}])
    before = sorted(p.as_posix() for p in crops.rglob("*"))

    pages = run_review(tmp_path, monkeypatch, ["--audit-per-camera", "0"])

    assert 'data-identity="flagged"' in pages and 'data-identity="b1"' in pages
    assert 'data-identity="clean"' not in pages
    assert len([p for p in (tmp_path / "review" / "assets").glob("car__flagged__*.jpg")]) == 12
    assert sorted(p.as_posix() for p in crops.rglob("*")) == before


def test_review_skips_identities_below_min_crops(tmp_path: Path, monkeypatch) -> None:
    build_crops(tmp_path / "crops", [("car", "short", 3), ("bus", "short_bus", 3), ("car", "long", 4)])
    write_flags(tmp_path / "reports" / "class_consistency.csv", [
        consistency_row("car", "short", 3, flagged=True), consistency_row("bus", "short_bus", 3),
        consistency_row("car", "long", 4, flagged=True)])

    pages = run_review(tmp_path, monkeypatch, ["--audit-per-camera", "0"])

    assert 'data-identity="long"' in pages
    assert 'data-identity="short"' not in pages and 'data-identity="short_bus"' not in pages


def test_review_pages_are_grouped_by_reason_with_static_first(tmp_path: Path, monkeypatch) -> None:
    build_crops(tmp_path / "crops", [("car", "sign", 6), ("scooter", "jumpy", 6)])
    write_flags(tmp_path / "reports" / "motion_flags.csv", [
        motion_row("car", "sign", 6, static=True, jump=True), motion_row("scooter", "jumpy", 6, jump=True)])

    run_review(tmp_path, monkeypatch, ["--audit-per-camera", "0"])

    out = tmp_path / "review"
    static_page = (out / "identity_review_static_page1.html").read_text(encoding="utf-8")
    jump_page = (out / "identity_review_motion_jump_page1.html").read_text(encoding="utf-8")
    assert 'data-identity="sign"' in static_page and 'data-identity="sign"' not in jump_page
    assert 'data-identity="jumpy"' in jump_page
    assert "select-all" in static_page


def test_random_audit_is_per_camera_and_reproducible(tmp_path: Path, monkeypatch) -> None:
    build_crops(tmp_path / "crops", [("car", f"a{i}", 4) for i in range(10)], camera="C000011")
    build_crops(tmp_path / "crops", [("car", f"b{i}", 4) for i in range(10)], camera="C000012")
    (tmp_path / "reports").mkdir()

    first = run_review(tmp_path, monkeypatch, ["--audit-per-camera", "3", "--seed", "7"])
    second = run_review(tmp_path, monkeypatch, ["--audit-per-camera", "3", "--seed", "7"])

    audited = re.findall(r'data-identity="([ab]\d)"', first)
    assert first == second
    assert sum(i.startswith("a") for i in audited) == 3 and sum(i.startswith("b") for i in audited) == 3


def test_apply_review_exclusions_merges_without_moving_files(tmp_path: Path) -> None:
    crops = tmp_path / "crops"
    build_crops(crops, [("car", "a", 4)])
    before = sorted(p.as_posix() for p in crops.rglob("*"))
    confirm = tmp_path / "confirm.csv"
    write_flags(confirm, [{"class": "car", "identity": "a", "image": "", "action": "remove"},
                          {"class": "car", "identity": "a", "image": "", "action": "remove"}])
    exclusions = tmp_path / "exclusions.csv"
    cmd = [sys.executable, str(SCRIPTS / "apply_review_exclusions.py"), "--confirm-csv", str(confirm), "--exclusions", str(exclusions)]

    subprocess.run(cmd, check=True, capture_output=True)
    assert not exclusions.exists()
    subprocess.run(cmd + ["--apply"], check=True, capture_output=True)
    subprocess.run(cmd + ["--apply"], check=True, capture_output=True)

    with exclusions.open(encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == [{"class": "car", "identity": "a", "image": "", "action": "remove"}]
    assert sorted(p.as_posix() for p in crops.rglob("*")) == before


def index(cls: str, identity: str, camera: str, n: int = 4) -> list[dict]:
    return [{"class": cls, "identity": identity, "camera": camera, "time_s": str(k * 0.5),
             "crop_path": f"{cls}/{identity}/{k:06d}.jpg"} for k in range(n)]


def audit_population() -> list[dict]:
    rows = []
    rows += [r for i in range(200) for r in index("car", f"A_car{i}", "C000054")]
    rows += [r for i in range(3) for r in index("truck", f"A_truck{i}", "C000054")]
    rows += [r for i in range(25) for r in index("bus", f"A_bus{i}", "C000054")]
    rows += [r for i in range(4) for r in index("person", f"A_person{i}", "C000054")]
    rows += [r for i in range(9) for r in index("car", f"B_car{i}", "C000085")]
    rows += index("car", "A_short", "C000054", n=3)
    return rows


def select(auto: list[dict] = (), seed: int = 0) -> list[dict]:
    module = load("generate_identity_review")
    return module.select_audit(audit_population(), list(auto), [], min_crops=4, seed=seed)


def test_audit_skips_auto_excluded_and_short_identities() -> None:
    picks = select(auto=[{"class": "car", "identity": f"A_car{i}", "image": "", "action": "remove"} for i in range(200)])

    ids = {p["identity"] for p in picks}
    assert not any(i.startswith("A_car") for i in ids)
    assert "A_short" not in ids


def test_major_camera_is_stratified_and_minor_camera_taken_whole() -> None:
    picks = select()
    camera_audit = [p for p in picks if p["purpose"] == "camera_audit"]
    major = [p for p in camera_audit if p["camera"] == "C000054"]
    minor = [p for p in camera_audit if p["camera"] == "C000085"]

    assert len(major) == 60 and sum(p["class"] == "truck" for p in major) >= 2
    assert sorted(p["identity"] for p in minor) == sorted(f"B_car{i}" for i in range(9))


def test_bus_is_sampled_and_person_reviewed_in_full() -> None:
    picks = select()

    assert sum(p["purpose"] == "bus_audit" for p in picks) == 20
    assert sorted(p["identity"] for p in picks if p["purpose"] == "person_review") == [f"A_person{i}" for i in range(4)]
    assert len({(p["class"], p["identity"]) for p in picks}) == len(picks)


def test_audit_selection_is_reproducible_and_seed_dependent() -> None:
    assert select(seed=3) == select(seed=3)
    assert select(seed=3) != select(seed=4)


def test_audit_only_writes_pages_and_sample_without_excluded_crops(tmp_path: Path, monkeypatch) -> None:
    crops = tmp_path / "crops"
    build_crops(crops, [("car", "kept", 5), ("car", "gone", 5), ("person", "p1", 4)])
    (tmp_path / "reports").mkdir()
    auto = tmp_path / "auto.csv"
    auto.write_text("class,identity,image,action,rule\ncar,gone,,remove,static\n"
                    "car,kept,000002.jpg,remove,appearance_outlier\n", encoding="utf-8")
    manual = tmp_path / "manual.csv"
    manual.write_text("class,identity,image,action\nperson,p1,000000.jpg,remove\n", encoding="utf-8")

    pages = run_review(tmp_path, monkeypatch, ["--audit-only", "--auto-exclusions", str(auto),
                                               "--exclusions", str(manual)])

    out = tmp_path / "review"
    with (out / "audit_sample.csv").open(encoding="utf-8") as f:
        sample = {(r["identity"], r["purpose"], r["reviewed"]) for r in csv.DictReader(f)}
    assert sample == {("kept", "camera_audit", "False"), ("p1", "person_review", "True")}
    assert 'data-identity="gone"' not in pages and "已人工審查" in pages
    assert not list((out / "assets").glob("car__kept__000002.jpg"))
