"""Contamination checks: flags only, single read per crop (spec reid-dataset, design D6)."""

import importlib.util
from pathlib import Path

import numpy as np

from safetrajai.paths import PROJECT_ROOT
from safetrajai.reid.contamination import appearance_outliers, class_consistency, motion_flags


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def make_rows(spec: list[tuple[str, str, int]]) -> list[dict]:
    return [{"class": c, "identity": i, "crop_path": f"{c}/{i}/{k:06d}.jpg"} for c, i, n in spec for k in range(n)]


def track_rows(identity: str, centers: list[tuple[float, float]], interval_s: float = 1.0, side: float = 40.0) -> list[dict]:
    return [
        {"class": "car", "identity": identity, "crop_path": f"car/{identity}/{k:06d}.jpg", "time_s": str(k * interval_s),
         "x1": str(x - side / 2), "y1": str(y - side / 2), "x2": str(x + side / 2), "y2": str(y + side / 2)}
        for k, (x, y) in enumerate(centers)
    ]


def test_class_consistency_ignores_four_wheel_subclass_confusion() -> None:
    rng = np.random.default_rng(2)
    car_dir, truck_dir, scooter_dir = np.eye(8)[0], np.eye(8)[1], np.eye(8)[2]
    rows = make_rows([("car", f"c{i}", 5) for i in range(4)] + [("small_truck", f"t{i}", 5) for i in range(4)]
                     + [("scooter", f"s{i}", 5) for i in range(4)] + [("car", "looks_like_truck", 5)])
    base = [car_dir] * 20 + [truck_dir] * 20 + [scooter_dir] * 20 + [truck_dir] * 5
    feats = unit(np.array(base) + rng.normal(0, 0.05, (65, 8)))

    records, predicted = class_consistency(rows, feats)
    record = next(r for r in records if r["identity"] == "looks_like_truck")
    assert record["agreement_rate"] < 0.5
    assert not record["flagged"]
    assert predicted[-1] == "small_truck"


def test_motion_flags_mark_static_identity() -> None:
    rows = track_rows("sign", [(100.0 + k * 0.5, 100.0) for k in range(10)])
    rows += track_rows("moving", [(100.0 + k * 20.0, 100.0) for k in range(10)])

    flags = {r["identity"]: r for r in motion_flags(rows)}
    assert flags["sign"]["static"] and not flags["sign"]["motion_jump"]
    assert not flags["moving"]["static"]


def test_motion_flags_require_minimum_duration_for_static() -> None:
    rows = track_rows("brief_stop", [(100.0, 100.0)] * 4)

    assert not motion_flags(rows)[0]["static"]


def test_motion_flags_mark_sudden_velocity_change_but_not_constant_motion() -> None:
    steady = [(100.0 + k * 20.0, 100.0) for k in range(6)]
    jumped = steady[:3] + [(300.0, 300.0), (320.0, 300.0), (340.0, 300.0)]
    rows = track_rows("steady", steady) + track_rows("jumped", jumped)

    flags = {r["identity"]: r for r in motion_flags(rows)}
    assert flags["jumped"]["motion_jump"]
    assert not flags["steady"]["motion_jump"]


def test_class_consistency_flags_identity_that_looks_like_another_class() -> None:
    rng = np.random.default_rng(0)
    car_dir, person_dir = np.eye(8)[0], np.eye(8)[1]
    rows = make_rows([("car", f"c{i}", 5) for i in range(4)] + [("person", f"p{i}", 5) for i in range(4)] + [("car", "wrong", 5)])
    base = [car_dir] * 20 + [person_dir] * 20 + [person_dir] * 5
    feats = unit(np.array(base) + rng.normal(0, 0.05, (45, 8)))

    records, _ = class_consistency(rows, feats)
    flagged = {r["identity"] for r in records if r["flagged"]}
    assert flagged == {"wrong"}


def test_appearance_outliers_flag_mixed_identity() -> None:
    rng = np.random.default_rng(1)
    rows = make_rows([("bus", f"b{i}", 10) for i in range(10)] + [("bus", "mixed", 10)])
    clean = [np.eye(16)[i % 8] for i in range(10) for _ in range(10)]
    mixed = [np.eye(16)[12]] * 5 + [np.eye(16)[13]] * 5
    feats = unit(np.array(clean + mixed) + rng.normal(0, 0.02, (110, 16)))

    records, _, _ = appearance_outliers(rows, feats, percentile=90)
    assert "mixed" in {r["identity"] for r in records if r["flagged"]}


def test_detect_script_reads_each_crop_once_and_changes_no_files(tmp_path: Path, monkeypatch) -> None:
    import cv2
    import timm
    import torch

    from safetrajai.reid.dataset_index import INDEX_FIELDS, index_path, write_index

    rows = []
    for ident in ("a", "b"):
        for k in range(3):
            rel = f"car/{ident}/{k:06d}.jpg"
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tmp_path / rel), np.full((40, 30, 3), k * 40, np.uint8))
            rows.append(dict(zip(INDEX_FIELDS, [rel, ident, "car", "C000054", "s", "20250623", k, k, 0, 0, 30, 40, 0.9])))
    write_index(index_path(tmp_path, "s"), rows)
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())

    class DummyModel(torch.nn.Module):
        pretrained_cfg = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}

        def forward(self, x):
            return x.mean(dim=(2, 3))

    monkeypatch.setattr(timm, "create_model", lambda *a, **k: DummyModel())
    reads = []
    real_imread = cv2.imread
    monkeypatch.setattr(cv2, "imread", lambda p, *a: reads.append(p) or real_imread(p, *a))

    spec = importlib.util.spec_from_file_location("detect", PROJECT_ROOT / "scripts/reid_data/detect_contamination.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr("sys.argv", ["detect", "--crop-root", str(tmp_path), "--out-dir", str(tmp_path / "reports")])
    module.main()

    after = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file() and "reports" not in p.parts)
    assert after == before
    assert sorted(reads) == sorted(str(tmp_path / r["crop_path"]) for r in rows)
    assert {p.name for p in (tmp_path / "reports").iterdir()} == {
        "class_consistency.csv", "appearance_outliers.csv", "contamination_crops.csv", "motion_flags.csv"}
