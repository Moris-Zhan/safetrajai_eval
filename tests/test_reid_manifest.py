"""Scenarios from specs/reid-dataset: versioned manifest rules."""

import pytest

from safetrajai.paths import CONFIG_DIR
from safetrajai.reid.manifest import build_manifest, load_split_rules

RULES = load_split_rules(CONFIG_DIR / "reid_split.yaml")


def crops(identity: str, n: int, camera: str = "C000054", date: str = "20250623", cls: str = "car") -> list[dict]:
    return [
        {"crop_path": f"{cls}/{identity}/{i:06d}.jpg", "identity": identity, "class": cls,
         "camera": camera, "date": date, "time_s": f"{i * 0.5:.3f}"}
        for i in range(n)
    ]


def test_identity_crop_count_rules() -> None:
    manifest = build_manifest(crops("few", 3) + crops("many", 80), [], RULES, {"C000025"})
    identities = [r["identity"] for r in manifest]
    assert "few" not in identities
    assert identities.count("many") == 30


def test_exclusions_apply_before_min_count() -> None:
    exclusions = [{"class": "car", "identity": "a", "image": "000000.jpg", "action": "remove"},
                  {"class": "car", "identity": "b", "image": "", "action": "remove"}]
    manifest = build_manifest(crops("a", 4) + crops("b", 10), exclusions, RULES, {"C000025"})
    assert manifest == []


def test_split_by_camera_and_date() -> None:
    rows = crops("d23", 5) + crops("d29", 5, date="20250629") + crops("other", 5, camera="C000043", date="20250629")
    splits = {r["identity"]: r["split"] for r in build_manifest(rows, [], RULES, {"C000025"})}
    assert splits == {"d23": "train", "d29": "val", "other": "train"}


def test_eval_camera_is_rejected() -> None:
    with pytest.raises(ValueError, match="C000025"):
        build_manifest(crops("x", 5, camera="C000025"), [], RULES, {"C000025"})


def test_manifest_is_reproducible() -> None:
    rows = crops("b", 40) + crops("a", 12, cls="bus")
    assert build_manifest(rows, [], RULES, set()) == build_manifest(list(reversed(rows)), [], RULES, set())


def test_build_reid_split_cli_writes_version(tmp_path) -> None:
    import csv
    import subprocess
    import sys

    from safetrajai.paths import PROJECT_ROOT
    from safetrajai.reid.dataset_index import INDEX_FIELDS, index_path, write_index

    rows = [dict(r, sequence="s", frame=i, x1=0, y1=0, x2=40, y2=40, score="0.9")
            for i, r in enumerate(crops("s_track1", 6))]
    write_index(index_path(tmp_path / "crops", "s"), [{k: r[k] for k in INDEX_FIELDS} for r in rows])
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/reid_data/build_reid_split.py"), "--version", "t1",
           "--crop-root", str(tmp_path / "crops"), "--exclusions", str(tmp_path / "none.csv"),
           "--splits-root", str(tmp_path / "splits")]
    subprocess.run(cmd, check=True, capture_output=True)
    with (tmp_path / "splits" / "t1" / "manifest.csv").open(encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 6
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    assert not list((tmp_path / "splits").rglob("*.jpg"))
