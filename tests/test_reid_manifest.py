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
           "--auto-exclusions", str(tmp_path / "none_auto.csv"), "--splits-root", str(tmp_path / "splits")]
    subprocess.run(cmd, check=True, capture_output=True)
    with (tmp_path / "splits" / "t1" / "manifest.csv").open(encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 6
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    assert not list((tmp_path / "splits").rglob("*.jpg"))


def test_build_reid_split_applies_both_lists_and_excluded_cameras(tmp_path) -> None:
    import csv
    import json
    import subprocess
    import sys
    from pathlib import Path

    from safetrajai.paths import PROJECT_ROOT
    from safetrajai.reid.dataset_index import INDEX_FIELDS, index_path, write_index

    rows = crops("A", 6) + crops("B", 6) + crops("C", 6, camera="C000011", date="20250512")
    rows = [dict(r, sequence="s", frame=i, x1=0, y1=0, x2=40, y2=40, score="0.9") for i, r in enumerate(rows)]
    write_index(index_path(tmp_path / "crops", "s"), [{k: r[k] for k in INDEX_FIELDS} for r in rows])
    (tmp_path / "auto.csv").write_text(
        "class,identity,image,action,rule\ncar,A,,remove,static\n", encoding="utf-8")
    (tmp_path / "manual.csv").write_text(
        "class,identity,image,action\ncar,B,000001.jpg,remove\n", encoding="utf-8")

    def build(version: str) -> Path:
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts/reid_data/build_reid_split.py"), "--version", version,
               "--crop-root", str(tmp_path / "crops"), "--exclusions", str(tmp_path / "manual.csv"),
               "--auto-exclusions", str(tmp_path / "auto.csv"), "--exclude-cameras", "C000011",
               "--splits-root", str(tmp_path / "splits")]
        subprocess.run(cmd, check=True, capture_output=True)
        return tmp_path / "splits" / version

    first, second = build("t1"), build("t2")
    with (first / "manifest.csv").open(encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    assert {r["identity"] for r in manifest} == {"B"}
    assert "car/B/000001.jpg" not in {r["crop_path"] for r in manifest} and len(manifest) == 5
    assert (first / "manifest.csv").read_bytes() == (second / "manifest.csv").read_bytes()
    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["exclusions"] == {"auto": 1, "manual": 1}
    assert summary["excluded_cameras"] == ["C000011"]


def test_merged_fragments_count_together() -> None:
    rows = crops("A", 3) + [dict(r, time_s=f"{10 + float(r['time_s']):.3f}") for r in crops("B", 2)]
    manifest = build_manifest(rows, [], RULES, {"C000025"}, {("car", "B"): "A"})
    assert [r["identity"] for r in manifest] == ["A"] * 5
    assert [r["crop_path"] for r in manifest][-2:] == ["car/B/000000.jpg", "car/B/000001.jpg"]


def test_exclusions_apply_before_merges() -> None:
    exclusions = [{"class": "car", "identity": "B", "image": "", "action": "remove"}]
    manifest = build_manifest(crops("A", 4) + crops("B", 3), exclusions, RULES, {"C000025"}, {("car", "B"): "A"})
    assert {r["crop_path"] for r in manifest} == {f"car/A/{i:06d}.jpg" for i in range(4)}


def test_manifest_with_merges_is_reproducible() -> None:
    rows = crops("A", 20) + crops("B", 20) + crops("C", 3, cls="bus")
    merges = {("car", "B"): "A"}
    first = build_manifest(rows, [], RULES, set(), merges)
    assert first == build_manifest(list(reversed(rows)), [], RULES, set(), dict(merges))
    assert len(first) == 30 and {r["identity"] for r in first} == {"A"}


def test_build_reid_split_applies_merges_and_records_summary(tmp_path) -> None:
    import csv
    import json
    import subprocess
    import sys

    from safetrajai.paths import PROJECT_ROOT
    from safetrajai.reid.dataset_index import INDEX_FIELDS, index_path, write_index

    rows = crops("A", 3) + crops("B", 2) + crops("C", 5)
    rows = [dict(r, sequence="s", frame=i, x1=0, y1=0, x2=40, y2=40, score="0.9") for i, r in enumerate(rows)]
    write_index(index_path(tmp_path / "crops", "s"), [{k: r[k] for k in INDEX_FIELDS} for r in rows])
    (tmp_path / "merges.csv").write_text(
        "class,identity,merged_identity,sequence,score\ncar,B,A,s,0.950000\n", encoding="utf-8")
    base = [sys.executable, str(PROJECT_ROOT / "scripts/reid_data/build_reid_split.py"),
            "--crop-root", str(tmp_path / "crops"), "--exclusions", str(tmp_path / "none.csv"),
            "--auto-exclusions", str(tmp_path / "none_auto.csv"), "--splits-root", str(tmp_path / "splits")]
    subprocess.run([*base, "--version", "plain"], check=True, capture_output=True)
    subprocess.run([*base, "--version", "merged", "--merges", str(tmp_path / "merges.csv")],
                   check=True, capture_output=True)

    def identities(version: str) -> list[str]:
        with (tmp_path / "splits" / version / "manifest.csv").open(encoding="utf-8") as f:
            return sorted({r["identity"] for r in csv.DictReader(f)})

    assert identities("plain") == ["C"]
    assert identities("merged") == ["A", "C"]
    summary = json.loads((tmp_path / "splits" / "merged" / "summary.json").read_text(encoding="utf-8"))
    assert summary["merges"]["merged_identities"] == 1 and summary["merges"]["merged_into"] == 1
    plain = json.loads((tmp_path / "splits" / "plain" / "summary.json").read_text(encoding="utf-8"))
    assert plain["merges"]["merged_identities"] == 0
