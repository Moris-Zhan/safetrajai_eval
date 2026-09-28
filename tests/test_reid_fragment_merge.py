"""Scenarios from specs/reid-dataset: occlusion fragment merging."""

import re

import numpy as np

from safetrajai.reid.fragment_merge import (
    calibrate_threshold,
    collect_tracks,
    endpoint_crop_paths,
    find_candidates,
    merge_records,
    score_candidates,
    select_pairs,
    stratified_sample,
)

BUFFERS = {"car": 3.0, "small_truck": 3.0}
THRESHOLD = 0.8


def track(identity: str, t0: float, n: int, x0: float, vx: float = 20.0,
          cls: str = "car", seq: str = "s", camera: str = "C000012") -> list[dict]:
    rows = []
    for k in range(n):
        t = t0 + k * 0.5
        x = x0 + vx * (t - t0)
        rows.append({"crop_path": f"{cls}/{identity}/{k:06d}.jpg", "identity": identity, "class": cls,
                     "camera": camera, "sequence": seq, "time_s": f"{t:.3f}",
                     "x1": x - 20, "y1": 100, "x2": x + 20, "y2": 140})
    return rows


def features_for(rows: list[dict], vectors: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {r["crop_path"]: np.asarray(vectors[r["identity"]], dtype=float) for r in rows}


def run(rows: list[dict], vectors: dict[str, list[float]], exclusions: list[dict] = ()) -> list[dict]:
    tracks = collect_tracks(rows, list(exclusions))
    candidates = find_candidates(tracks, BUFFERS, 1.0, 2.0)
    scored = score_candidates(candidates, tracks, features_for(rows, vectors), 3)
    return merge_records(select_pairs(scored, THRESHOLD), tracks)


def test_occluded_fragment_is_merged() -> None:
    # A ends at t=10.0, x=200; extrapolated to t=11.0 -> x=220; B starts at x=260 (1 box away).
    rows = track("A", 8.0, 5, 120.0) + track("B", 11.0, 4, 260.0)
    records = run(rows, {"A": [1, 0], "B": [0.95, 0.05]})
    assert [(r["identity"], r["merged_identity"]) for r in records] == [("B", "A")]


def test_dissimilar_neighbour_is_not_merged() -> None:
    rows = track("A", 8.0, 5, 120.0) + track("B", 11.0, 4, 260.0)
    assert run(rows, {"A": [1, 0], "B": [0, 1]}) == []


def test_no_merge_across_class_or_sequence() -> None:
    rows = (track("A", 8.0, 5, 120.0) + track("B", 11.0, 4, 260.0, cls="small_truck")
            + track("C", 11.0, 4, 260.0, seq="other"))
    tracks = collect_tracks(rows, [])
    assert find_candidates(tracks, BUFFERS, 1.0, 2.0) == []
    assert run(rows, {"A": [1, 0], "B": [1, 0], "C": [1, 0]}) == []


def test_one_to_one_keeps_most_similar() -> None:
    rows = track("A", 8.0, 5, 120.0) + track("B", 11.0, 4, 250.0) + track("C", 11.5, 4, 240.0)
    records = run(rows, {"A": [1, 0], "B": [0.99, 0.01], "C": [0.9, 0.1]})
    assert [(r["identity"], r["merged_identity"]) for r in records] == [("B", "A")]


def test_chain_of_fragments_maps_to_earliest() -> None:
    # B: t=11.0..12.5 ending at x=290; C starts at t=14.0 near the extrapolation (x=320).
    rows = track("A", 8.0, 5, 120.0) + track("B", 11.0, 4, 260.0) + track("C", 14.0, 4, 330.0)
    records = run(rows, {"A": [1, 0], "B": [1, 0], "C": [1, 0]})
    assert [(r["identity"], r["merged_identity"]) for r in records] == [("B", "A"), ("C", "A")]


def test_candidate_time_and_position_limits() -> None:
    late = track("A", 8.0, 5, 120.0) + track("B", 14.5, 4, 330.0)  # gap 4.5 s > 3 + 1
    far = track("A", 8.0, 5, 120.0) + track("B", 11.0, 4, 360.0)  # 140 px > 2 x 40 px
    for rows in (late, far):
        assert find_candidates(collect_tracks(rows, []), BUFFERS, 1.0, 2.0) == []


def test_excluded_identity_does_not_participate() -> None:
    rows = track("A", 8.0, 5, 120.0) + track("B", 11.0, 4, 260.0)
    exclusions = [{"class": "car", "identity": "B", "image": "", "action": "remove"}]
    assert run(rows, {"A": [1, 0], "B": [1, 0]}, exclusions) == []


def test_endpoint_paths_are_only_fragment_ends() -> None:
    rows = track("A", 8.0, 5, 120.0) + track("B", 11.0, 4, 260.0)
    tracks = collect_tracks(rows, [])
    paths = endpoint_crop_paths(find_candidates(tracks, BUFFERS, 1.0, 2.0), tracks, 3)
    assert paths == ["car/A/000002.jpg", "car/A/000003.jpg", "car/A/000004.jpg",
                     "car/B/000000.jpg", "car/B/000001.jpg", "car/B/000002.jpg"]


def test_merge_records_are_order_independent() -> None:
    rows = track("A", 8.0, 5, 120.0) + track("B", 11.0, 4, 260.0) + track("C", 14.0, 4, 330.0)
    vectors = {"A": [1, 0], "B": [1, 0], "C": [1, 0]}
    assert run(rows, vectors) == run(list(reversed(rows)), vectors)


def test_calibrate_threshold_picks_lowest_accurate_value() -> None:
    labels = [(0.9, True)] * 19 + [(0.85, False)] + [(0.7, True), (0.6, False), (0.5, False)]
    result = calibrate_threshold(labels, 0.95)
    # >= 0.7: 20/21 correct (95.2%); >= 0.6: 20/22 (90.9%).
    assert result["threshold"] == 0.7 and result["n_above"] == 21 and result["n_wrong_above"] == 1
    assert calibrate_threshold([(0.9, False)], 0.95) is None


def test_stratified_sample_is_reproducible_and_keeps_other_cameras() -> None:
    from safetrajai.reid.fragment_merge import Candidate, ScoredCandidate

    rng = np.random.default_rng(1)
    scored = [ScoredCandidate(Candidate("car", "s", "C000054" if i % 20 else "C000012", f"a{i}", f"b{i}", 1.0, 1.0),
                              float(rng.random())) for i in range(1000)]
    cfg = {"strata_quantiles": [0.5, 0.9], "counts": {"low": 20, "mid": 30, "high": 30},
           "major_cameras": ["C000054"], "min_other": 10}
    first, second = stratified_sample(scored, cfg, 0), stratified_sample(list(reversed(scored)), cfg, 0)
    assert first == second and len(first) == 80
    assert [sum(r["stratum"] == s for r in first) for s in ("low", "mid", "high")] == [20, 30, 30]
    assert sum(r["camera"] != "C000054" for r in first) >= 10
    assert min(float(r["score"]) for r in first if r["stratum"] == "high") >= np.quantile([s.score for s in scored], 0.9)


def _write_fragment_crops(root, tracks_spec):
    """tracks_spec: list of (identity, t0, n, x0, seq, gray)."""
    import cv2

    from safetrajai.reid.dataset_index import INDEX_FIELDS, index_path, write_index

    by_seq = {}
    for identity, t0, n, x0, seq, gray in tracks_spec:
        for r in track(identity, t0, n, x0, seq=seq):
            (root / r["crop_path"]).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / r["crop_path"]), np.full((40, 40, 3), gray, np.uint8))
            by_seq.setdefault(seq, []).append(
                {**r, "date": "20250512", "frame": int(float(r["time_s"]) * 10), "score": "0.9"})
    for seq, rows in by_seq.items():
        write_index(index_path(root, seq), [{k: r[k] for k in INDEX_FIELDS} for r in rows])


def _fragment_config(tmp_path, threshold):
    from safetrajai.paths import CONFIG_DIR

    text = (CONFIG_DIR / "fragment_merge.yaml").read_text(encoding="utf-8")
    text = re.sub(r"(?m)^threshold:.*$", f"threshold: {threshold}", text)
    path = tmp_path / "fragment_merge.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _file_listing(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def _load_script(name):
    import importlib.util

    from safetrajai.paths import PROJECT_ROOT

    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / "reid_data" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path, monkeypatch):
    import timm
    import torch

    crops = tmp_path / "crops"
    # A -> B is an occlusion fragment (same appearance); X is a lone track; C is in another sequence.
    _write_fragment_crops(crops, [("A", 8.0, 6, 120.0, "s", 200), ("B", 11.0, 5, 260.0, "s", 200),
                                  ("X", 30.0, 6, 500.0, "s", 50), ("C", 11.0, 5, 260.0, "t", 200)])
    (tmp_path / "auto.csv").write_text("class,identity,image,action,rule\n", encoding="utf-8")
    (tmp_path / "manual.csv").write_text("class,identity,image,action\n", encoding="utf-8")

    class DummyModel(torch.nn.Module):
        pretrained_cfg = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}

        def forward(self, x):
            return torch.cat([x.mean(dim=(2, 3)), torch.ones(len(x), 1, device=x.device)], dim=1)

    monkeypatch.setattr(timm, "create_model", lambda *a, **k: DummyModel())
    return crops


def _common_args(tmp_path, crops, config):
    return ["--crop-root", str(crops), "--auto-exclusions", str(tmp_path / "auto.csv"),
            "--exclusions", str(tmp_path / "manual.csv"), "--config", str(config)]


def test_extract_script_reads_only_endpoint_crops(tmp_path, monkeypatch) -> None:
    import cv2

    crops = _fixture(tmp_path, monkeypatch)
    before = _file_listing(crops)
    reads = []
    real_imread = cv2.imread
    monkeypatch.setattr(cv2, "imread", lambda p, *a: reads.append(p) or real_imread(p, *a))
    out = tmp_path / "reports" / "fragment_features.npz"
    monkeypatch.setattr("sys.argv", ["extract", *_common_args(tmp_path, crops, _fragment_config(tmp_path, "null")),
                                     "--out", str(out)])
    _load_script("extract_fragment_features").main()

    expected = ["car/A/000003.jpg", "car/A/000004.jpg", "car/A/000005.jpg",
                "car/B/000000.jpg", "car/B/000001.jpg", "car/B/000002.jpg"]
    assert sorted(reads) == [str(crops / p) for p in expected]
    assert _file_listing(crops) == before
    data = np.load(out)
    assert data["crop_path"].tolist() == expected and data["features"].dtype == np.float32


def test_build_merges_script_is_reproducible_and_changes_no_files(tmp_path, monkeypatch) -> None:
    crops = _fixture(tmp_path, monkeypatch)
    config = _fragment_config(tmp_path, 0.9)
    features = tmp_path / "features.npz"
    monkeypatch.setattr("sys.argv", ["extract", *_common_args(tmp_path, crops, config), "--out", str(features)])
    _load_script("extract_fragment_features").main()
    before = _file_listing(crops)
    lists_before = [(tmp_path / n).read_bytes() for n in ("auto.csv", "manual.csv")]

    build = _load_script("build_fragment_merges")
    args = [*_common_args(tmp_path, crops, config), "--features", str(features)]
    out = tmp_path / "merges.csv"
    monkeypatch.setattr("sys.argv", ["build", *args, "--out", str(out)])
    build.main()
    assert not out.exists()

    outputs = []
    for name in ("m1.csv", "m2.csv"):
        monkeypatch.setattr("sys.argv", ["build", *args, "--out", str(tmp_path / name), "--apply"])
        build.main()
        outputs.append((tmp_path / name).read_bytes())
    assert outputs[0] == outputs[1]
    assert outputs[0].decode("utf-8").splitlines()[1:] == ["car,B,A,s,1.000000"]
    assert _file_listing(crops) == before
    assert [(tmp_path / n).read_bytes() for n in ("auto.csv", "manual.csv")] == lists_before

    samples = []
    for name in ("r1", "r2"):
        monkeypatch.setattr("sys.argv", ["build", *args, "--review-dir", str(tmp_path / name),
                                         "--sample", "80", "--seed", "0"])
        build.main()
        samples.append((tmp_path / name / "merge_sample.csv").read_bytes())
        assert (tmp_path / name / "fragment_review.html").exists()
    assert samples[0] == samples[1]
    assert _file_listing(crops) == before


def test_build_merges_refuses_apply_without_threshold(tmp_path, monkeypatch) -> None:
    import pytest

    crops = _fixture(tmp_path, monkeypatch)
    config = _fragment_config(tmp_path, "null")
    features = tmp_path / "features.npz"
    monkeypatch.setattr("sys.argv", ["extract", *_common_args(tmp_path, crops, config), "--out", str(features)])
    _load_script("extract_fragment_features").main()
    monkeypatch.setattr("sys.argv", ["build", *_common_args(tmp_path, crops, config), "--features", str(features),
                                     "--out", str(tmp_path / "m.csv"), "--apply"])
    with pytest.raises(SystemExit):
        _load_script("build_fragment_merges").main()
    assert not (tmp_path / "m.csv").exists()


def test_supplementary_sample_only_draws_unseen_high_scores() -> None:
    from safetrajai.reid.fragment_merge import Candidate, ScoredCandidate, supplementary_sample

    scored = [ScoredCandidate(Candidate("car", "s", "C000054", f"a{i}", f"b{i}", 1.0, 1.0), i / 100) for i in range(100)]
    seen = {("car", "a95", "b95")}
    first = supplementary_sample(scored, 0.9, seen, 5, 1, 81)
    assert first == supplementary_sample(list(reversed(scored)), 0.9, seen, 5, 1, 81)
    assert [r["sample_id"] for r in first] == [81, 82, 83, 84, 85] and {r["stratum"] for r in first} == {"extra"}
    assert all(float(r["score"]) >= 0.9 and r["identity_a"] != "a95" for r in first)
