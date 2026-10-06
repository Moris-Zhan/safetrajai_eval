"""Scenarios from specs/reid-dataset: occlusion fragment merging."""

import re

import numpy as np
import pytest

from safetrajai.reid.fragment_merge import (
    calibrate_threshold,
    collect_tracks,
    endpoint_crop_paths,
    find_candidates,
    load_merge_config,
    load_retention,
    merge_records,
    score_candidates,
    select_pairs,
    stratified_sample,
)

BUFFERS = {"car": 30, "small_truck": 30}  # retention in frames for both trackers
MARGIN_FRAMES = 10
THRESHOLD = 0.8


def track(identity: str, f0: int, n: int, x0: float, vx: float = 2.0, step: int = 5, fps: float = 10.0,
          cls: str = "car", seq: str = "s", camera: str = "C000012") -> list[dict]:
    """n crops every `step` frames from frame f0, moving vx px/frame."""
    rows = []
    for k in range(n):
        frame = f0 + k * step
        x = x0 + vx * (frame - f0)
        rows.append({"crop_path": f"{cls}/{identity}/{k:06d}.jpg", "identity": identity, "class": cls,
                     "camera": camera, "sequence": seq, "frame": str(frame), "time_s": f"{frame / fps:.3f}",
                     "x1": x - 20, "y1": 100, "x2": x + 20, "y2": 140})
    return rows


def candidates(rows: list[dict], buffers: dict = BUFFERS) -> list:
    return find_candidates(collect_tracks(rows, []), buffers, MARGIN_FRAMES, 2.0)


def features_for(rows: list[dict], vectors: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {r["crop_path"]: np.asarray(vectors[r["identity"]], dtype=float) for r in rows}


def run(rows: list[dict], vectors: dict[str, list[float]], exclusions: list[dict] = ()) -> list[dict]:
    tracks = collect_tracks(rows, list(exclusions))
    found = find_candidates(tracks, BUFFERS, MARGIN_FRAMES, 2.0)
    scored = score_candidates(found, tracks, features_for(rows, vectors), 3)
    return merge_records(select_pairs(scored, THRESHOLD), tracks)


def test_occluded_fragment_is_merged() -> None:
    # A: frames 80..100 ending at x=160; extrapolated 10 frames -> x=180; B starts at frame 110, x=220 (1 box away).
    rows = track("A", 80, 5, 120.0) + track("B", 110, 4, 220.0)
    records = run(rows, {"A": [1, 0], "B": [0.95, 0.05]})
    assert [(r["identity"], r["merged_identity"]) for r in records] == [("B", "A")]


@pytest.mark.parametrize("tracker", ["botsort", "bytetrack"])
@pytest.mark.parametrize("fps", [10, 14])
def test_candidate_window_is_retention_plus_margin_frames(tmp_path, tracker, fps) -> None:
    import yaml

    cfg = tmp_path / "track.yaml"
    cfg.write_text(yaml.safe_dump({"tracking_params": {"tracker": tracker},
                                   "class_tracking": {"class_tracking_params": {"car": {"track_buffer": 30}}}}))
    buffers = load_retention(cfg)
    assert buffers == {"car": 30} and type(buffers["car"]) is int
    a = track("A", 80, 5, 120.0, fps=fps)  # ends at frame 100, x=160
    inside = a + track("B", 140, 4, 240.0, fps=fps)
    outside = a + track("B", 141, 4, 242.0, fps=fps)
    assert [(c.identity_b, c.gap_frames) for c in candidates(inside, buffers)] == [("B", 40)]
    assert candidates(outside, buffers) == []


def test_candidate_window_does_not_require_inferable_fps() -> None:
    rows = track("A", 100, 1, 160.0) + track("B", 140, 1, 160.0)
    for row in rows:
        row["time_s"] = "0"
    assert [c.gap_frames for c in candidates(rows)] == [40]


def test_extrapolation_uses_per_frame_velocity() -> None:
    # A's last two crops are 10 frames and 20 px apart -> 2 px/frame; 30 frames later -> +60 px.
    a = track("A", 80, 3, 100.0, step=10)  # frames 80, 90, 100 ending at x=140
    [found] = candidates(a + track("B", 130, 3, 200.0))
    assert found.gap_frames == 30 and found.distance_ratio == pytest.approx(0.0)


def test_dissimilar_neighbour_is_not_merged() -> None:
    rows = track("A", 80, 5, 120.0) + track("B", 110, 4, 220.0)
    assert run(rows, {"A": [1, 0], "B": [0, 1]}) == []


def test_no_merge_across_class_or_sequence() -> None:
    rows = (track("A", 80, 5, 120.0) + track("B", 110, 4, 220.0, cls="small_truck")
            + track("C", 110, 4, 220.0, seq="other"))
    assert candidates(rows) == []
    assert run(rows, {"A": [1, 0], "B": [1, 0], "C": [1, 0]}) == []


def test_one_to_one_keeps_most_similar() -> None:
    rows = track("A", 80, 5, 120.0) + track("B", 110, 4, 210.0) + track("C", 115, 4, 200.0)
    records = run(rows, {"A": [1, 0], "B": [0.99, 0.01], "C": [0.9, 0.1]})
    assert [(r["identity"], r["merged_identity"]) for r in records] == [("B", "A")]


def test_chain_of_fragments_maps_to_earliest() -> None:
    # B: frames 110..125 ending at x=250; C starts at frame 140 near the extrapolation (x=280), too far from A's.
    rows = track("A", 80, 5, 120.0) + track("B", 110, 4, 220.0) + track("C", 140, 4, 330.0)
    records = run(rows, {"A": [1, 0], "B": [1, 0], "C": [1, 0]})
    assert [(r["identity"], r["merged_identity"]) for r in records] == [("B", "A"), ("C", "A")]


def test_candidate_time_and_position_limits() -> None:
    late = track("A", 80, 5, 120.0) + track("B", 145, 4, 250.0)  # gap 45 frames > 30 + 10
    far = track("A", 80, 5, 120.0) + track("B", 110, 4, 320.0)  # 140 px > 2 x 40 px
    for rows in (late, far):
        assert candidates(rows) == []


def test_excluded_identity_does_not_participate() -> None:
    rows = track("A", 80, 5, 120.0) + track("B", 110, 4, 220.0)
    exclusions = [{"class": "car", "identity": "B", "image": "", "action": "remove"}]
    assert run(rows, {"A": [1, 0], "B": [1, 0]}, exclusions) == []


def test_endpoint_paths_are_only_fragment_ends() -> None:
    rows = track("A", 80, 5, 120.0) + track("B", 110, 4, 220.0)
    tracks = collect_tracks(rows, [])
    paths = endpoint_crop_paths(find_candidates(tracks, BUFFERS, MARGIN_FRAMES, 2.0), tracks, 3)
    assert paths == ["car/A/000002.jpg", "car/A/000003.jpg", "car/A/000004.jpg",
                     "car/B/000000.jpg", "car/B/000001.jpg", "car/B/000002.jpg"]


def test_merge_records_are_order_independent() -> None:
    rows = track("A", 80, 5, 120.0) + track("B", 110, 4, 220.0) + track("C", 140, 4, 330.0)
    vectors = {"A": [1, 0], "B": [1, 0], "C": [1, 0]}
    assert run(rows, vectors) == run(list(reversed(rows)), vectors)


def test_load_retention_reads_integer_buffers() -> None:
    from safetrajai.paths import CONFIG_DIR

    buffers = load_retention(CONFIG_DIR / "reid_crops.yaml")
    assert buffers["car"] == 30 and {"person", "scooter", "truck"} <= set(buffers)


def test_merge_config_uses_margin_frames_and_rejects_seconds(tmp_path) -> None:
    from safetrajai.paths import CONFIG_DIR

    text = (CONFIG_DIR / "fragment_merge.yaml").read_text(encoding="utf-8")
    assert load_merge_config(CONFIG_DIR / "fragment_merge.yaml")["margin_frames"] == 10
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(text + "time_margin_s: 1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="time_margin_s.*margin_frames"):
        load_merge_config(legacy)


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
    """tracks_spec: list of (identity, f0, n, x0, seq, gray)."""
    import cv2

    from safetrajai.reid.dataset_index import INDEX_FIELDS, index_path, write_index

    by_seq = {}
    for identity, f0, n, x0, seq, gray in tracks_spec:
        for r in track(identity, f0, n, x0, seq=seq):
            (root / r["crop_path"]).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / r["crop_path"]), np.full((40, 40, 3), gray, np.uint8))
            by_seq.setdefault(seq, []).append({**r, "date": "20250512", "score": "0.9"})
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
    _write_fragment_crops(crops, [("A", 80, 6, 120.0, "s", 200), ("B", 110, 5, 220.0, "s", 200),
                                  ("X", 300, 6, 500.0, "s", 50), ("C", 110, 5, 220.0, "t", 200)])
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
