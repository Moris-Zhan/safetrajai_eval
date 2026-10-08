"""Scenarios from specs/reid-model: distractor-aware PK sampling only picks co-occurring, nearby identities."""

from collections import Counter

import pytest

from safetrajai.reid.training.sampler import (
    CropPoint, DistractorPKBatchSampler, PKBatchSampler, distractor_radius, find_distractors,
)


def track(sequence: str, cls: str, frames: range, x: float, y: float = 100.0) -> tuple[str, str, list[CropPoint]]:
    return sequence, cls, [CropPoint(frame, x, y) for frame in frames]


def test_radius_scales_with_camera_width() -> None:
    assert distractor_radius(80.0, 1280) == 320.0
    assert distractor_radius(80.0, 640) == 160.0
    assert distractor_radius(200.0, 704) == pytest.approx(440.0)


def test_cooccurring_nearby_identities_are_distractors() -> None:
    tracks = {0: track("s", "car", range(100, 131, 5), 100.0), 1: track("s", "car", range(100, 131, 5), 300.0)}
    result = find_distractors(tracks, {0: 320.0, 1: 320.0}, cooccur_min_frames=10, pair_max_frame_gap=5)
    assert result == {0: frozenset({1}), 1: frozenset({0})}


def test_fragment_is_not_a_distractor() -> None:
    tracks = {0: track("s", "car", range(100, 131, 5), 100.0), 1: track("s", "car", range(140, 171, 5), 100.0)}
    result = find_distractors(tracks, {0: 320.0, 1: 320.0}, cooccur_min_frames=10, pair_max_frame_gap=5)
    assert result == {0: frozenset(), 1: frozenset()}


def test_cooccurring_but_far_identity_is_not_a_distractor() -> None:
    tracks = {0: track("s", "car", range(100, 131, 5), 100.0), 1: track("s", "car", range(100, 131, 5), 421.0)}
    result = find_distractors(tracks, {0: 320.0, 1: 320.0}, cooccur_min_frames=10, pair_max_frame_gap=5)
    assert result[0] == frozenset()


def test_short_overlap_class_and_sequence_must_match() -> None:
    tracks = {
        0: track("s", "car", range(100, 131, 5), 100.0),
        1: track("s", "car", range(125, 160, 5), 120.0),
        2: track("s", "scooter", range(100, 131, 5), 110.0),
        3: track("t", "car", range(100, 131, 5), 110.0),
    }
    result = find_distractors(tracks, {i: 320.0 for i in tracks}, cooccur_min_frames=10, pair_max_frame_gap=5)
    assert all(not v for v in result.values())


def clustered_labels(clusters: int, size: int, crops: int) -> tuple[list[int], dict[int, frozenset[int]]]:
    labels = [label for label in range(clusters * size) for _ in range(crops)]
    distractors = {}
    for label in range(clusters * size):
        start = label // size * size
        distractors[label] = frozenset(range(start, start + size)) - {label}
    return labels, distractors


def test_distractor_sampler_batches_are_p_by_k_with_unique_identities() -> None:
    labels, distractors = clustered_labels(clusters=4, size=20, crops=3)
    sampler = DistractorPKBatchSampler(labels, p=8, k=4, seed=0, distractors=distractors, fraction=0.5)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 10
    seen = []
    for batch in batches:
        assert len(batch) == 32
        counts = Counter(labels[i] for i in batch)
        assert len(counts) == 8 and set(counts.values()) == {4}
        seen.extend(counts)
    assert len(seen) == len(set(seen))


def test_distractor_sampler_fills_half_with_seed_distractors() -> None:
    labels, distractors = clustered_labels(clusters=4, size=20, crops=3)
    sampler = DistractorPKBatchSampler(labels, p=8, k=4, seed=0, distractors=distractors, fraction=0.5)
    first = next(iter(sampler))
    seed_label = labels[first[0]]
    batch_labels = {labels[i] for i in first}
    assert len(batch_labels & (distractors[seed_label] | {seed_label})) >= 4
    list(sampler)
    assert sampler.epoch_stats["distractor_identities"] >= 3 * len(sampler) // 2
    assert 0 < sampler.epoch_stats["distractor_share"] <= 0.5


def test_distractor_sampler_never_falls_back_to_non_distractors() -> None:
    labels, distractors = clustered_labels(clusters=1, size=3, crops=2)
    labels += [label for label in range(3, 20) for _ in range(2)]
    distractors.update({label: frozenset() for label in range(3, 20)})
    sampler = DistractorPKBatchSampler(labels, p=10, k=2, seed=1, distractors=distractors, fraction=0.5)
    for batch in sampler:
        assert len({labels[i] for i in batch}) == 10
    assert sampler.epoch_stats["distractor_identities"] <= 2


def test_fraction_zero_matches_pk_sampler() -> None:
    labels, distractors = clustered_labels(clusters=3, size=7, crops=3)
    plain = PKBatchSampler(labels, p=4, k=4, seed=3)
    distractor = DistractorPKBatchSampler(labels, p=4, k=4, seed=3, distractors=distractors, fraction=0.0)
    for epoch in (0, 2):
        plain.set_epoch(epoch)
        distractor.set_epoch(epoch)
        assert list(plain) == list(distractor)


def test_crop_points_come_from_index_metadata(tmp_path) -> None:
    from safetrajai.reid.dataset_index import index_path, write_index
    from safetrajai.reid.training.sampler import camera_widths, load_crop_points

    seq = "C000054 Road_20250623080000"
    crop = f"car/{seq}_track3/000010.jpg"
    write_index(index_path(tmp_path, seq), [{
        "crop_path": crop, "identity": f"{seq}_track3", "class": "car", "camera": "C000054", "sequence": seq,
        "date": "20250623", "frame": 10, "time_s": 1.0, "x1": 10, "y1": 20, "x2": 30, "y2": 60, "score": 0.9,
    }])
    points = load_crop_points([{"crop_path": crop}], tmp_path)
    assert points == {crop: (seq, CropPoint(10, 20.0, 40.0))}
    with pytest.raises(ValueError, match="找不到"):
        load_crop_points([{"crop_path": crop}, {"crop_path": f"car/{seq}_track4/000011.jpg"}], tmp_path)

    (tmp_path / "roi").mkdir()
    (tmp_path / "roi" / "C000054.yaml").write_text("camera: C000054\nimage_size: [1280, 720]\npolygon: []\n")
    assert camera_widths({"C000054"}, tmp_path / "roi") == {"C000054": 1280}
