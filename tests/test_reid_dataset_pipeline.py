"""Unit tests for the rebuild-reid-dataset pipeline changes (frame rate, MOT output)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from safetrajai.pipeline.tracking_pipeline import resolve_frame_rate


def write_video(path: Path, fps: float, frames: int = 5, size: tuple[int, int] = (64, 48)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    return path


def test_frame_rate_auto_reads_video_fps(tmp_path: Path) -> None:
    video = write_video(tmp_path / "C000054 test_20250623065608.mp4", fps=10)
    assert resolve_frame_rate("auto", str(video)) == 10


def test_frame_rate_number_is_used_as_is(tmp_path: Path) -> None:
    video = write_video(tmp_path / "C000054 test_20250623065608.mp4", fps=10)
    assert resolve_frame_rate(30, str(video)) == 30


def test_frame_rate_auto_without_readable_video_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_frame_rate("auto", str(tmp_path / "missing.mp4"))


def test_save_mot_format_handles_non_c000025_sequences(tmp_path: Path) -> None:
    from safetrajai.evaluation.mot_format import save_mot_format

    seq = "C000054 民族一路 天祥二路_20250623065608"
    seq_dir = tmp_path / seq
    labels = seq_dir / "labels" / seq
    images = seq_dir / "images" / seq
    labels.mkdir(parents=True)
    images.mkdir(parents=True)
    (images / f"{seq}_f000001.jpg").touch()
    (labels / f"{seq}_f000001.txt").write_text("1 0.5 0.5 0.1 0.1 7 0.9000\n", encoding="utf-8")

    save_mot_format(images, labels, seq_dir / f"{seq}.txt", img_width=1280, img_height=720)

    car_file = seq_dir / "trackers" / "mot_challenge" / "CCTV-val" / "YOLO11x__car" / "data" / f"{seq}.txt"
    assert car_file.read_text(encoding="utf-8").startswith("1,7,")


@pytest.mark.parametrize(
    ("seq_name", "date"),
    [
        ("C000054 民族一路 天祥二路_20250629080441", "20250629"),
        ("C000025 中華三路 民生二路(東北角)_20251209145630_15s_1", "20251209"),
    ],
)
def test_date_from_seq_name(seq_name: str, date: str) -> None:
    from safetrajai.reid.dataset_index import date_from_seq_name

    assert date_from_seq_name(seq_name) == date


def test_date_from_seq_name_without_timestamp_fails() -> None:
    from safetrajai.reid.dataset_index import date_from_seq_name

    with pytest.raises(ValueError):
        date_from_seq_name("C000054 民族一路 天祥二路")


def test_index_round_trip(tmp_path: Path) -> None:
    from safetrajai.reid.dataset_index import INDEX_FIELDS, index_path, read_all_indexes, write_index

    row = dict(zip(INDEX_FIELDS, ["car/s_track7/000001.jpg", "s_track7", "car", "C000054", "s", "20250623",
                                  "1", "0.100", "10", "20", "50", "60", "0.9000"]))
    write_index(index_path(tmp_path, "s"), [row])
    assert read_all_indexes(tmp_path) == [row]


SEQ_C000054 = "C000054 民族一路 天祥二路_20250629080441"
FULL_ROI = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32)


def run_collector(tmp_path: Path, frames: list[tuple[float, list]]):
    from safetrajai.pipeline.reid_crops import ReidCropCollector

    collector = ReidCropCollector(tmp_path, FULL_ROI, SEQ_C000054)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    for frame_id, (time_s, tracks) in enumerate(frames, start=1):
        collector.collect(np.array(tracks, dtype=float), image, frame_id, time_s, 200, 200)
    return collector, collector.finalize()


def test_collector_samples_every_half_second_at_10fps(tmp_path: Path) -> None:
    from safetrajai.reid.dataset_index import index_path, read_index

    frames = [(i / 10, [[10, 10, 60, 80, 7, 0.9, 1]]) for i in range(1, 31)]
    _, stats = run_collector(tmp_path, frames)

    rows = read_index(index_path(tmp_path, SEQ_C000054))
    times = [float(r["time_s"]) for r in rows]
    assert stats["crops"] in (6, 7)
    assert all(b - a >= 0.5 - 1e-9 for a, b in zip(times, times[1:]))


def test_collector_skips_small_boxes_without_consuming_interval(tmp_path: Path) -> None:
    from safetrajai.reid.dataset_index import index_path, read_index

    frames = [(0.1, [[10, 10, 30, 60, 7, 0.9, 1]]), (0.2, [[10, 10, 60, 80, 7, 0.9, 1]])]
    _, stats = run_collector(tmp_path, frames)

    rows = read_index(index_path(tmp_path, SEQ_C000054))
    assert [r["frame"] for r in rows] == ["2"]
    assert stats["too_small"] == {"car": 1}


def test_collector_skips_detections_outside_roi(tmp_path: Path) -> None:
    from safetrajai.pipeline.reid_crops import ReidCropCollector

    roi = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.float32)
    collector = ReidCropCollector(tmp_path, roi, SEQ_C000054)
    collector.collect(np.array([[100, 100, 160, 180, 7, 0.9, 1]], dtype=float), np.zeros((200, 200, 3), np.uint8), 1, 0.1, 200, 200)
    assert collector.finalize()["crops"] == 0


def test_collector_index_matches_files_and_carries_camera_date(tmp_path: Path) -> None:
    from safetrajai.reid.dataset_index import index_path, read_index

    frames = [(0.1, [[10, 10, 60, 80, 7, 0.9, 1], [70, 10, 120, 80, 3, 0.8, 0]]), (0.7, [[10, 10, 60, 80, 7, 0.9, 1]])]
    run_collector(tmp_path, frames)

    rows = read_index(index_path(tmp_path, SEQ_C000054))
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.jpg"))
    assert sorted(r["crop_path"] for r in rows) == files
    assert {(r["camera"], r["date"]) for r in rows} == {("C000054", "20250629")}
    assert {r["identity"] for r in rows} == {f"{SEQ_C000054}_track7", f"{SEQ_C000054}_track3"}


def test_collector_refuses_existing_index(tmp_path: Path) -> None:
    from safetrajai.pipeline.reid_crops import ReidCropCollector

    run_collector(tmp_path, [(0.1, [[10, 10, 60, 80, 7, 0.9, 1]])])
    with pytest.raises(FileExistsError):
        ReidCropCollector(tmp_path, FULL_ROI, SEQ_C000054)


def test_frame_time_falls_back_to_frame_index() -> None:
    from safetrajai.pipeline.reid_crops import frame_time_s

    assert frame_time_s(1234.0, 12, 10.0, None) == pytest.approx(1.234)
    assert frame_time_s(0.0, 12, 10.0, None) == pytest.approx(1.2)
    assert frame_time_s(1000.0, 12, 10.0, 1.5) == pytest.approx(1.2)


REGISTRY = {
    "eval_set": {"role": "eval", "cameras": ["C000025"]},
    "source_set": {"role": "source", "cameras": ["C000054", "C000043"]},
}


def test_crop_sources_reject_eval_camera(tmp_path: Path) -> None:
    from safetrajai.pipeline.tracking_pipeline import check_crop_sources

    (tmp_path / "C000025.yaml").touch()
    with pytest.raises(ValueError, match="C000025"):
        check_crop_sources(["x/C000025 中華三路 民生二路(東北角)_20251209145630_1.mp4"], tmp_path, REGISTRY)


def test_crop_sources_reject_missing_roi(tmp_path: Path) -> None:
    from safetrajai.pipeline.tracking_pipeline import check_crop_sources

    (tmp_path / "C000054.yaml").touch()
    videos = [f"x/{SEQ_C000054}.mp4", "x/C000043 民族一路 九如一路_20250514071352.mp4"]
    with pytest.raises(ValueError, match="C000043"):
        check_crop_sources(videos, tmp_path, REGISTRY)


def test_run_stage1_rejects_before_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import types
    from safetrajai.pipeline import tracking_pipeline

    video_dir = tmp_path / "raw" / "C000025"
    video_dir.mkdir(parents=True)
    write_video(video_dir / "C000025 中華三路 民生二路(東北角)_20251209145630_1.mp4", fps=10)
    crop_dir = tmp_path / "crops"
    cfg = types.SimpleNamespace(
        OUTPUT_ROOT=tmp_path / "out", run_det_eval_dir="a", run_track_eval_dir="b",
        SOURCE_VIDEO_DIR=tmp_path / "raw", dataset_layout="raw", dataset_cameras=["C000025"],
        save_reid_crops=True, reid_crop_dir=str(crop_dir),
    )
    monkeypatch.setattr(tracking_pipeline.dataset_layout, "load_dataset_registry", lambda: REGISTRY)
    monkeypatch.setattr(tracking_pipeline.TrackingEvalPipeline, "_process_single_video",
                        lambda *a, **k: pytest.fail("tracking must not start"))

    with pytest.raises(ValueError, match="評估鏡頭"):
        tracking_pipeline.TrackingEvalPipeline(cfg).run_stage1()
    assert not crop_dir.exists()


def _write_config(tmp_path: Path, cameras: list[str]) -> Path:
    import yaml
    from safetrajai.paths import CONFIG_DIR

    raw = yaml.safe_load((CONFIG_DIR / "reid" / "botsort_save_reid_crops_video.yaml").read_text(encoding="utf-8"))
    raw["paths"]["cameras"] = cameras
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def test_paths_cameras_narrows_registered_cameras(tmp_path: Path) -> None:
    from safetrajai.evaluation.config_loader import load_config

    assert load_config(str(_write_config(tmp_path, ["C000043"]))).dataset_cameras == ["C000043"]


def test_paths_cameras_rejects_unregistered_camera(tmp_path: Path) -> None:
    from safetrajai.evaluation.config_loader import load_config

    with pytest.raises(ValueError, match="C000025"):
        load_config(str(_write_config(tmp_path, ["C000043", "C000025"])))
