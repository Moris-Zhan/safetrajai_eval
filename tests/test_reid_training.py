"""Scenarios from specs/reid-model: training reads only the dataset version through the local pack."""

import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from safetrajai.reid.training.data import (
    CropPack, ManifestDataset, build_transform, identity_labels, pack_manifest, read_manifest, verify_pack,
)
from safetrajai.reid.training.evaluate import retrieval_metrics
from safetrajai.reid.training.losses import batch_hard_triplet_loss
from safetrajai.reid.training.sampler import PKBatchSampler

FIELDS = ("crop_path", "identity", "class", "camera", "date", "split")


def write_synthetic_version(root: Path, identities: dict[str, int], version: str = "vtest") -> tuple[Path, Path]:
    """Writes crops for identities ("class/name" -> n crops) plus one crop that is on disk but not in the manifest."""
    crop_root = root / "crops"
    rows = []
    rng = np.random.default_rng(0)
    for n_id, (key, count) in enumerate(identities.items()):
        cls, name = key.split("/")
        color = rng.integers(0, 255, 3)
        for i in range(count):
            crop_path = f"{cls}/{name}/{i:06d}.jpg"
            image = np.full((40 + i, 24, 3), color, dtype=np.uint8)
            (crop_root / crop_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_root / crop_path), image)
            split = "val" if n_id % 4 == 3 else "train"
            rows.append({"crop_path": crop_path, "identity": name, "class": cls, "camera": "C000054",
                         "date": "20250623", "split": split})
    excluded = crop_root / "car" / "excluded_track9" / "000000.jpg"
    excluded.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(excluded), np.zeros((30, 30, 3), dtype=np.uint8))

    manifest = root / "splits" / version / "manifest.csv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest, crop_root


IDENTITIES = {f"{cls}/{cls}_track{i}": 3 + i for cls in ("car", "scooter") for i in range(4)}


def test_excluded_crop_is_never_packed_or_read(tmp_path, monkeypatch) -> None:
    manifest, crop_root = write_synthetic_version(tmp_path, IDENTITIES)
    rows = read_manifest(manifest)
    info = pack_manifest(rows, crop_root, tmp_path / "pack", "vtest")
    pack = CropPack(tmp_path / "pack")
    assert info["packed"] == info["manifest_rows"] == len(rows) == len(pack)
    assert "car/excluded_track9/000000.jpg" not in pack

    read = []
    real = CropPack.read_bgr
    monkeypatch.setattr(CropPack, "read_bgr", lambda self, p: read.append(p) or real(self, p))
    dataset = ManifestDataset(rows, pack, identity_labels(rows), 32, build_transform([0.5] * 3, [0.5] * 3, train=True))
    for i in range(len(dataset)):
        dataset[i]
    assert sorted(read) == sorted(r["crop_path"] for r in rows)


def test_pack_roundtrip_and_refuses_overwrite(tmp_path) -> None:
    manifest, crop_root = write_synthetic_version(tmp_path, IDENTITIES)
    rows = read_manifest(manifest)
    pack_manifest(rows, crop_root, tmp_path / "pack", "vtest")
    pack = CropPack(tmp_path / "pack")
    verify_pack(pack, rows)
    for row in rows:
        assert np.array_equal(pack.read_bgr(row["crop_path"]), cv2.imread(str(crop_root / row["crop_path"])))
    with pytest.raises(FileExistsError):
        pack_manifest(rows, crop_root, tmp_path / "pack", "vtest")


def test_verify_pack_detects_manifest_mismatch(tmp_path) -> None:
    manifest, crop_root = write_synthetic_version(tmp_path, IDENTITIES)
    rows = read_manifest(manifest)
    pack_manifest(rows[:-1], crop_root, tmp_path / "pack", "vtest")
    with pytest.raises(ValueError, match="不一致"):
        verify_pack(CropPack(tmp_path / "pack"), rows)


def test_pk_sampler_yields_p_identities_times_k() -> None:
    labels = [0] * 2 + [1] * 5 + [2] * 4 + [3] * 1 + [4] * 6 + [5] * 3 + [6] * 7
    sampler = PKBatchSampler(labels, p=3, k=4, seed=0)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 2
    for batch in batches:
        assert len(batch) == 12
        counts = Counter(labels[i] for i in batch)
        assert len(counts) == 3 and set(counts.values()) == {4}
    sampler.set_epoch(1)
    assert list(sampler) != batches


def test_batch_hard_triplet_loss() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    mixed = torch.tensor([[0.0, 0.0], [5.0, 5.0], [0.1, 0.0], [5.1, 5.0]])
    assert batch_hard_triplet_loss(separated, labels, margin=0.3).item() == 0.0
    assert batch_hard_triplet_loss(mixed, labels, margin=0.3).item() > 1.0


def test_retrieval_metrics_perfect_features() -> None:
    labels = np.repeat(np.arange(4), 3)
    features = np.eye(4)[labels].astype(np.float32)
    classes = ["car"] * 6 + ["bus"] * 6
    metrics = retrieval_metrics(features, labels, classes, seed=0)
    assert metrics["overall"] == {"queries": 4, "mAP": 1.0, "rank1": 1.0}
    assert metrics["per_class"]["bus"]["queries"] == 2


def run_script(name: str, *args: str) -> str:
    import subprocess
    import sys

    from safetrajai.paths import PROJECT_ROOT

    script = PROJECT_ROOT / "scripts" / "reid_train" / name
    result = subprocess.run([sys.executable, str(script), *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                            timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout


def test_train_one_epoch_then_load_with_custom_reid(tmp_path) -> None:
    import json

    from safetrajai.reid.encoders import Custom_ReID

    manifest, crop_root = write_synthetic_version(tmp_path, IDENTITIES)
    common = ["--version", "vtest", "--splits-root", str(tmp_path / "splits"), "--pack-dir", str(tmp_path / "pack")]
    run_script("pack_manifest.py", *common, "--crop-root", str(crop_root))

    out_dir = tmp_path / "class_reid_weights" / "smoke"
    run_script("train_reid.py", *common, "--output", str(out_dir), "--arch", "resnet18", "--no-pretrained",
               "--input-size", "32", "--epochs", "1", "--p", "2", "--k", "2", "--workers", "0", "--seed", "7")

    record = json.loads((out_dir / "train_record.json").read_text(encoding="utf-8"))
    assert record["manifest_version"] == "vtest"
    assert record["seed"] == 7
    assert record["arch"] == "resnet18" and record["input_size"] == 32
    assert record["best_epoch"] == 1 and len(record["epochs"]) == 1

    encoder = Custom_ReID(str(out_dir / "model.pt"))
    frame = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    feats = encoder(frame, np.array([[40.0, 40.0, 20.0, 30.0, 0], [100.0, 60.0, 30.0, 20.0, 1]]))
    assert len(feats) == 2 and feats[0].shape == (record["embed_dim"],)

    result = json.loads(run_script("eval_reid.py", *common, "--weights", str(out_dir / "model.pt"), "--workers", "0"))
    assert result["overall"]["queries"] == 2


def test_pack_reuses_other_version_and_reads_only_missing(tmp_path, monkeypatch) -> None:
    manifest, crop_root = write_synthetic_version(tmp_path, IDENTITIES)
    rows = read_manifest(manifest)
    pack_manifest(rows[:-2], crop_root, tmp_path / "old", "vold")

    disk_reads = []
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: disk_reads.append(self.name) or real(self))
    info = pack_manifest(rows, crop_root, tmp_path / "new", "vtest", reuse=CropPack(tmp_path / "old"))
    monkeypatch.undo()

    assert info["read_from_disk"] == len(disk_reads) == 2
    assert info["reused_from"] == "vold"
    pack = CropPack(tmp_path / "new")
    verify_pack(pack, rows)
    for row in rows:
        assert np.array_equal(pack.read_bgr(row["crop_path"]), cv2.imread(str(crop_root / row["crop_path"])))
