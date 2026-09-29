"""manifest 資料集與只讀一次的本機打包快取（design D2）。

快取目錄（預設 ~/.cache/safetrajai/reid_packs/<version>/）：
  - crops.bin    manifest 列出的每張裁切的 JPEG 位元組，依 manifest 順序串接
  - index.csv    crop_path,offset,length
  - pack.json    資料集版本、manifest 列數、打包張數、位元組數、從磁碟讀取張數、複用的版本、耗時；
                 最後寫出，存在即代表打包完成
訓練與 val 只經由快取讀取影像，快取只含 manifest 列出的裁切。
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from safetrajai.reid.dinov2_features import letterbox

PACK_DATA = "crops.bin"
PACK_INDEX = "index.csv"
PACK_INFO = "pack.json"


def read_manifest(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def manifest_version(path: Path) -> str:
    return Path(path).parent.name


def default_pack_dir(version: str) -> Path:
    return Path.home() / ".cache" / "safetrajai" / "reid_packs" / version


def pack_manifest(rows: list[dict], crop_root: Path, pack_dir: Path, version: str,
                  reuse: "CropPack | None" = None) -> dict:
    """依 manifest 從 crop_root 各讀一次裁切，寫成本機快取；快取已存在時報錯。

    reuse：另一個版本的快取；其中已有的 crop_path 直接複製位元組，只從 crop_root 補讀缺少的裁切。
    """
    pack_dir = Path(pack_dir)
    if (pack_dir / PACK_INFO).exists():
        raise FileExistsError(f"快取已存在，要重建請先移走：{pack_dir}")
    pack_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    offset = 0
    from_disk = 0
    with (pack_dir / PACK_DATA).open("wb") as data_file, (pack_dir / PACK_INDEX).open("w", newline="", encoding="utf-8") as index_file:
        writer = csv.writer(index_file)
        writer.writerow(["crop_path", "offset", "length"])
        for i, row in enumerate(rows, 1):
            if reuse is not None and row["crop_path"] in reuse:
                payload = reuse.read_bytes(row["crop_path"])
            else:
                payload = (Path(crop_root) / row["crop_path"]).read_bytes()
                from_disk += 1
            data_file.write(payload)
            writer.writerow([row["crop_path"], offset, len(payload)])
            offset += len(payload)
            if i % 10000 == 0:
                print(f"  packed {i}/{len(rows)}")

    info = {
        "version": version, "manifest_rows": len(rows), "packed": len(rows), "bytes": offset,
        "read_from_disk": from_disk, "reused_from": reuse.info["version"] if reuse is not None else None,
        "seconds": round(time.time() - start, 1),
    }
    (pack_dir / PACK_INFO).write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


class CropPack:
    """Read-only view over a finished pack; images are decoded as BGR like cv2.imread."""

    def __init__(self, pack_dir: Path):
        pack_dir = Path(pack_dir)
        info_path = pack_dir / PACK_INFO
        if not info_path.exists():
            raise FileNotFoundError(f"快取不存在或未完成，請先執行 pack_manifest.py：{pack_dir}")
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        with (pack_dir / PACK_INDEX).open(newline="", encoding="utf-8") as f:
            self.index = {r["crop_path"]: (int(r["offset"]), int(r["length"])) for r in csv.DictReader(f)}
        self._path = pack_dir / PACK_DATA
        self._data = None

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, crop_path: str) -> bool:
        return crop_path in self.index

    def read_bytes(self, crop_path: str) -> bytes:
        if self._data is None:
            self._data = np.memmap(self._path, dtype=np.uint8, mode="r")
        offset, length = self.index[crop_path]
        return self._data[offset:offset + length].tobytes()

    def read_bgr(self, crop_path: str) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.read_bytes(crop_path), dtype=np.uint8), cv2.IMREAD_COLOR)


def verify_pack(pack: CropPack, rows: list[dict]) -> None:
    wanted = {r["crop_path"] for r in rows}
    if len(pack) != len(rows) or set(pack.index) != wanted:
        raise ValueError(f"快取內容與 manifest 不一致：快取 {len(pack)} 張、manifest {len(rows)} 列")


def identity_labels(rows: list[dict]) -> dict[tuple[str, str], int]:
    """(class, identity) -> contiguous label shared by all classes."""
    return {key: i for i, key in enumerate(sorted({(r["class"], r["identity"]) for r in rows}))}


def build_transform(mean: list[float], std: list[float], train: bool) -> v2.Compose:
    steps = [v2.RandomHorizontalFlip(), v2.ColorJitter(0.2, 0.2, 0.2)] if train else []
    steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    if train:
        steps.append(v2.RandomErasing(p=0.5, value=0))
    return v2.Compose(steps)


class ManifestDataset(Dataset):
    """Items are (image tensor, label, row index); images come only from the pack."""

    def __init__(self, rows: list[dict], pack: CropPack, labels: dict[tuple[str, str], int],
                 input_size: int, transform: v2.Compose):
        self.rows = rows
        self.pack = pack
        self.labels = [labels[(r["class"], r["identity"])] for r in rows]
        self.input_size = input_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        rgb = letterbox(self.pack.read_bgr(self.rows[i]["crop_path"]), self.input_size)
        image = self.transform(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
        return image, self.labels[i], i
