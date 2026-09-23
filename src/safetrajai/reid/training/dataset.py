"""
Re-ID 訓練資料讀取（對應 track_eval.py 的 save_reid_crops 蒐集結果）。

目錄假設
--------
{root}/{class_name}/{identity_folder}/*.jpg

每個 identity_folder 是一個身分（同一支影片裡同一個 track_id 的所有裁切影像）。
資料夾命名由 track_eval.py 產生，格式為 "{影片檔名}_track{track_id}"，但這裡不
依賴這個命名規則本身——只要是同一個資料夾底下的影像就視為同一身分，資料夾名稱
只當作身分的唯一識別字串。

與舊版 dataset/reid/crop_dataset/train/{train,val}/{identity}/ 的差異：
- 舊版沒有 class_name 這一層，所有類別的身分混在一起。
- 舊版的 train/val 是資料夾層級先切好的；這裡改成用 split_identities() 依身分
  （而非影像）隨機切分，避免同一身分的照片同時出現在 train 和 val 造成資料外洩。
- 這裡的 Dataset 只服務單一 class_name（一次只訓練一個類別的模型），跟
  track_config 逐類別指定 model 路徑的慣例一致。
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, Sampler


def discover_identities(root: str | Path, class_name: str) -> list[Path]:
    """回傳 {root}/{class_name}/ 底下每個身分資料夾的路徑（已排序，確保可重現）。"""
    class_dir = Path(root) / class_name
    if not class_dir.is_dir():
        raise FileNotFoundError(f"找不到類別資料夾: {class_dir}")
    identity_dirs = sorted(p for p in class_dir.iterdir() if p.is_dir())
    if not identity_dirs:
        raise ValueError(f"{class_dir} 底下沒有任何身分資料夾")
    return identity_dirs


def split_identities(
    identity_dirs: list[Path],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """依身分（資料夾）隨機切 train/val，同一身分的所有影像只會出現在其中一邊。"""
    rng = random.Random(seed)
    shuffled = list(identity_dirs)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_dirs = shuffled[:n_val]
    train_dirs = shuffled[n_val:]
    return train_dirs, val_dirs


class ReIDCropDataset(Dataset):
    """
    單一類別的 Re-ID 訓練資料集。

    Parameters
    ----------
    identity_dirs : list[Path]
        身分資料夾清單（通常是 split_identities() 回傳的其中一半）。
    transform : callable | None
        套用在 PIL Image 上的 torchvision transform。
    min_images_per_identity : int
        身分資料夾內影像數低於此值就整個身分捨棄（太少張無法組成有效的 positive pair）。
    """

    def __init__(
        self,
        identity_dirs: list[Path],
        transform=None,
        min_images_per_identity: int = 2,
    ):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.identity_names: list[str] = []

        label = 0
        for identity_dir in identity_dirs:
            img_paths = sorted(
                p for p in identity_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            if len(img_paths) < min_images_per_identity:
                continue
            for p in img_paths:
                self.samples.append((p, label))
            self.identity_names.append(identity_dir.name)
            label += 1

        if not self.samples:
            raise ValueError(
                f"沒有任何身分資料夾的影像數 >= {min_images_per_identity}，"
                "無法組成訓練集（檢查資料是否蒐集足夠，或降低 min_images_per_identity）"
            )

    @property
    def num_identities(self) -> int:
        return len(self.identity_names)

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class PKSampler(Sampler[list[int]]):
    """
    每個 batch 固定抽 P 個身分 × K 張影像（batch-hard triplet mining 的標準抽樣方式）。
    身分影像數不足 K 張時用取後放回抽樣補足；跟 DataLoader 搭配時傳入 batch_sampler=。

    Examples
    --------
    >>> loader = DataLoader(dataset, batch_sampler=PKSampler(dataset.labels, P=8, K=4))
    """

    def __init__(self, labels: list[int], P: int, K: int, iterations: int | None = None):
        self.P = P
        self.K = K
        self.label_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        self.unique_labels = list(self.label_to_indices.keys())
        if len(self.unique_labels) < 2:
            raise ValueError("PKSampler 至少需要 2 個身分才能組出有效的 triplet")
        self.P = min(P, len(self.unique_labels))
        self.iterations = iterations or max(1, len(self.unique_labels) // self.P)

    def __iter__(self):
        for _ in range(self.iterations):
            batch: list[int] = []
            chosen_labels = random.sample(self.unique_labels, self.P)
            for label in chosen_labels:
                indices = self.label_to_indices[label]
                if len(indices) >= self.K:
                    batch.extend(random.sample(indices, self.K))
                else:
                    batch.extend(random.choices(indices, k=self.K))
            yield batch

    def __len__(self) -> int:
        return self.iterations
