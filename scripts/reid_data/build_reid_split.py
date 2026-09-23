#!/usr/bin/env python3
"""
dataset/reid/crop_dataset/ 底下的裁切影像 -> Re-ID train/val 資料集切分工具。

兩種模式：
  --mode category  6 個類別「各自」切一次 train/val，輸出到
                    dataset/reid/reid_category_dataset/{class}/{train,val}/{identity}/
  --mode flat      6 個類別的身分全部混在一起切「一次」train/val，輸出到
                    dataset/reid/reid_dataset/{train,val}/{class}_{identity}/
                    （身分資料夾名稱加上類別前綴，避免不同類別的 track_id 撞名——
                     class-aware tracker 的 track_id 是逐類別各自配號，同一支影片
                     同一個編號在不同類別代表不同物體）

依「身分」（資料夾）切分，不是依影像切分：同一身分的所有照片只會出現在 train 或
val 其中一邊，避免同一身分同時出現在兩邊造成資料外洩。

兩種模式跑完都會在各自輸出目錄下產生：
  - numbers.txt   每個身分一行 "{張數:>6d} {相對路徑}"
  - res.txt       tree -h 輸出，顯示到身分資料夾這一層（不展開個別影像）

用法：
    python dataset/reid/build_reid_split.py --mode category
    python dataset/reid/build_reid_split.py --mode flat --val-ratio 0.2 --seed 42
    python dataset/reid/build_reid_split.py --mode category --overwrite   # 清空重建既有輸出
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
from pathlib import Path

CLASSES = ["bus", "car", "person", "scooter", "small_truck", "truck"]
IMG_SUFFIXES = (".jpg", ".jpeg", ".png")


def discover_identities(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.is_dir())


def split_by_ratio(items: list, val_ratio: float, seed: int) -> tuple[list, list]:
    """依身分隨機切 train/val，同一身分只會出現在其中一邊。"""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_items = shuffled[:n_val]
    train_items = shuffled[n_val:]
    return train_items, val_items


def copy_identity(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in sorted(src_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_SUFFIXES:
            continue
        shutil.copy2(img_path, dst_dir / img_path.name)
        count += 1
    return count


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"{output_root} 已存在且非空，加 --overwrite 才會清空重建（避免誤刪既有產出）"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def write_numbers_txt(entries: list[tuple[int, str]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for count, label in entries:
            f.write(f"{count:>6d} {label}\n")


def write_res_txt(output_root: Path, depth: int, path: Path) -> None:
    result = subprocess.run(
        ["tree", "-h", "-L", str(depth), "."],
        cwd=output_root, capture_output=True, text=True,
    )
    path.write_text(result.stdout, encoding="utf-8")


def run_category_mode(crop_root: Path, output_root: Path, val_ratio: float, seed: int) -> None:
    entries: list[tuple[int, str]] = []
    total_identities = total_images = 0
    for cls in CLASSES:
        class_dir = crop_root / cls
        if not class_dir.is_dir():
            print(f"skip {cls}: {class_dir} not found")
            continue
        identity_dirs = discover_identities(class_dir)
        if not identity_dirs:
            print(f"skip {cls}: no identity folders under {class_dir}")
            continue
        train_dirs, val_dirs = split_by_ratio(identity_dirs, val_ratio, seed)
        for split_name, dirs in (("train", train_dirs), ("val", val_dirs)):
            for identity_dir in dirs:
                dst = output_root / cls / split_name / identity_dir.name
                count = copy_identity(identity_dir, dst)
                entries.append((count, f"{cls}/{split_name}/{identity_dir.name}"))
                total_identities += 1
                total_images += count
        print(f"  {cls}: {len(identity_dirs)} identities -> train {len(train_dirs)} / val {len(val_dirs)}")

    write_numbers_txt(entries, output_root / "numbers.txt")
    write_res_txt(output_root, depth=3, path=output_root / "res.txt")
    print(f"done: {total_identities} identities, {total_images} images -> {output_root}")


def run_flat_mode(crop_root: Path, output_root: Path, val_ratio: float, seed: int) -> None:
    pooled: list[tuple[str, Path]] = []
    for cls in CLASSES:
        class_dir = crop_root / cls
        if not class_dir.is_dir():
            print(f"skip {cls}: {class_dir} not found")
            continue
        for identity_dir in discover_identities(class_dir):
            pooled.append((f"{cls}_{identity_dir.name}", identity_dir))

    if not pooled:
        raise SystemExit(f"no identity folders found under {crop_root}")

    train_items, val_items = split_by_ratio(pooled, val_ratio, seed)
    entries: list[tuple[int, str]] = []
    total_images = 0
    for split_name, items in (("train", train_items), ("val", val_items)):
        for flat_name, identity_dir in items:
            dst = output_root / split_name / flat_name
            count = copy_identity(identity_dir, dst)
            entries.append((count, f"{split_name}/{flat_name}"))
            total_images += count

    write_numbers_txt(entries, output_root / "numbers.txt")
    write_res_txt(output_root, depth=2, path=output_root / "res.txt")
    print(
        f"done: {len(pooled)} identities (train {len(train_items)} / val {len(val_items)}), "
        f"{total_images} images -> {output_root}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["category", "flat"], required=True)
    parser.add_argument("--crop-root", default="dataset/reid/crop_dataset")
    parser.add_argument(
        "--output-root", default=None,
        help="預設 category -> dataset/reid/reid_category_dataset，flat -> dataset/reid/reid_dataset",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="輸出目錄已存在且非空時，清空重建",
    )
    args = parser.parse_args()

    crop_root = Path(args.crop_root)
    default_output = (
        "dataset/reid/reid_category_dataset" if args.mode == "category" else "dataset/reid/reid_dataset"
    )
    output_root = Path(args.output_root or default_output)

    prepare_output_root(output_root, args.overwrite)

    if args.mode == "category":
        run_category_mode(crop_root, output_root, args.val_ratio, args.seed)
    else:
        run_flat_mode(crop_root, output_root, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
