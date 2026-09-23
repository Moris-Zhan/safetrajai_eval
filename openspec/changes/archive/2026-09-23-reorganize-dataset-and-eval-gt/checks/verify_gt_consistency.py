"""One-off check for task 3.3: per-frame box counts of the YOLO GT must equal the MOT GT.

YOLO label files are 0-based (`*_f000000.txt`), MOT GT frames are 1-based.
Run from the repo root: python openspec/changes/reorganize-dataset-and-eval-gt/checks/verify_gt_consistency.py
"""

from collections import Counter
from pathlib import Path

EVAL_ROOT = Path("dataset/eval/cctv_all8")


def yolo_counts(labels_dir: Path) -> Counter:
    return Counter({
        int(p.stem.rsplit("_f", 1)[1]) + 1: sum(1 for line in p.read_text().splitlines() if line.strip())
        for p in labels_dir.glob("*.txt")
    })


def mot_counts(gt_txt: Path) -> Counter:
    return Counter(int(line.split(",")[0]) for line in gt_txt.read_text().splitlines() if line.strip())


def main() -> None:
    total_yolo = total_mot = 0
    all_match = True
    for seq_dir in sorted((EVAL_ROOT / "annotations_all8").glob("annotation_*")):
        seq = seq_dir.name[len("annotation_"):]
        yolo = yolo_counts(seq_dir / "labels" / seq)
        mot = mot_counts(EVAL_ROOT / "mot_annotations_all8" / f"mot_annotation_{seq}" / "gt" / "gt.txt")
        frames = sorted(set(yolo) | set(mot))
        mismatched = [f for f in frames if yolo.get(f, 0) != mot.get(f, 0)]
        all_match &= not mismatched
        total_yolo += sum(yolo.values())
        total_mot += sum(mot.values())
        print(f"{seq}: frames={len(frames)} yolo={sum(yolo.values())} mot={sum(mot.values())} mismatched_frames={mismatched[:5]}")
    print(f"TOTAL yolo={total_yolo} mot={total_mot} all_frames_match={all_match}")


if __name__ == "__main__":
    main()
