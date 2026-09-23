import cv2
import numpy as np
from pathlib import Path

from safetrajai.classes import CLASS_NAMES

NAMES = CLASS_NAMES


def save_mot_format(imgs_path, labels_path, output_file, img_width: int, img_height: int,
    det_corner: np.array = None):

    if det_corner is not None:
        polygon = det_corner
    else:
        polygon = np.array([
            [0, 0],
            [img_width, 0],
            [img_width, img_height],
            [0, img_height],
        ], dtype=np.float32)

    img_files = sorted(Path(imgs_path).glob("C000025_*.jpg"))

    img_paths_txt = Path(output_file).parent / Path(output_file).name.replace("annotation_", "")
    with open(img_paths_txt, 'w') as out_f:
        for img_file in img_files:
            # 由 img_file.name 往前取得倒數第三層路徑
            # path_3rd_from_last = img_file.parent.parent.parent /
            out_f.write(f"{img_file}\n")
    print(f"img files list saved to: {img_paths_txt}")

    label_files = sorted(Path(labels_path).glob("C000025_*.txt"))

    # 建立一個多類別（multi-class）的 dict 以便依據 cls_id 累積 MOT 欄位資訊，例：
    mot_by_class = dict()
    # 這段 loop 已經偵測 cls_id
    with open(output_file, 'w') as out_f:
        for frame_num, label_file in enumerate(label_files, start=1):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7:
                        cls_id = int(parts[0])
                        cx = float(parts[1]) * img_width
                        cy = float(parts[2]) * img_height
                        w = float(parts[3]) * img_width
                        h = float(parts[4]) * img_height
                        track_id = int(parts[5])
                        score = float(parts[6])

                        left = cx - w / 2
                        top = cy - h / 2
                        mot_class = cls_id + 1

                        result = cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False)

                        # 只允許在polygon裡的bbox才保存
                        if result >= 0:
                            line_txt = (
                                f"{frame_num},{track_id},{left:.2f},{top:.2f},"
                                f"{w:.2f},{h:.2f},{score:.4f}\n"
                            )
                            if cls_id not in mot_by_class:
                                mot_by_class[cls_id] = []
                            mot_by_class[cls_id].append(line_txt)
                            # 同時繼續保存到 out_f 以維持現有寫檔行為
                            out_f.write(line_txt)

    print(f"MOT format saved to: {output_file}")

    # 根據 mot_by_class，把每個類別的軌跡內容寫入各自對應的檔案
    for cls_id in NAMES.keys():
        class_out_file = Path(f"{labels_path.parent.parent}/trackers/mot_challenge/CCTV-val/YOLO11x__{NAMES[cls_id]}/data/{labels_path.stem}.txt")
        class_out_file.parent.mkdir(parents=True, exist_ok=True)
        lines = mot_by_class.get(cls_id, [])
        with open(class_out_file, "w") as class_f:
            class_f.writelines(lines)

def write_det_data_yaml(out_root: Path, names: dict, split: str = "val") -> Path:
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        path_str = out_root.as_posix().replace("runs/detect/", "./中華三路_民生二路(東北角)yolo驗證用_all8/")
        f.write(f'path: "{path_str}"\n')
        f.write(f'train: "images/{Path(out_root).stem.replace("annotation_", "")}"\n')
        f.write(f'val: "images/{Path(out_root).stem.replace("annotation_", "")}"\n')
        f.write("names:\n")
        for k in sorted(names.keys()):
            f.write(f"  {int(k)}: {names[k]}\n")
    return yaml_path


def make_det_eval_dataset(out_dir: Path, names: dict, split: str = "val"):
    yaml_path = write_det_data_yaml(out_dir, names=names, split=split)
    print("✅ data.yaml:", yaml_path.resolve())
