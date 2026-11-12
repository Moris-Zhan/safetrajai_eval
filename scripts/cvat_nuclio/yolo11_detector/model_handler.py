"""YOLO11 封裝：與 utils/modules/safetrajai/detection/detector.py 的推論邏輯一致
（相同權重、conf/iou/imgsz、逐類別信心門檻），供 CVAT 自動標註呼叫。獨立於主
專案套件之外，不依賴 safetrajai 套件匯入，與 CVAT 其他 nuclio 函式的慣例一致。
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "model" / "yolo11x_dennis.pt"

# 數值來源：config/tracking/botsort_reid_timm.yaml 的 tracking_params.iou/imgsz 與
# class_tracking.class_conf_thresh；該 config 改動時要手動同步這裡。
IOU = 0.50
IMGSZ = 640
CLASS_CONF_THRESH = {
    "person": 0.15,
    "car": 0.15,
    "scooter": 0.10,
    "small_truck": 0.15,
    "bus": 0.15,
    "truck": 0.15,
}
MIN_CONF_THRESH = min(CLASS_CONF_THRESH.values())


class ModelHandler:
    def __init__(self, labels: dict):
        self.labels = labels
        self.model = YOLO(str(MODEL_PATH))

    def infer(self, image: np.ndarray) -> list:
        results = self.model.predict(image, conf=MIN_CONF_THRESH, iou=IOU, imgsz=IMGSZ, verbose=False)[0]
        boxes = results.boxes

        detections = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            label = self.labels.get(cls_id, "unknown")
            if conf < CLASS_CONF_THRESH.get(label, MIN_CONF_THRESH):
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            detections.append(
                {
                    "confidence": str(conf),
                    "label": label,
                    "points": [x1, y1, x2, y2],
                    "type": "rectangle",
                }
            )
        return detections
