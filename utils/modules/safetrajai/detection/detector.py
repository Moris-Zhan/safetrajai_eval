"""
Detector — YOLO 偵測封裝。

負責單一 frame 的推論與逐類別信心過濾，供 safetrajai/pipeline/sequence_tracker.py 的追蹤迴圈呼叫。
訓練腳本目前不在此專案範圍內（YOLO 權重為外部訓練產物）。
"""

import numpy as np
from ultralytics import YOLO

from safetrajai.tracking.config import get_conf_thresh_for_class, get_min_conf_thresh


class Detector:
    """包裝 ultralytics YOLO 模型：以全類別最低信心門檻推論，再逐偵測套用類別專屬門檻過濾。"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray, iou: float, imgsz: int):
        """
        對單一 frame 執行推論並回傳逐類別信心過濾後的 boxes。

        Returns
        -------
        boxes : ultralytics.engine.results.Boxes
        """
        min_conf = get_min_conf_thresh()
        results = self.model.predict(frame, conf=min_conf, iou=iou, imgsz=imgsz, verbose=False)[0]

        boxes = results.boxes
        if len(boxes) > 0:
            keep = [
                i for i in range(len(boxes))
                if float(boxes.conf[i]) >= get_conf_thresh_for_class(int(boxes.cls[i]))
            ]
            boxes = boxes[keep] if keep else boxes[:0]
        return boxes
