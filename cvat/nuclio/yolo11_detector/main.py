import base64
import json

import cv2
import numpy as np

from model_handler import ModelHandler

# 類別順序與 ID 對照 utils/modules/safetrajai/classes.py 的 CLASS_NAMES，也是
# function-gpu.yaml 的 metadata.annotations.spec 給 CVAT 的那份標籤清單。
# 這裡不讀 /opt/nuclio/function.yaml：在這個 nuctl/nuclio 版本的本機部署模式下，
# 容器內不會有這個檔案（已實測確認，不是猜測），直接內嵌避免執行期出錯。
LABELS = {
    0: "person",
    1: "car",
    2: "scooter",
    3: "small_truck",
    4: "bus",
    5: "truck",
}


def init_context(context):
    context.logger.info("Init context... 0%")
    context.user_data.model = ModelHandler(LABELS)
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run safetrajai YOLO11 detector")
    data = event.body
    image_bytes = base64.b64decode(data["image"])
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    results = context.user_data.model.infer(image)

    return context.Response(
        body=json.dumps(results), headers={}, content_type="application/json", status_code=200
    )
