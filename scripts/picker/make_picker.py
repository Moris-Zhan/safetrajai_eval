#!/usr/bin/env python3
"""
產生鏡頭的離線點選頁（單檔 HTML，影像以 base64 內嵌），在任何瀏覽器開啟後點選，頁面會即時產生 YAML 可複製或下載。

模式（--mode）：
    homography       左（CCTV 首幀）右（衛星圖）成對點選 ≥ 4 組 → cctv_points／map_points，貼進 config/homography/<camera>.yaml
    checkpoints      成對點選驗證點（check points） → config/homography_checkpoints/<camera>.yaml
    counting-lines   在 CCTV 首幀點兩點成一條計數線，可填名稱與方向標籤 → config/counting_lines/<camera>.yaml
    camera-position  在衛星圖上點相機位置 → camera_view_point，貼進 config/homography/<camera>.yaml

輸出放在 gitignore 的 runs/picker/，CCTV 與衛星圖不進 git、不上傳。不會修改任何 config。

用法：
    python scripts/picker/make_picker.py --camera C000054 --mode homography
    python scripts/picker/make_picker.py --camera C000025 --mode counting-lines --out-dir runs/picker
"""
from __future__ import annotations

import argparse

from safetrajai.picker.build import DEFAULT_OUT_DIR, PickerInputError, write_picker_page
from safetrajai.picker.page import MODES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", required=True, help="鏡頭編號，例如 C000054")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    try:
        out_path = write_picker_page(args.camera, args.mode, out_dir=args.out_dir)
    except PickerInputError as error:
        raise SystemExit(f"錯誤：{error}")
    print(f"{args.camera} [{args.mode}] -> {out_path}")


if __name__ == "__main__":
    main()
