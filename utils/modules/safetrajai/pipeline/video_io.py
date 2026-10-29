"""Annotated video writer with ffmpeg H.264 re-encoding."""

import os
from pathlib import Path

import cv2
import numpy as np


class VideoWriterWithFfmpeg:
    """包裝 cv2.VideoWriter，close() 時額外做 ffmpeg H.264 轉檔
    （失敗則保留原始 mp4v 檔並印警告，行為與現行一致）。

    ⚠️ 建構子的 cv2.VideoWriter fps 沿用現行寫死值 5（不是傳入的 fps 參數）——
    這是既有行為，重構時不修正，只搬移位置。
    """

    def __init__(self, output_path: Path, fps: int, width: int, height: int):
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(output_path), fourcc, 5, (width, height))
        print(f"Output video : {output_path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()
        if not self.output_path.exists():
            return

        ffmpeg_out = self.output_path.with_stem(self.output_path.stem + "_h264")
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{self.output_path}" '
            f'-vcodec libx264 -pix_fmt yuv420p -acodec aac "{ffmpeg_out}"'
        )
        print(f"  ffmpeg    : {ffmpeg_cmd}")
        ret = os.system(ffmpeg_cmd)
        if ret == 0 and ffmpeg_out.exists():
            self.output_path.unlink()
            ffmpeg_out.rename(self.output_path)
            print(f"  video     : {self.output_path}")
        else:
            print("  ffmpeg conversion failed; keeping original mp4v file.")
