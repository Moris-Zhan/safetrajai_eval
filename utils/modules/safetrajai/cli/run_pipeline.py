"""CLI for the stage1/stage2/stage3 pipeline (shared by track_eval.py and scripts/pipeline/run_pipeline.py)."""

import argparse

from safetrajai.tracking.config import get_track_config_path, load_track_config
from safetrajai.evaluation.config_loader import load_config
from safetrajai.pipeline.tracking_pipeline import TrackingEvalPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO class-aware tracking + TrackEval evaluation"
    )

    # 影片儲存控制
    parser.add_argument("--config", default="config/tracking/bytetrack.yaml", help="流程用 config yaml（路徑、eval、class_tracking 等）")
    parser.add_argument(
        "--track-config",
        default=None,
        help=(
            "class-aware tracker 設定（CLASS_TRACKING_PARAMS、置信度門檻等）YAML 路徑。"
            "不指定時使用 --config 內建的 class_tracking 章節。"
        ),
    )

    args = parser.parse_args()

    load_track_config(args.track_config or args.config)
    print(f"Loaded track config: {get_track_config_path()}")

    # 允許 CLI 覆蓋路徑常數
    cfg = load_config(args.config)

    pipeline = TrackingEvalPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
