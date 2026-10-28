"""Run stage2 (TrackEval + Excel export) on an existing stage1 output tree."""

import argparse

from safetrajai.evaluation.config_loader import load_config
from safetrajai.evaluation.pipeline import run_eval_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MOT/TrackEval pipeline with specified config.")
    parser.add_argument("--config", type=str, default="config/bytetrack.yaml", help="Path to config yaml file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_eval_pipeline(cfg)


if __name__ == "__main__":
    main()
