"""Compatibility entry point: `python track_eval.py --config <path>`.

The pipeline lives in safetrajai.pipeline; the CLI is safetrajai.cli.run_pipeline
(also exposed as scripts/run_pipeline.py). Output layout is documented in
utils/modules/safetrajai/pipeline/tracking_pipeline.py.
"""

import sys

INSTALL_HINT = "safetrajai is not installed. Run: conda activate track_eval && pip install -e . --no-deps"

try:
    from safetrajai.cli.run_pipeline import main
except ModuleNotFoundError as error:
    if error.name != "safetrajai" and not (error.name or "").startswith("safetrajai."):
        raise
    print(INSTALL_HINT, file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
