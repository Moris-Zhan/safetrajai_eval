"""Pickle-compatibility shim (see docs/adr/0001-keep-utils-model-pickle-shim.md).

models/class_reid_weights/*/best_full_model.pt are full-model pickles that
reference classes as ``utils.model.<Class>``. The implementation lives in
safetrajai.reid.model; do not delete this module.
"""

from safetrajai.reid.model import *  # noqa: F401,F403
from safetrajai.reid.model import CameraEmbedding, OcclusionRepair, ReIDNet  # noqa: F401
