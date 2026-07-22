"""
Custom Tracker module entry point.
Provides unified class-aware tracker API.
"""
from .class_aware_botsort import ClassAwareBOTSORT
from .class_aware_bytetrack import ClassAwareBYTETracker
from .config import (
    CLASS_TRACKING_PARAMS,
    CLASS_NAMES,
    CLASS_CONF_THRESH,
    get_params_for_class,
    get_conf_thresh_for_class,
    get_min_conf_thresh,
    load_track_config,
    get_track_config_path,
)

# Default tracker configuration (compatible with Ultralytics args format)
DEFAULT_TRACKER_CONFIG = {
    # Detection thresholds
    'track_high_thresh': 0.15,
    'track_low_thresh': 0.05,
    'new_track_thresh': 0.15,

    # Matching thresholds
    'match_thresh': 0.85,

    # Track buffer (C1: increased from 30 to 120 for better occlusion handling)
    'track_buffer': 120,

    # BoTSORT specific
    'gmc_method': 'sparseOptFlow',
    'proximity_thresh': 0.5,
    'appearance_thresh': 0.4,
    # FINAL CONFIG: ReID disabled (優先 Scooter 追蹤)
    # 實驗結論: ReID ON 提升 Person (+5%) 但損害 Scooter (-4%) 和增加 IDSW
    'with_reid': False,
    'model': None,

    # Common
    # fuse_score 會將偵測信心融入匹配成本，低信心偵測（如 scooter）會被額外懲罰
    # 關閉以改善低信心物件的召回率
    'fuse_score': False,
}


class SimpleNamespace:
    """Simple namespace class for tracker args."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def create_tracker(tracker_type='botsort', frame_rate=30, **kwargs):
    """
    Create a class-aware tracker instance.

    Args:
        tracker_type: 'botsort' or 'bytetrack'
        frame_rate: Video frame rate (default: 30)
        **kwargs: Override default tracker configuration

    Returns:
        Class-aware tracker instance

    Example:
        >>> tracker = create_tracker(tracker_type='botsort')
        >>> # Use with custom settings
        >>> tracker = create_tracker(
        ...     tracker_type='botsort',
        ...     track_high_thresh=0.1,
        ...     track_buffer=120
        ... )
    """
    # Merge default config with user overrides
    config = DEFAULT_TRACKER_CONFIG.copy()
    config.update(kwargs)

    # Create args namespace
    args = SimpleNamespace(**config)

    if tracker_type == 'botsort':
        return ClassAwareBOTSORT(args=args, frame_rate=frame_rate)
    elif tracker_type == 'bytetrack':
        return ClassAwareBYTETracker(args=args, frame_rate=frame_rate)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}. Use 'botsort' or 'bytetrack'.")


__all__ = [
    'create_tracker',
    'ClassAwareBOTSORT',
    'ClassAwareBYTETracker',
    'CLASS_TRACKING_PARAMS',
    'CLASS_NAMES',
    'CLASS_CONF_THRESH',
    'get_params_for_class',
    'get_conf_thresh_for_class',
    'get_min_conf_thresh',
    'load_track_config',
    'get_track_config_path',
    'DEFAULT_TRACKER_CONFIG',
]
