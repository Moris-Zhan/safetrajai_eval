"""
Class-specific tracking parameter configuration.
Supports future ReID model extension.

Parameter Reference:
====================
- match_thresh: First association matching threshold (cost = 1 - IoU)
                Higher value = more lenient matching (accepts lower IoU)
                e.g., 0.90 accepts IoU >= 0.10, 0.85 accepts IoU >= 0.15

- appearance_thresh: ReID appearance similarity threshold (only used when use_reid=True)
                     Lower value = stricter appearance matching

- new_track_thresh: Minimum score to initialize a new track
                    NOTE: Must be >= track_high_thresh (default 0.15) to have effect

- second_match_thresh: Second association (low-confidence detections) threshold
                       Used for IoU-only matching with low-conf detections

- unconfirmed_match_thresh: Unconfirmed tracks matching threshold
                            Used for tracks with only one frame of history

- track_buffer: Per-class max frames to keep lost track (overrides global setting)

- use_reid: Whether to use ReID features for this class (requires global with_reid=True)
"""

# YOLO class definitions (0-indexed)
CLASS_NAMES = {
    0: 'person',
    1: 'car',
    2: 'scooter',
    3: 'small_truck',
    4: 'bus',
    5: 'truck'
}

# =============================================================================
# FINAL CONFIGURATION: BoTSORT ClassAware (優先 Scooter 追蹤)
# =============================================================================
# 實驗結論:
#   - ReID OFF: Scooter MOTA 42.06%, IDSW 5
#   - ReID ON:  Scooter MOTA 38.10%, IDSW 11
# 選擇 ReID OFF 以優化 Scooter 追蹤和減少 ID Switch
# =============================================================================

CLASS_TRACKING_PARAMS = {
    'car': {
        'center_dist_weight': 0.4,
        'fuse_score': True,
        'match_thresh': 0.7,
        'new_track_thresh': 0.25,
        'norm_factor': 80.0,
        'second_match_thresh': 0.6,
        'unconfirmed_match_thresh': 0.9,        
        # 'use_reid': True,
        # 'proximity_thresh': 0.3,
        # 'appearance_thresh': 0.5,
        # 'model': 'runs/triplet-yolo/train-007/best.pt',
    },
    'bus': {
        'center_dist_weight': 0.25,    # 中心點距離佔 25%，IoU 佔 75%
        'fuse_score': False,
        'match_thresh': 0.7,
        'norm_factor': 200.0,          # 依你的畫面解析度與機車典型移動幅度調整
        'new_track_thresh': 0.15,
        'second_match_thresh': 0.55,
        'unconfirmed_match_thresh': 0.95,        
        # 'use_reid': True,
        # 'proximity_thresh': 0.5,
        # 'appearance_thresh': 0.45,
        # 'model': 'runs/triplet-yolo/train-007/best.pt',
    },
    'small_truck': {
        'center_dist_weight': 0.2,    # 中心點距離佔 25%，IoU 佔 75%
        'fuse_score': True,
        'match_thresh': 0.6,
        'norm_factor': 100.0,          # 依你的畫面解析度與機車典型移動幅度調整
        'new_track_thresh': 0.15,
        'second_match_thresh': 0.45,
        'unconfirmed_match_thresh': 0.9,
        # 'use_reid': True,
        # 'proximity_thresh': 0.5,
        # 'appearance_thresh': 0.4,
        # 'model': 'runs/triplet-yolo/train-007/best.pt',
    },
    'truck': {
        'center_dist_weight': 0.2,    # 中心點距離佔 25%，IoU 佔 75%
        'fuse_score': False,
        'match_thresh': 0.6,
        'norm_factor': 100.0,          # 依你的畫面解析度與機車典型移動幅度調整
        'new_track_thresh': 0.15,
        'second_match_thresh': 0.45,
        'unconfirmed_match_thresh': 0.9,
        # 'use_reid': True,
        # 'proximity_thresh': 0.5,
        # 'appearance_thresh': 0.45,
        # 'model': 'runs/triplet-yolo/train-007/best.pt',
    },
    'person': {
        'center_dist_weight': 0.25,
        'fuse_score': False,
        'match_thresh': 0.9,
        'new_track_thresh': 0.25,
        'norm_factor': 150.0,
        'second_match_thresh': 0.5,
        'unconfirmed_match_thresh': 0.85,
        # 'use_reid': True,
        # 'proximity_thresh': 0.5,        
        # 'appearance_thresh': 0.4,
        # 'model': 'runs/triplet-yolo/train-007/best.pt',
    },   
     'scooter': {
        'center_dist_weight': 0.25,
        'fuse_score': True,
        'match_thresh': 0.9,
        'new_track_thresh': 0.1,
        'norm_factor': 200.0,
        'second_match_thresh': 0.4,
        'unconfirmed_match_thresh': 0.8,
        # 'use_reid': True,
        # 'proximity_thresh': 0.1,
        # 'appearance_thresh': 0.8,
        # 'model': 'runs/triplet-yolo/train-007/best.pt',
    },  
}

# Default parameters (for unknown classes)
DEFAULT_TRACKING_PARAMS = {
    'match_thresh': 0.80,
    'appearance_thresh': 0.35,
    'new_track_thresh': 0.15,
    'second_match_thresh': 0.5,
    'unconfirmed_match_thresh': 0.7,
    'use_reid': False,
    'model': './yolo11x-cls-reid.pt'    
}
# run_tracking_with_model("./yolo11x-cls-reid.pt")

# Class-specific confidence thresholds for detection
# scooter 降低至 0.10，搭配改善後的 tracker 參數（second_match_thresh=0.75,
# unconfirmed_match_thresh=0.85, fuse_score=False）讓更多低信心偵測被有效關聯
CLASS_CONF_THRESH = {
    'person': 0.15,
    'car': 0.15,
    'scooter': 0.10,      # 降低門檻，救回更多 scooter 偵測
    'small_truck': 0.15,
    'bus': 0.15,
    'truck': 0.15,
}

# Default confidence threshold
DEFAULT_CONF_THRESH = 0.15


def get_params_for_class(cls_id):
    """Get tracking parameters for a specific class ID.

    Args:
        cls_id: Class ID (int) or class name (str)

    Returns:
        dict: Tracking parameters for the class
    """
    if isinstance(cls_id, str):
        cls_name = cls_id
    else:
        cls_name = CLASS_NAMES.get(int(cls_id))

    if cls_name and cls_name in CLASS_TRACKING_PARAMS:
        return CLASS_TRACKING_PARAMS[cls_name]
    return DEFAULT_TRACKING_PARAMS


def get_class_name(cls_id):
    """Get class name from class ID."""
    return CLASS_NAMES.get(int(cls_id), f'class_{cls_id}')


def get_conf_thresh_for_class(cls_id):
    """Get confidence threshold for a specific class ID.

    Args:
        cls_id: Class ID (int) or class name (str)

    Returns:
        float: Confidence threshold for the class
    """
    if isinstance(cls_id, str):
        cls_name = cls_id
    else:
        cls_name = CLASS_NAMES.get(int(cls_id))

    if cls_name and cls_name in CLASS_CONF_THRESH:
        return CLASS_CONF_THRESH[cls_name]
    return DEFAULT_CONF_THRESH


def get_min_conf_thresh():
    """Get the minimum confidence threshold across all classes.

    Use this as the global conf threshold for YOLO inference,
    then filter by class-specific thresholds in post-processing.
    """
    return min(CLASS_CONF_THRESH.values())
