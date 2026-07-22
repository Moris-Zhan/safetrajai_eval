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
    'person': {
        'match_thresh': 0.85,
        'appearance_thresh': 0.40,
        'proximity_thresh': 0.5,        # 獨立放寬 IoU（原全域值通常 0.5）
        'center_dist_weight': 0.25,    # 中心點距離佔 25%，IoU 佔 75%
        'norm_factor': 150.0,          # 依你的畫面解析度與機車典型移動幅度調整
        'new_track_thresh': 0.15,
        'second_match_thresh': 0.5,      # Second association (low-conf) threshold
        'unconfirmed_match_thresh': 0.95, # Unconfirmed tracks matching threshold
        # 'use_reid': True,
        # 'fuse_score': True,
        # 'model': 'runs/triplet-yolo/train-007/best.pt'    
        'use_reid': False,  
    },
    # 提升 new_track_thresh（新軌跡門檻）會讓低分數 detection 無法及時產生新 ID，
    # 牽連到多輪配對與後處理，增加計算量，因此讓 BoTSORT 推理變慢。

    # 若要提升 unconfirmed 框被 track 成功匹配的數量，可考慮以下調整方式：
    # 1. 調低 'unconfirmed_match_thresh'，允許更寬鬆/更大距離範圍的配對（matching.linear_assignment閥值變寬）。
    #    預設值通常為 0.7 or 0.85，調整至更高（如 0.90, 0.95）會放鬆 unmatched 框的配對條件。
    # 2. 視任務需要，也可酌微調 'second_match_thresh'（但 primarily unconfirmed_track 用於 unconfirmed_match_thresh）。

    # 注意：閥值放太寬可能會增加誤配（false positive）的風險，建議逐步調整並觀察配對效果。
    # 'car': {
    #     'match_thresh': 0.85,
    #     'appearance_thresh': 0.8,
    #     'proximity_thresh': 0.5,        # 獨立放寬 IoU（原全域值通常 0.5）
    #     'center_dist_weight': 0.25,    # 中心點距離佔 25%，IoU 佔 75%
    #     'norm_factor': 150.0,          # 依你的畫面解析度與機車典型移動幅度調整
    #     'new_track_thresh': 0.15,       # 提高產生新ID困難度
    #     'second_match_thresh': 0.7,
    #     'unconfirmed_match_thresh': 0.95,
    #     'use_reid': True,
    #     'fuse_score': True,
    #     'model': 'runs/triplet-yolo/train-007/best.pt'    
    # },  
    # 
    # 一、近距離 → 中距離特別容易換 ID    
    'car': {
        'match_thresh': 0.9,
        'appearance_thresh': 0.5,
        'proximity_thresh': 0.3,        # 獨立放寬 IoU（原全域值通常 0.5）
        'center_dist_weight': 0.5,    # 中心點距離佔 25%，IoU 佔 75%
        'norm_factor': 80.0,          # 依你的畫面解析度與機車典型移動幅度調整
        'new_track_thresh': 0.3,       # 提高產生新ID困難度
        'second_match_thresh': 0.7,
        'unconfirmed_match_thresh': 0.95,
        'use_reid': True,
        'fuse_score': True,
        'model': 'runs/triplet-yolo/train-007/best.pt'    
    }, 
    # 'scooter': {
    #     'match_thresh': 0.85,            # 放寬匹配 (接受 IoU >= 0.1)
    #     'appearance_thresh': 0.80,       # 增加外觀權重 (嚴格增加 embedding 截斷)
    #     'proximity_thresh': 0.15,        # 獨立放寬 IoU（原全域值通常 0.5）
    #     'center_dist_weight': 0.25,    # 中心點距離佔 25%，IoU 佔 75%
    #     'norm_factor': 150.0,          # 依你的畫面解析度與機車典型移動幅度調整

    #     'new_track_thresh': 0.15,        # NOTE: 必須 >= track_high_thresh (0.15) 才會生效
    #     'second_match_thresh': 0.75,     # 放寬 second association（原 0.6），救回低信心偵測
    #     'unconfirmed_match_thresh': 0.95,# 放寬 unconfirmed matching（原 0.75），減少新 track 被丟棄
    #     'track_buffer': 180,             # 適度緩衝（原 60），給遮擋後重現更多時間
    #     'use_reid': True,
    #     'fuse_score': True,
    #     'model': 'runs/triplet-yolo/train-007/best.pt'
    # },
    # 'scooter': {
    #     'match_thresh': 0.8,            # 放寬匹配 (接受 IoU >= 0.1)
    #     'appearance_thresh': 0.80,       # 增加外觀權重 (嚴格增加 embedding 截斷)
    #     'proximity_thresh': 0.1,        # 獨立放寬 IoU（原全域值通常 0.5）
    #     'center_dist_weight': 0.25,    # 中心點距離佔 25%，IoU 佔 75%
    #     'norm_factor': 150.0,          # 依你的畫面解析度與機車典型移動幅度調整

    #     'new_track_thresh': 0.55,        # NOTE: 必須 >= track_high_thresh (0.15) 才會生效
    #     'second_match_thresh': 0.75,     # 放寬 second association（原 0.6），救回低信心偵測
    #     'unconfirmed_match_thresh': 0.95,# 放寬 unconfirmed matching（原 0.75），減少新 track 被丟棄
    #     'track_buffer': 360,             # 適度緩衝（原 60），給遮擋後重現更多時間
    #     'use_reid': True,
    #     'fuse_score': True,
    #     'model': 'runs/triplet-yolo/train-007/best.pt'
    # },
    'scooter': {
        'match_thresh': 0.7,            # Global Golden
        'new_track_thresh': 0.25,        # Global Golden
        'second_match_thresh': 0.5,     # Global Golden
        'unconfirmed_match_thresh': 0.75, # Global Golden
        'proximity_thresh': 0.1,        # 獨立放寬 IoU（暫定預設0.1）

        'appearance_thresh': 0.8,       # 增加外觀權重 (嚴格增加 embedding 截斷)
        'center_dist_weight': 0.25,    # 中心點距離佔 25%，IoU 佔 75%
        'norm_factor': 150.0,          # 依你的畫面解析度與機車典型移動幅度調整        
        
        'track_buffer': 360,             # 適度緩衝（原 60），給遮擋後重現更多時間
        'use_reid': True,
        'fuse_score': True,
        'model': 'runs/triplet-yolo/train-007/best.pt'
    },
    'small_truck': {
        'match_thresh': 0.85,
        'appearance_thresh': 0.40,
        'proximity_thresh': 0.5,        # 獨立放寬 IoU（原全域值通常 0.5）
        'center_dist_weight': 0.25,    # 中心點距離佔 25%，IoU 佔 75%
        'norm_factor': 150.0,          # 依你的畫面解析度與機車典型移動幅度調整
        'new_track_thresh': 0.15,
        'second_match_thresh': 0.5,
        'unconfirmed_match_thresh': 0.95,
        # 'use_reid': True,
        # 'fuse_score': True,
        # 'model': 'runs/triplet-yolo/train-007/best.pt'    
        'use_reid': False, 
    },
    'bus': {
        'match_thresh': 0.85,
        'appearance_thresh': 0.45,
        'proximity_thresh': 0.5,        # 獨立放寬 IoU（原全域值通常 0.5）
        'center_dist_weight': 0.25,    # 中心點距離佔 25%，IoU 佔 75%
        'norm_factor': 150.0,          # 依你的畫面解析度與機車典型移動幅度調整
        'new_track_thresh': 0.15,
        'second_match_thresh': 0.5,
        'unconfirmed_match_thresh': 0.95,
        # 'use_reid': True,
        # 'fuse_score': True,
        # 'model': 'runs/triplet-yolo/train-007/best.pt'    
        'use_reid': False, 
    },
    'truck': {
        'match_thresh': 0.85,
        'appearance_thresh': 0.45,
        'proximity_thresh': 0.5,        # 獨立放寬 IoU（原全域值通常 0.5）
        'center_dist_weight': 0.25,    # 中心點距離佔 25%，IoU 佔 75%
        'norm_factor': 150.0,          # 依你的畫面解析度與機車典型移動幅度調整
        'new_track_thresh': 0.15,
        'second_match_thresh': 0.5,
        'unconfirmed_match_thresh': 0.95,
        # 'use_reid': True,
        # 'fuse_score': True,
        # 'model': 'runs/triplet-yolo/train-007/best.pt'    
        'use_reid': False,   
    },
}

# Default parameters (for unknown classes)
DEFAULT_TRACKING_PARAMS = {
    'match_thresh': 0.80,
    'appearance_thresh': 0.35,
    'new_track_thresh': 0.15,
    'second_match_thresh': 0.5,
    'unconfirmed_match_thresh': 0.7,
    'use_reid': True,
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
