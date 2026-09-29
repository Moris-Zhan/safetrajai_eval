import numpy as np
import scipy
from scipy.spatial.distance import cdist

from ultralytics.utils.metrics import batch_probiou, bbox_ioa

try:
    import lap  # for linear_assignment

    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements("lap>=0.5.12")  # https://github.com/gatagat/lap
    import lap


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True) -> tuple:
    """
    進行匈牙利算法線性匹配，可選用 lap.lapjv 或 scipy.optimize.linear_sum_assignment。
    流程說明:
        1. 根據 use_lap 參數，決定採用哪一套 assignment 套件 (lap 或 scipy)
        2. 根據給定的門檻值 thresh，僅保留距離小於等於該閾值的配對結果
        3. 輸出：所有匹配成功的索引組合，以及未被分配的兩組元素索引

    參數說明:
        cost_matrix (np.ndarray): 匹配成本矩陣，形狀為 (N, M)
        thresh (float): 匹配的最大允許門檻（成本大於此值者視為無效配對）
        use_lap (bool): 是否啟用 lap.lapjv 處理，如果為 False 則改用 scipy.optimize.linear_sum_assignment

    回傳:
        matched_indices (np.ndarray): 匹配成功的二元組索引 (K, 2)
        unmatched_a (np.ndarray or list): 第一組中未分配到的元素索引
        unmatched_b (np.ndarray or list): 第二組中未分配到的元素索引

    範例:
        >>> cost_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> thresh = 5.0
        >>> matched_indices, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh, use_lap=True)
    """
    # 若成本矩陣為空，直接回傳空的matches以及所有元素全部未分配
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:
        # 使用 lap.lapjv 進行 assignment，lap 支援成本閾值，extend_cost=True 允許虛擬節點
        # 參考: https://github.com/gatagat/lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        # x[i] 表示第i列（第一組）被分配到的第二組 index（若沒分配為-1）
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]  # 通過門檻，成功配對
        unmatched_a = np.where(x < 0)[0]  # 第一組未分配
        unmatched_b = np.where(y < 0)[0]  # 第二組未分配
    else:
        # 使用 scipy.optimize.linear_sum_assignment 進行 assignment，只能一次配出所有最小總成本
        # 參考: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)  # x 標記行(第一組), y 標記列(第二組)
        # 只保留在門檻範圍以內的配對
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            # 若一組有效配對都沒有，則所有元素皆未分配
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            # 利用集合減法取得未分配的元素
            unmatched_a = list(frozenset(np.arange(cost_matrix.shape[0])) - frozenset(matches[:, 0]))
            unmatched_b = list(frozenset(np.arange(cost_matrix.shape[1])) - frozenset(matches[:, 1]))

    return matches, unmatched_a, unmatched_b



def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """
    計算兩組追蹤物件/偵測框的 IoU（Intersection over Union）距離成本矩陣。
    步驟說明與詳細中文註解如下：

    流程說明:
        1. 檢查輸入的 `atracks` 和 `btracks` (可以是 STrack 物件列表或 numpy array)
        2. 若是 numpy array（直接給定的bbox陣列），直接賦值給 atlbrs/btlbrs
        3. 若是 STrack 物件，提取其 xywha（旋轉框）或 xyxy（水平框）
        4. 判斷盒型格式（5維為旋轉框、4維為水平框）
        5. 若為旋轉框，呼叫 batch_probiou 算法計算 IoU 矩陣
        6. 若為水平框，呼叫 bbox_ioa 算法並設 iou=True，計算IoU
        7. 回傳 1-IoU 作為 cost matrix，IoU 越大成本越低

    參數:
        atracks (List[STrack] | List[np.ndarray]): 第一組追蹤或框。
        btracks (List[STrack] | List[np.ndarray]): 第二組追蹤或框。

    回傳:
        (np.ndarray): IoU距離成本矩陣，形狀為 (len(atracks), len(btracks))

    範例:
        >>> atracks = [np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])]
        >>> btracks = [np.array([5, 5, 15, 15]), np.array([25, 25, 35, 35])]
        >>> cost_matrix = iou_distance(atracks, btracks)
    """
    # 判斷輸入資料類型，若 atracks 或 btracks 為 numpy array 則直接解包
    if (atracks and isinstance(atracks[0], np.ndarray)) or (btracks and isinstance(btracks[0], np.ndarray)):
        # 若傳入的是直接的bbox陣列，直接賦值
        atlbrs = atracks
        btlbrs = btracks
    else:
        # 傳入的是 STrack 類別，需依據有無角度資訊選擇 xywha 或 xyxy
        # xywha: (x, y, w, h, angle), xyxy: (x1, y1, x2, y2)
        atlbrs = [track.xywha if hasattr(track, 'angle') and track.angle is not None else track.xyxy for track in atracks]
        btlbrs = [track.xywha if hasattr(track, 'angle') and track.angle is not None else track.xyxy for track in btracks]

    # 初始化 ious陣列，大小:[atracks數, btracks數]
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    if len(atlbrs) and len(btlbrs):
        if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
            # 旋轉框格式（x, y, w, h, angle），呼叫batch_probiou計算 IoU
            ious = batch_probiou(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
            ).numpy()
        else:
            # 水平框格式（x1, y1, x2, y2），呼叫bbox_ioa，iou=True 得到IoU
            ious = bbox_ioa(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
                iou=True,
            )
    # 以 1-IoU 回傳作為成本，IoU 越大距離越小
    return 1 - ious  # 傳回distance cost matrix


def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
    """
    計算 tracks 與 detections 之間，基於特徵（embedding）的距離成本矩陣。

    詳細流程說明：
        1. 檢查輸入的 tracks, detections 為空（長度為 0）時直接返回空的成本矩陣。
        2. 對所有的 detections，收集它們的 embedding 特徵（curr_feat），形成一個 (M, feat_dim) 的二維數組。
        3. 對所有的 tracks，收集它們的 embedding 特徵（smooth_feat），形成一個 (N, feat_dim) 的二維數組。
        4. 使用指定的距離度量（如 cosine、euclidean 等）計算 tracks 特徵與 detections 特徵的距離矩陣。
           - 每個距離都經由 np.maximum(0.0, value) 處理，確保成本非負。
        5. 返回 shape=(N, M) 的成本矩陣，每一項值表示一個 track 與一個 detection embedding 之間的距離。

    參數:
        tracks (List[STrack]): 每個元素為含有 smooth_feat 特徵的追蹤物件列表。
        detections (List[BaseTrack]): 每個元素為含有 curr_feat 特徵的偵測物件列表。
        metric (str): 特徵距離計算時使用的度量方式，例如 'cosine' 或 'euclidean'。

    回傳:
        (np.ndarray): shape 為 (N, M) 的特徵距離成本矩陣（N: track 數量, M: detection 數量）

    範例:
        計算 cosine 距離矩陣:
        >>> tracks = [STrack(...), STrack(...)]  # 每個 STrack 需含 smooth_feat
        >>> detections = [BaseTrack(...), BaseTrack(...)]  # 每個 BaseTrack 需含 curr_feat
        >>> cost_matrix = embedding_distance(tracks, detections, metric="cosine")
    """
    # 初始化距離成本矩陣，大小為 [tracks 數, detections 數]，類型為 float32
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

    # 若無 track 或 detection，直接回傳空的成本矩陣
    if cost_matrix.size == 0:
        return cost_matrix

    # 取得所有 detection 物件的 curr_feat 特徵組成二維特徵矩陣
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)

    # 取得所有 track 物件的 smooth_feat 特徵組成二維特徵矩陣
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)

    # 計算特徵間距離，採用指定 metric；np.maximum(0.0, ...) 保證所有距離為非負
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # 標準化後特徵的距離

    # 回傳成本矩陣
    return cost_matrix



def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """
    結合成本矩陣與 detection 的信心分數，產生融合後的相似度/成本矩陣（fused cost matrix）

    詳細說明流程：
        1. 如果輸入的成本矩陣為空（沒有追蹤或偵測物件），直接回傳不做處理。
        2. 由於原始成本（cost_matrix）一般表示為 1-IoU（IoU 越大代表框越重疊，cost越小），先轉換回「IoU 相似度」：iou_sim = 1 - cost_matrix
        3. 收集所有 detection 的分數（score, 例如 detector 給每個 bounding box 的概率信心值），整理成一個 array
        4. 為了與 iou_sim 相乘，將 det_scores array shape 從 (M,) 擴展為 (N, M)，使每一行（每個 track 對所有 detection）都可以進行逐元素相乘
        5. 元素乘法：fuse_sim = iou_sim * det_scores      # 結合重疊程度與偵測信心（高信心分數的 detection 對映結果更為可信）
        6. 最後再轉回「融合後的 cost」輸出，即 1 - fuse_sim。這樣 fuse_sim 越大表示跟 track 越接近且信心越高，則 cost 越小，更傾向於分配。

    參數:
        cost_matrix (np.ndarray): shape (N, M), 配對的成本（通常是 1-IoU）
        detections (list): detection 物件組，每個包含 score 屬性

    回傳:
        np.ndarray: shape (N, M) 融合成本矩陣，值越小代表配對越可能

    範例:
        >>> cost_matrix = np.random.rand(5, 10)  # 5 tracks, 10 det
        >>> detections = [BaseTrack(score=np.random.rand()) for _ in range(10)]
        >>> fused_matrix = fuse_score(cost_matrix, detections)
    """
    # 1. 若成本矩陣為空（通常代表沒有 track 或 detection），直接回傳
    if cost_matrix.size == 0:
        return cost_matrix

    # 2. 計算 IoU 相似度矩陣（將 cost_matrix 轉換為相似度分數）
    iou_sim = 1 - cost_matrix  # IoU 越大越相似，越值得配對

    # 3. 擷取所有 detection 的 score，轉為 numpy array（形狀為 (M,)）
    det_scores = np.array([det.score for det in detections])

    # 4. 將 detection 分數複製 N 次（N=track數），reshape 為 (N, M)，讓每一行都對應到每一個 track
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)

    # 5. 融合相似度計算：IoU 相似度 和 detection 信心分數逐元素相乘
    fuse_sim = iou_sim * det_scores

    # 6. 輸出最終的融合後成本矩陣（值越小越容易分配給 track）
    return 1 - fuse_sim  # 輸出 fused cost 矩陣


def fuse_appearance(
    motion: np.ndarray,
    appearance: np.ndarray,
    proximity_mask: np.ndarray,
    appearance_thresh: float,
    is_lost: np.ndarray | None = None,
    center_px: np.ndarray | None = None,
    gate_radius: np.ndarray | None = None,
    veto_sim: float | None = None,
) -> np.ndarray:
    """Fuse motion and appearance cost with an optional center gate for lost tracks and an appearance veto.

    appearance is the embedding distance / 2 in [0, 1]. A pair may use its appearance cost when it is inside
    the IoU proximity (proximity_mask False) or, for lost tracks, when center_px <= gate_radius of that track.
    Appearance costs above 1 - appearance_thresh are ignored; the result is min(motion, appearance). With
    veto_sim set, pairs whose appearance distance exceeds 1 - veto_sim cost 1 (spec: reid-fusion).
    """
    allowed = ~proximity_mask
    if center_px is not None:
        allowed = allowed | (is_lost[:, None] & (center_px <= gate_radius[:, None]))
    gated = appearance.copy()
    gated[gated > (1 - appearance_thresh)] = 1.0
    gated[~allowed] = 1.0
    fused = np.minimum(motion, gated)
    if veto_sim is not None:
        fused[appearance > (1 - veto_sim)] = 1.0
    return fused
