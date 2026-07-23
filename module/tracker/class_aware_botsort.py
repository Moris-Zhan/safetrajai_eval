"""
Class-aware BoTSORT tracker.
Extends Ultralytics BOTSORT to implement class-specific matching thresholds.
"""
import numpy as np

import torch
from ultralytics.trackers.bot_sort import BOTSORT, BOTrack
from ultralytics.trackers.byte_tracker import STrack
from ultralytics.trackers.basetrack import TrackState
from module.reid.encoders import ReID, SOTA_ReID, Custom_ReID
from module.tracker import matching
from ultralytics.trackers.utils.gmc import GMC
from torchinfo import summary

from .config import get_params_for_class, CLASS_TRACKING_PARAMS, CLASS_NAMES


class ClassAwareBOTSORT(BOTSORT):
    """
    Class-aware BoTSORT tracker.

    Core modification: Uses different matching thresholds for different object classes
    during the association stage.
    """

    def __init__(self, args, frame_rate=30):
        """Store frame_rate for per-class track_buffer scaling."""
        super().__init__(args, frame_rate)
        self.frame_rate = frame_rate
        #################################################################
        # ── 依實際 fps 重新縮放 Kalman Filter 的速度分量 ──
        # BOTrack 的 KF 狀態向量：[x, y, w, h, vx, vy, vw, vh]
        # 轉移矩陣中速度項的時間步長預設對應 30fps (dt=1)
        # 若實際為 5fps，每幀代表更長時間，速度貢獻應放大
        dt_scale = 30.0 / frame_rate  # 5fps → dt_scale = 6.0

        kf = self.kalman_filter
        # motion_mat 是 8x8 矩陣，(0,4),(1,5),(2,6),(3,7) 是速度項係數
        # 預設值為 1.0，乘以 dt_scale 後代表「每幀移動更多像素」
        for i in range(4):
            kf._motion_mat[i, i + 4] = dt_scale

        print(f"[ClassAwareBOTSORT] Kalman dt_scale = {dt_scale:.2f} "
            f"(frame_rate={frame_rate}fps)")
        #################################################################

        self.CLASS_KEYS = list(CLASS_TRACKING_PARAMS.keys())
        self.gmc = GMC(method=args.gmc_method)

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        # 建立一個字典 cls_encoder，為每個類別設置對應的 ReID 編碼器 (若有設定使用ReID模型)
        self.cls_encoder = dict()

        print(f"[Tracker] frame_rate received = {self.frame_rate}")
        for cls_id in range(len(CLASS_TRACKING_PARAMS)):
            cls_buffer_sec = get_params_for_class(cls_id).get('track_buffer')
            print(f"[Tracker] Loading track_buffer for class {CLASS_NAMES[cls_id]}: {cls_buffer_sec}")
            # 針對每個類別，查看其config是否啟用 ReID (use_reid)
            # 如果啟用則為該類建立 ReID 模型物件，否則設為None
            if get_params_for_class(cls_id).get('use_reid'):
                model_path = get_params_for_class(cls_id).get('model')
                print(f"\033[1;33m[ClassAwareBOTSORT] Loading ReID model for class {CLASS_NAMES[cls_id]}: \033[1;32m{model_path}\033[0m")                
                track_buffer = get_params_for_class(cls_id).get('track_buffer')
                print(f"[Tracker] Loading track_buffer for class {CLASS_NAMES[cls_id]}: {track_buffer}")

                if "reid_model_weights" in model_path:
                    self.cls_encoder[cls_id] = SOTA_ReID(model_path)
                    reid_model = self.cls_encoder[cls_id].model
                    dummy_input = torch.randn(1, 3, 256, 128)
                    if "vit_transreid" in model_path:
                        # input_data = (dummy_input, torch.tensor([-1]), torch.tensor([-1]))
                        input_data = (dummy_input, [0], [0])
                    else:
                        input_data = (dummy_input,)
                    # summary(
                    #     reid_model,
                    #     input_data=input_data,
                    #     device="cpu",
                    #     col_names=["num_params", "trainable", "mult_adds"],
                    #     depth=0,
                    #     row_settings=["var_names"],
                    # )
                elif "class_reid_weights" in model_path:
                    self.cls_encoder[cls_id] = Custom_ReID(model_path)
                    reid_model = self.cls_encoder[cls_id].model
                    # dummy_input = torch.randn(1, 3, 256, 128)                    
                    # input_data = (dummy_input, [])
                    # summary(
                    #     reid_model,
                    #     input_data=input_data,
                    #     device="cpu",
                    #     col_names=["num_params", "trainable", "mult_adds"],
                    #     depth=0,
                    #     row_settings=["var_names"],
                    # )
                else:
                    self.cls_encoder[cls_id] = ReID(model_path)
                #     reid_model = self.cls_encoder[cls_id].model
                #     nn_module = getattr(reid_model, "model", reid_model)
                #     summary(
                #         nn_module,
                #         input_size=(1, 3, 128, 64),
                #         device="cpu",
                #         col_names=["num_params", "trainable", "mult_adds"],
                #         depth=0,
                #         row_settings=["var_names"],
                #     )
           
            else:
                self.cls_encoder[cls_id] = None
        print("")

    def init_track(self, dets, scores, cls, img=None):
        """Initialize object tracks using detection bounding boxes, scores, class labels, and optional ReID features."""
        if len(dets) == 0:
            return []

        BOTrack_list = []
        for idx, cls_id in enumerate(cls):
            xyxy = dets[idx]
            s = scores[idx]
            c = int(cls_id)

            use_reid_cls = get_params_for_class(cls_id).get('use_reid')
            use_reid_model = get_params_for_class(cls_id).get('model')

            if use_reid_cls and use_reid_model is not None:
                encoder_fn = self.cls_encoder[c]
                if encoder_fn is not None:
                    # if "yolo" not in use_reid_model and "class_reid_weights" not in use_reid_model:
                    #     print()
                    f = encoder_fn(img, xyxy[None, ...])[0]
                else:
                    f = None
                BOTrack_list.append(BOTrack(xyxy, s, c, f))
            else:
                BOTrack_list.append(BOTrack(xyxy, s, c))
        return BOTrack_list


    def update(self, results, img=None):
        """
        Override update method to implement class-aware matching.

        Main modifications:
        1. Group detections and tracks by class
        2. Use class-specific match_thresh for each class
        3. Merge all class matching results

        Args:
            results: Detection results from YOLO
            img: Input image (optional, for ReID features)

        Returns:
            np.ndarray: Array of tracked object results
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        # Add index
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]

        # Initialize object tracks from detection bounding boxes, confidence scores, and class labels.
        # This includes optional ReID feature extraction per class if enabled.
        detections = self.init_track(dets, scores_keep, cls_keep, img)

        # Separate confirmed and unconfirmed tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Combine tracked and lost tracks for matching
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)

        # 檢查 Track class 是否與 Det class 不吻合（即可能出現 "class 抖動" 問題）
        track_classes_set = set(int(t.cls) for t in strack_pool)
        det_classes_set = set(int(d.cls) for d in detections)

        diff_tracks = []
        for t in strack_pool:
            if int(t.cls) not in det_classes_set:
                diff_tracks.append((t.track_id, t.cls))
        diff_dets = []
        for d in detections:
            if int(d.cls) not in track_classes_set:
                diff_dets.append(d.cls)

        # Predict current locations with Kalman filter
        self.multi_predict(strack_pool)

        # GMC (Global Motion Compensation)
        # 補償Kalman filter框的動量，使物件可以貼合真正的位置
        if hasattr(self, "gmc") and img is not None:
            try:
                warp = self.gmc.apply(img, dets)
            except Exception:
                warp = np.eye(2, 3)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        # === CLASS-AWARE MATCHING ===
        # Perform matching separately for each class with class-specific thresholds
        matched_track_indices = set()
        matched_det_indices = set()

        # Get unique classes from both detections and tracks
        det_classes = set(int(d.cls) for d in detections) if detections else set()
        track_classes = set(int(t.cls) for t in strack_pool) if strack_pool else set()
        all_classes = det_classes | track_classes

        for cls_id in all_classes:
            # Get class-specific parameters
            params = get_params_for_class(cls_id)
            cls_match_thresh = params['match_thresh']

            # Filter detections and tracks for this class
            cls_det_indices = [i for i, d in enumerate(detections) if int(d.cls) == cls_id]
            cls_track_indices = [i for i, t in enumerate(strack_pool) if int(t.cls) == cls_id]

            if not cls_det_indices or not cls_track_indices:
                continue

            cls_detections = [detections[i] for i in cls_det_indices]
            cls_tracks = [strack_pool[i] for i in cls_track_indices]

            # Calculate distance matrix (class-aware ReID / appearance gating)
            dists = self.get_dists_class_aware(cls_tracks, cls_detections, cls_id)

            # Use class-specific threshold for matching
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=cls_match_thresh)

            for itracked, idet in matches:
                track = cls_tracks[itracked]
                det = cls_detections[idet]

                original_track_idx = cls_track_indices[itracked]
                original_det_idx = cls_det_indices[idet]

                matched_track_indices.add(original_track_idx)
                matched_det_indices.add(original_det_idx)

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

        # Get unmatched tracks and detections (global indices)
        u_track_global = [i for i in range(len(strack_pool)) if i not in matched_track_indices]
        u_detection_global = [i for i in range(len(detections)) if i not in matched_det_indices]

        # === SECOND ASSOCIATION with low confidence detections ===
        detections_second = self.init_track(dets_second, scores_second, cls_second, img)
        r_tracked_stracks = [strack_pool[i] for i in u_track_global if strack_pool[i].state == TrackState.Tracked]

        # Class-aware second association (avoid cross-class matches)
        matched_rtrack_indices = set()
        if r_tracked_stracks and detections_second:
            det2_classes = set(int(d.cls) for d in detections_second)
            track2_classes = set(int(t.cls) for t in r_tracked_stracks)
            for cls_id in (det2_classes | track2_classes):
                cls_det_idx = [i for i, d in enumerate(detections_second) if int(d.cls) == cls_id]
                cls_track_idx = [i for i, t in enumerate(r_tracked_stracks) if int(t.cls) == cls_id]
                if not cls_det_idx or not cls_track_idx:
                    continue

                cls_dets = [detections_second[i] for i in cls_det_idx]
                cls_tracks = [r_tracked_stracks[i] for i in cls_track_idx]

                dists = matching.iou_distance(cls_tracks, cls_dets)
                params = get_params_for_class(cls_id)
                cls_second_thresh = params.get('second_match_thresh', 0.5)

                matches, _u_t, _u_d = matching.linear_assignment(dists, thresh=cls_second_thresh)

                for itracked, idet in matches:
                    track = cls_tracks[itracked]
                    det = cls_dets[idet]
                    matched_rtrack_indices.add(cls_track_idx[itracked])

                    if track.state == TrackState.Tracked:
                        track.update(det, self.frame_id)
                        activated_stracks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
                        refind_stracks.append(track)

        # Mark remaining unmatched tracks as lost
        u_track_second = [i for i in range(len(r_tracked_stracks)) if i not in matched_rtrack_indices]
        for it in u_track_second:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # === UNCONFIRMED TRACKS ===
        # Deal with unconfirmed tracks (usually tracks with only one beginning frame)
        detections_remain = [detections[i] for i in u_detection_global]

        matched_unconfirmed = set()
        matched_det_remain = set()

        if unconfirmed and detections_remain:
            detr_classes = set(int(d.cls) for d in detections_remain)
            unconf_classes = set(int(t.cls) for t in unconfirmed)
            for cls_id in (detr_classes | unconf_classes):
                cls_det_idx = [i for i, d in enumerate(detections_remain) if int(d.cls) == cls_id]
                cls_unconf_idx = [i for i, t in enumerate(unconfirmed) if int(t.cls) == cls_id]
                if not cls_det_idx or not cls_unconf_idx:
                    continue

                cls_dets = [detections_remain[i] for i in cls_det_idx]
                cls_unconf = [unconfirmed[i] for i in cls_unconf_idx]

                dists = self.get_dists_class_aware(cls_unconf, cls_dets, cls_id)
                params = get_params_for_class(cls_id)
                cls_unconf_thresh = params.get('unconfirmed_match_thresh', 0.7)

                matches, _u_t, _u_d = matching.linear_assignment(dists, thresh=cls_unconf_thresh)

                for itracked, idet in matches:
                    t = cls_unconf[itracked]
                    d = cls_dets[idet]
                    t.update(d, self.frame_id)
                    activated_stracks.append(t)
                    matched_unconfirmed.add(cls_unconf_idx[itracked])
                    matched_det_remain.add(cls_det_idx[idet])

        # Unmatched unconfirmed tracks -> removed
        u_unconfirmed = [i for i in range(len(unconfirmed)) if i not in matched_unconfirmed]
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Detections remaining after unconfirmed association
        u_detection = [i for i in range(len(detections_remain)) if i not in matched_det_remain]

        # === INITIALIZE NEW TRACKS ===
        # Create new tracks for unmatched detections with high enough confidence
        for inew in u_detection:
            track = detections_remain[inew]
            # Use class-specific new_track_thresh
            params = get_params_for_class(int(track.cls))
            new_track_thresh = params['new_track_thresh']

            if track.score < new_track_thresh:
                continue

            # Initialize a new track for unmatched detection with Kalman filter and assign track ID
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # === UPDATE STATE ===
        # Class-aware track buffer (max_time_lost) support
        # for track in self.lost_stracks:
        #     params = get_params_for_class(int(track.cls))
        #     cls_buffer = params.get('track_buffer', None)
        #     cls_max_time_lost = self.max_time_lost
        #     if cls_buffer is not None:
        #         cls_max_time_lost = int(self.frame_rate / 30.0 * cls_buffer)
        #     if self.frame_id - track.end_frame > cls_max_time_lost:
        #         track.mark_removed()
        #         removed_stracks.append(track)

        # 新版：cls_buffer 單位是秒，直接乘以 frame_rate 換算成幀數
        for track in self.lost_stracks:
            params = get_params_for_class(int(track.cls))
            cls_buffer_sec = params.get('track_buffer', None)
            
            if cls_buffer_sec is not None:
                # 秒數 × fps = 該 fps 下應等待的幀數
                cls_max_time_lost = int(cls_buffer_sec * self.frame_rate)
            else:
                cls_max_time_lost = self.max_time_lost  # fallback 用全域預設值

            if self.frame_id - track.end_frame > cls_max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # 第一步：移除 self.tracked_stracks 中非 Tracked 狀態的 track
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]

        # 第二步：將剛剛 activated 的 tracks 合併進 self.tracked_stracks
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)

        # 第三步：把剛剛重新找回來的 tracks（refind_stracks）合併進 self.tracked_stracks
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)

        # 第四步：從 lost_stracks 移除已經在 tracked_stracks 中的 track
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)

        # 第五步：將剛剛標記為 lost 的 tracks（lost_stracks）加入 self.lost_stracks
        self.lost_stracks.extend(lost_stracks)

        # 第六步：從 lost_stracks 中刪除已經被移除（removed）的 tracks
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)

        # 第七步：刪除 tracked/lost 之間的重複 track（僅同 class 可視為 duplicate）
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32), \
        np.asarray([x.result for x in self.tracked_stracks if not x.is_activated], dtype=np.float32), \
        np.asarray([x.result for x in removed_stracks], dtype=np.float32)

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Class-aware duplicate removal: only remove duplicates within the same class."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            # Skip cross-class pairs
            if int(stracksa[p].cls) != int(stracksb[q].cls):
                continue
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb

    def get_dists_class_aware(self, tracks, detections, cls_id):
        params = get_params_for_class(cls_id)
        appearance_thresh = params.get('appearance_thresh', self.appearance_thresh)
        proximity_thresh  = params.get('proximity_thresh', self.proximity_thresh)
        use_reid_cls      = params.get('use_reid', False)
        center_w          = params.get('center_dist_weight', 0.0)
        norm_factor       = params.get('norm_factor', 200.0)
        fuse_score        = params.get('fuse_score', False)

        # 以IoU計算追蹤框與偵測框之間的距離
        dists = matching.iou_distance(tracks, detections)
        # 找出所有超過 proximity threshold（代表兩者彼此距離太遠）的配對
        dists_mask = dists > (1 - proximity_thresh)

        # 如 center_w > 0，啟用中心點距離的融合（線性組合IoU與中心點距離）
        if center_w > 0:
            center_dists = self._center_distance(tracks, detections, norm_factor)
            dists = (1 - center_w) * dists + center_w * center_dists

        # 如啟用分數融合，調整距離分數以反映 detection 的置信度
        if fuse_score:
            dists = matching.fuse_score(dists, detections)

        # 每一類別可獨立決定是否使用 ReID/外觀特徵進行比對
        if use_reid_cls:
            encoder_fn = self.cls_encoder.get(cls_id)
            if encoder_fn is not None:
                # 若啟用 ReID，計算 tracks（smooth_feat）與 detections（curr_feat）的特徵距離成本矩陣
                # 該距離矩陣是 shape=(N_tracks, N_detections)，每一項表示一個 track 和一個 detection 的 embedding 特徵距離（通常為 cosine 或 euclidean）。
                # 用於判斷追蹤框和偵測框的外觀相似度
                emb_dists = matching.embedding_distance(tracks, detections) / 2.0
                emb_dists[emb_dists > (1 - appearance_thresh)] = 1.0  # 超過 appearance 門檻視為無法匹配
                emb_dists[dists_mask] = 1.0  # 若已超過 proximity 門檻，也視為無法匹配
                # 距離取IoU/center和ReID/appearance最小值（只要有一方無法匹配即不配對）
                dists = np.minimum(dists, emb_dists)

        # 回傳融合後的距離矩陣
        return dists

    def _center_distance(self, tracks, detections, norm_factor=200.0):
        """
        正規化中心點歐式距離，回傳 [0, 1] 的矩陣。
        norm_factor 為像素單位的正規化分母，依畫面解析度調整。
        """
        # 計算每個 track 的 bounding box 中心點座標 (x, y)
        t_centers = np.array([
            [
                t.tlwh[0] + (t.tlwh[2] / 2),  # x: 左上角 + 寬度一半
                t.tlwh[1] + (t.tlwh[3] / 2)   # y: 左上角 + 高度一半
            ]
            for t in tracks
        ])

        # 計算每個 detection 的 bounding box 中心點座標 (x, y)
        d_centers = np.array([
            [
                d.tlwh[0] + (d.tlwh[2] / 2),  # x: 左上角 + 寬度一半
                d.tlwh[1] + (d.tlwh[3] / 2)   # y: 左上角 + 高度一半
            ]
            for d in detections
        ])

        diff = t_centers[:, None, :] - d_centers[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        return np.clip(dist / norm_factor, 0.0, 1.0)
