"""
Class-aware ByteTrack tracker.
Extends Ultralytics BYTETracker to implement class-specific matching thresholds.
"""
import numpy as np

from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils import matching


from .config import get_params_for_class


class ClassAwareBYTETracker(BYTETracker):
    """
    Class-aware ByteTrack tracker.

    Core modification: Uses different matching thresholds for different object classes
    during the association stage. Unlike BoTSORT, ByteTrack does not use ReID features.
    """

    def __init__(self, args, frame_rate=30):
        """Store frame_rate for per-class track_buffer scaling."""
        super().__init__(args, frame_rate)
        self.frame_rate = frame_rate

    def update(self, results, img=None, feats=None):
        """
        Override update method to implement class-aware matching.

        Main modifications:
        1. Group detections and tracks by class
        2. Use class-specific match_thresh for each class
        3. Merge all class matching results

        Args:
            results: Detection results from YOLO
            img: Input image (optional)
            feats: Pre-computed features (optional, not used in ByteTrack)

        Returns:
            np.ndarray: Array of tracked object results
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # scores = results.conf
        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        # Add index
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        # 原本
        inds_second = inds_low & inds_high

        # 改成只要低於 high_thresh 就全進來
        # inds_second = scores < self.args.track_high_thresh

        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]

        # NOTE: ByteTrack 使用 results 物件（含 cls 資訊）初始化 detections，
        # 與 BoTSORT 分別傳入 dets/scores/cls 的方式不同。
        # feats_keep = feats_second = img
        # if feats is not None and len(feats):
        #     feats_keep = feats[remain_inds]
        #     feats_second = feats[inds_second]

        # Initialize detections with high confidence
        detections = self.init_track(dets, scores_keep, cls_keep)

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

        # Predict current locations with Kalman filter
        self.multi_predict(strack_pool)

        # NOTE: BYTETracker does not use GMC (Global Motion Compensation).
        # GMC is a BoTSORT-specific feature.

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

            # Calculate distance matrix (class-aware, IoU + optional center distance + fuse_score)
            dists = self.get_dists_class_aware(cls_tracks, cls_detections, cls_id)

            # Use class-specific threshold for matching
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=cls_match_thresh)

            # Process matches
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
        detections_second = self.init_track(dets_second, scores_second, cls_second)
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
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # === UPDATE STATE ===
        # Class-aware track buffer (max_time_lost) support
        for track in self.lost_stracks:
            params = get_params_for_class(int(track.cls))
            cls_buffer = params.get('track_buffer', None)
            cls_max_time_lost = self.max_time_lost
            if cls_buffer is not None:
                cls_max_time_lost = int(self.frame_rate / 30.0 * cls_buffer)
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
        """
        Calculate distances with class-specific thresholds.
        NOTE: ByteTrack does not support ReID; only IoU, center distance, and fuse_score are used.

        Args:
            tracks: List of track objects
            detections: List of detection objects
            cls_id: Class ID for threshold lookup

        Returns:
            np.ndarray: Distance matrix
        """
        params = get_params_for_class(cls_id)
        center_w        = params.get('center_dist_weight', 0.0)
        norm_factor     = params.get('norm_factor', 200.0)
        fuse_score      = params.get('fuse_score', False)

        dists = matching.iou_distance(tracks, detections)

        # 中心點距離融合（可選）
        if center_w > 0:
            center_dists = self._center_distance(tracks, detections, norm_factor)
            dists = (1 - center_w) * dists + center_w * center_dists

        # 若啟用 fuse_score，將 detection 的分數資訊融合進距離矩陣中，提升配對精度
        if fuse_score:
            dists = matching.fuse_score(dists, detections)

        # NOTE: ByteTrack 不支援 ReID 外觀特徵比對，此為與 BoTSORT 的核心差異之一。
        # proximity_thresh 在此僅保留供 center_w 等擴充功能參考，不用於 embedding gating。
        return dists

    def _center_distance(self, tracks, detections, norm_factor=200.0):
        """
        正規化中心點歐式距離，回傳 [0, 1] 的矩陣。
        norm_factor 為像素單位的正規化分母，依畫面解析度調整。
        """
        t_centers = np.array([
            [
                t.tlwh[0] + (t.tlwh[2] / 2),
                t.tlwh[1] + (t.tlwh[3] / 2)
            ]
            for t in tracks
        ])

        d_centers = np.array([
            [
                d.tlwh[0] + (d.tlwh[2] / 2),
                d.tlwh[1] + (d.tlwh[3] / 2)
            ]
            for d in detections
        ])

        diff = t_centers[:, None, :] - d_centers[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        return np.clip(dist / norm_factor, 0.0, 1.0)
