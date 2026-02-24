#!/usr/bin/env python3
"""Multi-object tracking for cone detection."""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
except Exception as exc:  # pragma: no cover - optional dependency
    linear_sum_assignment = None
    logging.getLogger(__name__).warning("Hungarian assignment disabled: %s", exc)

from .utils import ConeState, VisibilityState, bbox_distance, bbox_iou, log_event

logger = logging.getLogger(__name__)


# =========================
# MULTI TRACKING
# =========================
@dataclass
class Track:
    """Represents a tracked cone object.

    Units:
        - Position and size in px.
        - Velocity in px/frame.
    """
    track_id: int
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0

    state: ConeState = ConeState.SUSPECT
    visibility: VisibilityState = VisibilityState.VISIBLE
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_good_time: float = field(default_factory=time.time)

    score_hist: deque = field(default_factory=lambda: deque(maxlen=10))
    miss_counter: int = 0
    vx: float = 0.0
    vy: float = 0.0

    def bbox(self) -> Tuple[int, int, int, int]:
        """Get bounding box as (x, y, w, h).

        Units: px.
        Complexity: O(1).
        """
        return (int(self.cx - self.w / 2), int(self.cy - self.h / 2), int(self.w), int(self.h))

    @property
    def frames_seen(self) -> int:
        """Get the number of frames this track has been seen.

        Units: frames.
        Complexity: O(1).
        """
        return len(self.score_hist)

    def avg_score(self) -> float:
        """Get average score from history.

        Units: normalized 0..1.
        Complexity: O(n) over score history.
        """
        if not self.score_hist:
            return 0.0
        return float(np.mean(self.score_hist))

    def update(self, bbox: Tuple[int, int, int, int], score: float, alpha: float, score_window: int):
        """Update track with new detection.

        Units:
            - bbox in px
            - score normalized 0..1
        Complexity: O(1)
        """
        if self.score_hist.maxlen != score_window:
            self.score_hist = deque(self.score_hist, maxlen=score_window)

        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        if self.w <= 0:
            self.cx, self.cy, self.w, self.h = cx, cy, w, h
        else:
            prev_cx, prev_cy = self.cx, self.cy
            self.cx = alpha * cx + (1 - alpha) * self.cx
            self.cy = alpha * cy + (1 - alpha) * self.cy
            self.w = alpha * w + (1 - alpha) * self.w
            self.h = alpha * h + (1 - alpha) * self.h
            self.vx = self.cx - prev_cx
            self.vy = self.cy - prev_cy

        self.score_hist.append(float(score))
        now = time.time()
        self.last_seen = now
        self.last_good_time = now
        self.miss_counter = 0
        self.visibility = VisibilityState.VISIBLE

    def predict(self, inertia_decay: float = 0.9) -> None:
        """Predict next position for occluded track using inertial motion.

        Units: px; updates internal center position.
        Complexity: O(1).
        """
        self.cx += self.vx
        self.cy += self.vy
        self.vx *= inertia_decay
        self.vy *= inertia_decay


class MultiConeTracker:
    """Multi-object tracker for cone detection.

    Complexity: O(n*m) for association by default, or O(n^3) with Hungarian.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.cfg_root = config
        self.cfg = config["tracking"]
        self.geo = config["geometry"]
        self.tracks: List[Track] = []
        self.next_id = 0

    def _make_track(self, det_bbox: Tuple[int, int, int, int], det_score: float) -> Track:
        """Create a new track from detection.

        Units: bbox in px, score normalized 0..1.
        Complexity: O(1).
        """
        t = Track(track_id=self.next_id, score_hist=deque(maxlen=int(self.cfg["score_window"])))
        self.next_id += 1
        t.update(det_bbox, det_score, alpha=float(self.cfg["ema_alpha"]), score_window=int(self.cfg["score_window"]))
        return t

    def _association_cost(self, track_bbox: Tuple[int, int, int, int], det_bbox: Tuple[int, int, int, int]) -> float:
        """Compute association cost as normalized distance + IoU penalty.

        Units: distance normalized 0..1, IoU in 0..1.
        Complexity: O(1).
        """
        max_dist = float(self.cfg["association_max_distance"])
        dist = bbox_distance(track_bbox, det_bbox)
        if max_dist <= 0:
            dist_norm = 1.0
        else:
            dist_norm = min(dist / max_dist, 1.0)
        iou = bbox_iou(track_bbox, det_bbox)
        iou_weight = float(self.cfg.get("iou_penalty_weight", 0.35))
        return dist_norm + iou_weight * (1.0 - iou)

    def _associate_greedy(self, detections: List[Tuple[Tuple[int, int, int, int], float, dict]]) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        Associate tracks->detections by distance (greedy).
        Returns:
        - matches: {track_index: detection_index}
        - unmatched_tracks: [track_index]
        - unmatched_detections: [detection_index]
        """
        if not self.tracks:
            return {}, [], list(range(len(detections)))
        if not detections:
            return {}, list(range(len(self.tracks))), []

        if bool(self.cfg.get("use_hungarian", False)) and linear_sum_assignment is not None:
            return self._associate_hungarian(detections)
        if bool(self.cfg.get("use_hungarian", False)):
            logger.warning("Hungarian assignment requested but dependency unavailable, falling back to greedy")

        max_dist = float(self.cfg["association_max_distance"])
        pairs = []
        for ti, trk in enumerate(self.tracks):
            tb = trk.bbox()
            for di, (db, _s, _d) in enumerate(detections):
                dist = bbox_distance(tb, db)
                if dist <= max_dist:
                    pairs.append((dist, ti, di))

        pairs.sort(key=lambda x: x[0])  # smallest distance first

        matched_tracks = set()
        matched_dets = set()
        matches: Dict[int, int] = {}

        for dist, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matches[ti] = di
            matched_tracks.add(ti)
            matched_dets.add(di)

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_dets]
        return matches, unmatched_tracks, unmatched_detections

    def _associate_hungarian(self, detections: List[Tuple[Tuple[int, int, int, int], float, dict]]) -> Tuple[Dict[int, int], List[int], List[int]]:
        """Associate tracks->detections using Hungarian algorithm.

        Uses cost = normalized distance + IoU penalty. Filters out pairs beyond max distance.
        """
        if not self.tracks:
            return {}, [], list(range(len(detections)))
        if not detections:
            return {}, list(range(len(self.tracks))), []

        max_dist = float(self.cfg["association_max_distance"])
        large_cost = 1e6
        cost_matrix = np.full((len(self.tracks), len(detections)), large_cost, dtype=np.float32)
        for ti, trk in enumerate(self.tracks):
            tb = trk.bbox()
            for di, (db, _s, _d) in enumerate(detections):
                dist = bbox_distance(tb, db)
                if dist <= max_dist:
                    cost_matrix[ti, di] = self._association_cost(tb, db)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches: Dict[int, int] = {}
        matched_tracks = set()
        matched_dets = set()
        for ti, di in zip(row_ind.tolist(), col_ind.tolist()):
            if cost_matrix[ti, di] >= large_cost:
                continue
            matches[ti] = di
            matched_tracks.add(ti)
            matched_dets.add(di)

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_dets]
        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections: List[Tuple[Tuple[int, int, int, int], float, dict]]):
        """Update tracker with new detections.

        Units:
            - bbox in px
            - score normalized 0..1
        Complexity: O(n*m) association by default.
        """
        now = time.time()

        # 0) Predict positions for occluded tracks (inertial)
        inertia_decay = float(self.cfg.get("inertia_decay", 0.9))
        for t in self.tracks:
            if t.visibility == VisibilityState.OCCLUDED and t.state != ConeState.LOST:
                t.predict(inertia_decay=inertia_decay)

        # 1) Expire old tracks (suspects only, or LOST)
        alive = []
        for t in self.tracks:
            if t.state == ConeState.LOST:
                log_event(
                    logger,
                    self.cfg_root,
                    logging.INFO,
                    f"🗑️  Track {t.track_id} DELETADO: frames={len(t.score_hist)}, avg={t.avg_score():.2f}, idade={(now - t.created_at):.2f}s",
                    "track_deleted",
                    track_id=t.track_id,
                    frames=len(t.score_hist),
                    avg_score=float(t.avg_score()),
                    age_s=float(now - t.created_at),
                )
                continue
            if t.state != ConeState.CONFIRMED and now - t.last_seen > float(self.cfg["lost_timeout"]):
                log_event(
                    logger,
                    self.cfg_root,
                    logging.INFO,
                    f"🗑️  Track {t.track_id} DELETADO: frames={len(t.score_hist)}, avg={t.avg_score():.2f}, idade={(now - t.created_at):.2f}s",
                    "track_deleted",
                    track_id=t.track_id,
                    frames=len(t.score_hist),
                    avg_score=float(t.avg_score()),
                    age_s=float(now - t.created_at),
                )
                continue
            alive.append(t)
        self.tracks = alive

        # 2) Associate
        matches, unmatched_tracks, unmatched_dets = self._associate_greedy(detections)

        # 3) Update matched tracks
        for ti, di in matches.items():
            bbox, score, _data = detections[di]
            was_occluded = self.tracks[ti].visibility == VisibilityState.OCCLUDED
            self.tracks[ti].update(bbox, score, alpha=float(self.cfg["ema_alpha"]), score_window=int(self.cfg["score_window"]))
            if was_occluded:
                log_event(
                    logger,
                    self.cfg_root,
                    logging.INFO,
                    f"👀 Track {self.tracks[ti].track_id} OCCLUDED → VISIBLE",
                    "track_visible",
                    track_id=self.tracks[ti].track_id,
                )

        # 4) Update miss counter for unmatched tracks (for grace period)
        for ti in unmatched_tracks:
            t = self.tracks[ti]
            if t.state == ConeState.CONFIRMED:
                if t.visibility != VisibilityState.OCCLUDED:
                    t.visibility = VisibilityState.OCCLUDED
                t.miss_counter += 1
                grace_seconds = float(self.cfg.get("grace_seconds", 0.0))
                grace_frames = int(self.cfg.get("grace_frames", 12))
                expired = False
                if grace_seconds > 0.0:
                    expired = (now - t.last_good_time) > grace_seconds
                else:
                    expired = t.miss_counter > grace_frames

                if expired:
                    t.state = ConeState.LOST
                    log_event(
                        logger,
                        self.cfg_root,
                        logging.INFO,
                        f"❌ Track {t.track_id} CONFIRMED → LOST",
                        "track_lost",
                        track_id=t.track_id,
                        miss_frames=t.miss_counter,
                        grace_seconds=grace_seconds,
                        grace_frames=grace_frames,
                    )

        # 4b) Remove LOST tracks immediately after transition
        if any(t.state == ConeState.LOST for t in self.tracks):
            remaining = []
            for t in self.tracks:
                if t.state == ConeState.LOST:
                    log_event(
                        logger,
                        self.cfg_root,
                        logging.INFO,
                        f"🗑️  Track {t.track_id} DELETADO: frames={len(t.score_hist)}, avg={t.avg_score():.2f}, idade={(now - t.created_at):.2f}s",
                        "track_deleted",
                        track_id=t.track_id,
                        frames=len(t.score_hist),
                        avg_score=float(t.avg_score()),
                        age_s=float(now - t.created_at),
                    )
                    continue
                remaining.append(t)
            self.tracks = remaining

        # 5) Create tracks for unmatched detections
        for di in unmatched_dets:
            if len(self.tracks) >= int(self.cfg["max_tracks"]):
                break
            bbox, score, _data = detections[di]
            self.tracks.append(self._make_track(bbox, score))

        # 6) Decide state (CONFIRMED) by average
        for t in self.tracks:
            frames = len(t.score_hist)
            avg = t.avg_score()
            min_frames = int(self.cfg["min_frames_for_confirm"])
            threshold = float(self.geo["confirm_avg_score"])
            
            if frames >= min_frames and avg >= threshold:
                if t.state != ConeState.CONFIRMED:
                    # LOG when confirmed
                    log_event(
                        logger,
                        self.cfg_root,
                        logging.INFO,
                        f"✅ Track {t.track_id} CONFIRMADO! frames={frames}, avg={avg:.2f}",
                        "track_confirmed",
                        track_id=t.track_id,
                        frames=frames,
                        avg_score=float(avg),
                    )
                t.state = ConeState.CONFIRMED
            else:
                if t.state != ConeState.CONFIRMED:
                    t.state = ConeState.SUSPECT

    def confirmed_tracks(self) -> List[Track]:
        """Get list of confirmed tracks.

        Returns tracks with lifecycle state CONFIRMED (visibility may be VISIBLE or OCCLUDED).
        Complexity: O(n).
        """
        # Note: min_confirmed_age_frames is currently not enforced in frame count
        # Returning all CONFIRMED tracks regardless of age
        return [t for t in self.tracks if t.state == ConeState.CONFIRMED]
