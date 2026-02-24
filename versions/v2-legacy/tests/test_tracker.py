#!/usr/bin/env python3
"""Unit tests for tracking state machine."""
import copy
import time

from cone_tracker.config import DEFAULT_CONFIG
from cone_tracker.tracker import MultiConeTracker
from cone_tracker.utils import ConeState, VisibilityState


def _make_config():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["geometry"]["confirm_avg_score"] = 0.5
    cfg["tracking"]["min_frames_for_confirm"] = 2
    cfg["tracking"]["grace_frames"] = 1
    cfg["tracking"]["grace_seconds"] = 0.0
    cfg["tracking"]["lost_timeout"] = 0.1
    cfg["tracking"]["max_tracks"] = 1
    return cfg


def _det(bbox=(10, 10, 20, 40), score=0.9):
    return [(bbox, score, {})]


def test_track_starts_suspect():
    cfg = _make_config()
    tracker = MultiConeTracker(cfg)
    tracker.update(_det())
    assert len(tracker.tracks) == 1
    t = tracker.tracks[0]
    assert t.state == ConeState.SUSPECT
    assert t.visibility == VisibilityState.VISIBLE


def test_confirm_after_min_frames():
    cfg = _make_config()
    tracker = MultiConeTracker(cfg)
    tracker.update(_det())
    tracker.update(_det())
    assert tracker.tracks[0].state == ConeState.CONFIRMED


def test_transition_to_occluded_and_lost():
    cfg = _make_config()
    tracker = MultiConeTracker(cfg)
    tracker.update(_det())
    tracker.update(_det())
    assert tracker.tracks[0].state == ConeState.CONFIRMED

    tracker.update([])
    assert tracker.tracks[0].visibility == VisibilityState.OCCLUDED

    tracker.update([])
    assert len(tracker.tracks) == 0


def test_deletion_after_lost_timeout_for_suspects():
    cfg = _make_config()
    tracker = MultiConeTracker(cfg)
    tracker.update(_det())
    assert len(tracker.tracks) == 1
    tracker.tracks[0].last_seen = time.time() - 1.0
    tracker.update([])
    assert len(tracker.tracks) == 0


def test_unmatched_detections_respect_max_tracks():
    cfg = _make_config()
    tracker = MultiConeTracker(cfg)
    dets = [
        ((10, 10, 20, 40), 0.9, {}),
        ((100, 100, 20, 40), 0.9, {}),
    ]
    tracker.update(dets)
    assert len(tracker.tracks) == 1
