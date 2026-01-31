#!/usr/bin/env python3
"""Unit tests for utility functions."""
from cone_tracker.utils import bbox_distance, bbox_union, x_overlap_ratio


def test_bbox_union_empty():
    assert bbox_union([]) is None


def test_bbox_union_negative_coords():
    boxes = [(-5, -5, 10, 10), (0, 0, 5, 5)]
    assert bbox_union(boxes) == (-5, -5, 10, 10)


def test_x_overlap_no_overlap():
    a = (0, 0, 10, 10)
    b = (20, 0, 5, 5)
    assert x_overlap_ratio(a, b) == 0.0


def test_x_overlap_negative_coords():
    a = (-10, 0, 10, 10)
    b = (-5, 0, 10, 10)
    ratio = x_overlap_ratio(a, b)
    assert 0.0 < ratio <= 1.0


def test_bbox_distance_negative_coords():
    a = (-10, -10, 10, 10)
    b = (10, 10, 10, 10)
    dist = bbox_distance(a, b)
    assert dist > 0
