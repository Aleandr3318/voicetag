"""Tests for voicetag.overlap — overlap detection and segment merging."""
from __future__ import annotations

import pytest

from voicetag.overlap import detect_overlaps, merge_segments


# ---------------------------------------------------------------------------
# detect_overlaps
# ---------------------------------------------------------------------------


class TestDetectOverlaps:
    def test_finds_overlapping_regions(self):
        segments = [
            {"speaker": "A", "start": 0.0, "end": 5.0},
            {"speaker": "B", "start": 3.0, "end": 8.0},
        ]
        overlaps = detect_overlaps(segments, threshold=0.5)
        assert len(overlaps) == 1
        assert overlaps[0]["start"] == 3.0
        assert overlaps[0]["end"] == 5.0
        assert sorted(overlaps[0]["speakers"]) == ["A", "B"]

    def test_no_overlaps_returns_empty(self):
        segments = [
            {"speaker": "A", "start": 0.0, "end": 3.0},
            {"speaker": "B", "start": 5.0, "end": 8.0},
        ]
        overlaps = detect_overlaps(segments, threshold=0.5)
        assert overlaps == []

    def test_adjacent_non_overlapping(self):
        """Segments that touch but do not overlap should not be detected."""
        segments = [
            {"speaker": "A", "start": 0.0, "end": 3.0},
            {"speaker": "B", "start": 3.0, "end": 6.0},
        ]
        overlaps = detect_overlaps(segments, threshold=0.0)
        assert overlaps == []

    def test_same_speaker_not_overlap(self):
        """Segments from the same speaker should not generate overlaps."""
        segments = [
            {"speaker": "A", "start": 0.0, "end": 5.0},
            {"speaker": "A", "start": 3.0, "end": 8.0},
        ]
        overlaps = detect_overlaps(segments, threshold=0.5)
        assert overlaps == []

    def test_short_overlap_below_threshold(self):
        segments = [
            {"speaker": "A", "start": 0.0, "end": 3.0},
            {"speaker": "B", "start": 2.8, "end": 5.0},
        ]
        # Overlap is only 0.2s, threshold is 0.5
        overlaps = detect_overlaps(segments, threshold=0.5)
        assert overlaps == []

    def test_single_segment_returns_empty(self):
        segments = [{"speaker": "A", "start": 0.0, "end": 3.0}]
        assert detect_overlaps(segments) == []

    def test_empty_segments(self):
        assert detect_overlaps([]) == []

    def test_three_way_overlap(self):
        segments = [
            {"speaker": "A", "start": 0.0, "end": 5.0},
            {"speaker": "B", "start": 2.0, "end": 7.0},
            {"speaker": "C", "start": 3.0, "end": 6.0},
        ]
        overlaps = detect_overlaps(segments, threshold=0.5)
        # Should have at least A-B and A-C and B-C pairs
        assert len(overlaps) >= 2


# ---------------------------------------------------------------------------
# merge_segments
# ---------------------------------------------------------------------------


class TestMergeSegments:
    def test_sorted_by_start_time(self):
        segments = [
            {"speaker": "B", "start": 5.0, "end": 8.0, "confidence": 0.9},
            {"speaker": "A", "start": 0.0, "end": 3.0, "confidence": 0.8},
        ]
        overlaps = [
            {"speakers": ["A", "B"], "start": 2.5, "end": 3.5},
        ]
        merged = merge_segments(segments, overlaps, min_duration=0.5)
        starts = [s["start"] for s in merged]
        assert starts == sorted(starts)

    def test_filters_short_segments(self):
        segments = [
            {"speaker": "A", "start": 0.0, "end": 0.3, "confidence": 0.5},  # too short
            {"speaker": "B", "start": 1.0, "end": 4.0, "confidence": 0.9},
        ]
        merged = merge_segments(segments, [], min_duration=0.5)
        assert len(merged) == 1
        assert merged[0]["speaker"] == "B"

    def test_overlap_segments_marked(self):
        segments = [
            {"speaker": "A", "start": 0.0, "end": 3.0, "confidence": 0.8},
        ]
        overlaps = [
            {"speakers": ["A", "B"], "start": 2.0, "end": 3.0},
        ]
        merged = merge_segments(segments, overlaps, min_duration=0.5)
        overlap_entries = [s for s in merged if s.get("speaker") == "OVERLAP"]
        assert len(overlap_entries) == 1
        assert overlap_entries[0]["speakers"] == ["A", "B"]

    def test_empty_inputs(self):
        assert merge_segments([], [], min_duration=0.5) == []

    def test_filters_short_overlap_segments(self):
        segments = []
        overlaps = [
            {"speakers": ["A", "B"], "start": 0.0, "end": 0.1},  # too short
        ]
        merged = merge_segments(segments, overlaps, min_duration=0.5)
        assert merged == []
