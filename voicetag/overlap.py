"""Overlap detection and segment merging.

Pure-function module with no internal state. Identifies time regions where
two or more speakers talk simultaneously and merges segments into a unified
timeline.
"""

from __future__ import annotations

from loguru import logger


def detect_overlaps(
    segments: list[dict],
    threshold: float = 0.5,
) -> list[dict]:
    """Find time regions where two or more speakers overlap.

    Args:
        segments: List of dicts with ``"speaker"``, ``"start"``, and
            ``"end"`` keys (as returned by ``Diarizer.diarize``).
        threshold: Minimum overlap duration in seconds to include.

    Returns:
        List of overlap dicts, each containing ``"speakers"`` (list of
        speaker names), ``"start"``, and ``"end"``.
    """
    if len(segments) < 2:
        return []

    sorted_segs = sorted(segments, key=lambda s: s["start"])
    overlaps: list[dict] = []

    for i in range(len(sorted_segs)):
        for j in range(i + 1, len(sorted_segs)):
            seg_a = sorted_segs[i]
            seg_b = sorted_segs[j]

            # Early exit: sorted by start, so no further overlaps possible
            if seg_b["start"] >= seg_a["end"]:
                break

            if seg_a["speaker"] == seg_b["speaker"]:
                continue

            overlap_start = max(seg_a["start"], seg_b["start"])
            overlap_end = min(seg_a["end"], seg_b["end"])
            overlap_duration = overlap_end - overlap_start

            if overlap_duration >= threshold:
                overlaps.append(
                    {
                        "speakers": sorted({seg_a["speaker"], seg_b["speaker"]}),
                        "start": overlap_start,
                        "end": overlap_end,
                    }
                )

    merged = _merge_overlap_regions(overlaps)
    logger.debug("Detected {} overlap region(s)", len(merged))
    return merged


def _merge_overlap_regions(overlaps: list[dict]) -> list[dict]:
    """Merge overlapping overlap regions that share the same speakers.

    Args:
        overlaps: Raw overlap dicts.

    Returns:
        Merged and deduplicated overlap dicts.
    """
    if not overlaps:
        return []

    sorted_overlaps = sorted(overlaps, key=lambda o: o["start"])
    merged: list[dict] = [sorted_overlaps[0]]

    for current in sorted_overlaps[1:]:
        last = merged[-1]
        if current["start"] <= last["end"] and current["speakers"] == last["speakers"]:
            merged[-1] = {
                "speakers": last["speakers"],
                "start": last["start"],
                "end": max(last["end"], current["end"]),
            }
        else:
            merged.append(current)

    return merged


def merge_segments(
    segments: list[dict],
    overlaps: list[dict],
    min_duration: float = 0.5,
) -> list[dict]:
    """Merge speaker segments and overlaps into a single sorted timeline.

    Args:
        segments: List of speaker segment dicts with ``"speaker"``,
            ``"start"``, ``"end"``, and optionally ``"confidence"`` keys.
        overlaps: List of overlap dicts with ``"speakers"``, ``"start"``,
            and ``"end"`` keys.
        min_duration: Minimum segment duration in seconds. Segments shorter
            than this are discarded.

    Returns:
        Unified list of segment dicts sorted by start time, with short
        segments removed.
    """
    all_segments: list[dict] = []

    for seg in segments:
        duration = seg["end"] - seg["start"]
        if duration >= min_duration:
            all_segments.append(seg)

    for ovlp in overlaps:
        duration = ovlp["end"] - ovlp["start"]
        if duration >= min_duration:
            all_segments.append(
                {
                    "speaker": "OVERLAP",
                    "speakers": ovlp["speakers"],
                    "start": ovlp["start"],
                    "end": ovlp["end"],
                }
            )

    all_segments.sort(key=lambda s: s["start"])

    logger.debug(
        "Merged timeline: {} segments ({} removed for min_duration={:.2f}s)",
        len(all_segments),
        len(segments) + len(overlaps) - len(all_segments),
        min_duration,
    )
    return all_segments
