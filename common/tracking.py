"""Thin wrapper around ultralytics ByteTrack that normalizes track IDs across ultralytics versions.

Both football_analytics.py and cricket_analytics.py call model.track() inline.
This module provides the shared ByteTrack helper so version-specific quirks (e.g.
result.boxes.id being None on the first frame) are handled in one place.

In practice, callers should prefer common.detection.run_track() which composes this
logic together with the YOLO predict call.  This module is kept separate per
TECHSPEC.md Section 4.2 to allow the tracker state to be tested independently of
the detector.

NOTE (TECHSPEC.md 4.2): ultralytics ByteTrack maintains its own internal state
across calls when persist=True is used.  This wrapper does NOT manage that state
— persist=True is the caller's responsibility.  This module's job is solely to
normalize the output format.
"""

from __future__ import annotations


def parse_track_results(result) -> list[dict]:
    """Normalize a single ultralytics YOLO result object (from model.track) into
    a list of track dicts.

    This is the bottleneck function called by both football and cricket pipelines.
    Keeping it here means any future ultralytics API change (e.g. boxes.id shape
    changing between versions) only needs to be fixed in one place.

    Parameters
    ----------
    result : ultralytics.engine.results.Results
        The first element of model.track(...) return value.

    Returns
    -------
    list of dicts with keys:
        "tid"  : int   — persistent track ID
        "cls"  : int   — COCO class ID
        "conf" : float — detection confidence
        "box"  : (x1, y1, x2, y2) in pixel space
    Empty list if no boxes or track IDs present this frame (common on frame 0).
    """
    tracks: list[dict] = []
    if result.boxes is None or result.boxes.id is None:
        return tracks
    for tid, cls, conf, box in zip(
        result.boxes.id.int().tolist(),
        result.boxes.cls.int().tolist(),
        result.boxes.conf.tolist(),
        result.boxes.xyxy.tolist(),
    ):
        tracks.append({
            "tid": int(tid),
            "cls": int(cls),
            "conf": float(conf),
            "box": tuple(float(v) for v in box),
        })
    return tracks
