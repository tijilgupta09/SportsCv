"""Loads YOLOv8 once and exposes a simple detect() wrapper shared by every sport pipeline.

Both football_analytics.py and cricket_analytics.py previously called
model.predict() / model.track() inline with no shared abstraction. This module
provides thin wrappers that:
  - normalise the conf / classes / imgsz call signature
  - handle the result.boxes None-check in one place
  - make unit-testing easier (mock this module, not ultralytics)

IMPORTANT: callers are still responsible for loading the YOLO model themselves
(via YOLO("yolov8n.pt")) because model choice and weight-file path are
sport-specific concerns decided by the caller, not this shared module.
"""

from __future__ import annotations
from typing import Any


def run_detect(model: Any, frame, *, classes: list[int], conf: float,
               imgsz: int = 640, verbose: bool = False) -> list[dict]:
    """Run a YOLO predict() pass (no tracking) and return normalized detection dicts.

    Returns
    -------
    list of dicts, each with keys:
        "cls"  : int   — COCO class ID
        "conf" : float — detection confidence
        "box"  : (x1, y1, x2, y2) floats in pixel space
    """
    result = model.predict(
        frame, classes=classes, conf=conf, imgsz=imgsz, verbose=verbose
    )[0]
    detections: list[dict] = []
    if result.boxes is None:
        return detections
    for cls, c, box in zip(
        result.boxes.cls.int().tolist(),
        result.boxes.conf.tolist(),
        result.boxes.xyxy.tolist(),
    ):
        detections.append({
            "cls": int(cls),
            "conf": float(c),
            "box": tuple(float(v) for v in box),
        })
    return detections


def run_track(model: Any, frame, *, classes: list[int], conf: float,
              imgsz: int = 640, tracker: str = "bytetrack.yaml",
              persist: bool = True, verbose: bool = False) -> list[dict]:
    """Run a YOLO track() pass (ByteTrack) and return normalized track dicts.

    Returns
    -------
    list of dicts, each with keys:
        "tid"  : int   — track ID (from ByteTrack; persistent across frames)
        "cls"  : int   — COCO class ID
        "conf" : float — detection confidence
        "box"  : (x1, y1, x2, y2) floats in pixel space
    Returns an empty list if no boxes or track IDs are available this frame.
    """
    result = model.track(
        frame, persist=persist, tracker=tracker,
        conf=conf, classes=classes, imgsz=imgsz, verbose=verbose
    )[0]
    tracks: list[dict] = []
    if result.boxes is None or result.boxes.id is None:
        return tracks
    for tid, cls, c, box in zip(
        result.boxes.id.int().tolist(),
        result.boxes.cls.int().tolist(),
        result.boxes.conf.tolist(),
        result.boxes.xyxy.tolist(),
    ):
        tracks.append({
            "tid": int(tid),
            "cls": int(cls),
            "conf": float(c),
            "box": tuple(float(v) for v in box),
        })
    return tracks
