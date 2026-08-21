"""EasyOCR-backed helpers for reading jersey numbers and broadcast scoreboard graphics."""
import re

import cv2

_READER = None  # lazily-initialized singleton; EasyOCR model load is expensive (~seconds)


def _get_reader():
    global _READER
    if _READER is None:
        import easyocr  # deferred import: only paid for by callers that actually use OCR
        print("[OCR] Loading EasyOCR model (first call only)...", flush=True)
        _READER = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("[OCR] EasyOCR ready ✓", flush=True)
    return _READER


def read_jersey_number(patch):
    """Best-effort digit read on a player torso-crop patch (same crop already
    used for team-color sampling - see common/team_classifier.torso_patch).
    Returns a digit string (e.g. "7", "23") or None if nothing legible was
    found. Not rate-limited here - callers (football_analytics.py) are
    responsible for only calling this every ocr_interval_frames per track
    per RULES.md Section 9."""
    if patch is None or patch.size == 0:
        return None
    upscaled = cv2.resize(patch, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    results = _get_reader().readtext(upscaled, allowlist='0123456789', detail=1)
    if not results:
        return None
    _, text, conf = max(results, key=lambda r: r[2])
    digits = re.sub(r'\D', '', text)
    if not digits or conf < 0.35:
        return None
    return digits


def read_scoreboard(frame, roi_pixels):
    """Best-effort OCR read of a broadcast scoreboard graphic within a
    configured pixel ROI (x1,y1,x2,y2). Returns the raw recognized text
    (caller is responsible for parsing runs/wickets/overs and treating it
    as a best-effort, not-guaranteed-correct source - see SCHEMA.md 2.4's
    "source": "ocr" tagging convention). Returns None if nothing legible."""
    x1, y1, x2, y2 = roi_pixels
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    results = _get_reader().readtext(crop, detail=1)
    if not results:
        return None
    return " ".join(text for _, text, conf in results if conf >= 0.35) or None
