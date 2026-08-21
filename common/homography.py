"""Generalized 4-point planar pixel<->world-coordinate mapper used by both Football and Cricket calibration."""
import math

import cv2
import numpy as np


class PlanarMapper:
    """4-point homography between pixel space and a real-world planar coordinate system
    (e.g. a pitch or a cricket wicket-to-wicket lane)."""

    def __init__(self, pixel_pts, world_pts):
        self.H = cv2.getPerspectiveTransform(np.float32(pixel_pts), np.float32(world_pts))
        self.Hinv = np.linalg.inv(self.H)

    def to_world(self, x, y):
        pt = np.array([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def to_pixel(self, wx, wy):
        pt = np.array([[[wx, wy]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.Hinv)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def reprojection_error(self, pixel_pts, world_pts):
        """Mean pixel-space distance between given check points and this mapper's
        own to_pixel(world_pt) — i.e. how well this homography explains points that
        were NOT used to fit it. Returns 0.0 if no check points are given (caller
        should treat that as "unverifiable", not "perfect")."""
        if not pixel_pts or not world_pts:
            return 0.0
        errs = []
        for (px, py), (wx, wy) in zip(pixel_pts, world_pts):
            bx, by = self.to_pixel(wx, wy)
            errs.append(math.hypot(bx - px, by - py))
        return float(np.mean(errs))

    @staticmethod
    def re_estimate(pixel_corners, world_corners, check_pixel_pts=None,
                     check_world_pts=None, max_reproj_error_px=40.0):
        """Build a candidate PlanarMapper from 4 freshly-detected corner pairs and score
        its confidence (0-1) via reprojection_error() on independent check points (e.g. a
        halfway-line intersection not used to fit the 4-corner homography itself — a 4-point
        homography fits its own corners exactly, so reprojection error against the corners
        themselves would trivially be ~0 and prove nothing).

        Does NOT mutate any existing mapper — this is a pure factory + scorer. The caller
        (football_analytics.py) decides whether to swap in the candidate based on confidence
        vs. config's min_calibration_confidence, per TECHSPEC.md Section 5.1 / APPFLOW.md
        Section 5. Without independent check points, confidence is capped at 0.5 (geometry
        alone was verified by the caller before calling this, but nothing here confirms the
        homography itself is correct) rather than assumed perfect.
        """
        candidate = PlanarMapper(pixel_corners, world_corners)
        if check_pixel_pts and check_world_pts:
            err = candidate.reprojection_error(check_pixel_pts, check_world_pts)
            confidence = max(0.0, 1.0 - err / max_reproj_error_px)
        else:
            confidence = 0.5
        return candidate, confidence

