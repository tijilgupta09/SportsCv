"""Pure-logic tests for common/homography.py (PlanarMapper) per TECHSPEC.md Section 8."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.homography import PlanarMapper

EPS = 1e-2

PIXEL_CORNERS = [(120, 90), (1160, 90), (1260, 690), (30, 690)]
WORLD_CORNERS = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)]


def test_round_trip_corners():
    mapper = PlanarMapper(PIXEL_CORNERS, WORLD_CORNERS)
    for px, wx in zip(PIXEL_CORNERS, WORLD_CORNERS):
        world = mapper.to_world(*px)
        assert abs(world[0] - wx[0]) < EPS
        assert abs(world[1] - wx[1]) < EPS
        pixel = mapper.to_pixel(*wx)
        assert abs(pixel[0] - px[0]) < EPS
        assert abs(pixel[1] - px[1]) < EPS


def test_round_trip_interior_point():
    mapper = PlanarMapper(PIXEL_CORNERS, WORLD_CORNERS)
    # Center of the pixel quad, roughly center of the world rectangle.
    interior_px = (
        sum(p[0] for p in PIXEL_CORNERS) / 4,
        sum(p[1] for p in PIXEL_CORNERS) / 4,
    )
    world = mapper.to_world(*interior_px)
    back = mapper.to_pixel(*world)
    assert abs(back[0] - interior_px[0]) < EPS
    assert abs(back[1] - interior_px[1]) < EPS



# ---------------------------------------------------------------------------
#  PlanarMapper.re_estimate() — accept/reject confidence tests
#  (TECHSPEC.md Section 8, Task 1.1/1.2 — previously manually verified only,
#  now automated per the bug report noting the stale comment gap).
# ---------------------------------------------------------------------------

# Halfway-line pixel/world check points NOT used to fit the 4 corners.
# These simulate the independent check points detect_pitch_corners() returns.
# Pixel coords computed via PlanarMapper.to_pixel(52.5, 34.0) on the test quad.
CHECK_PIXEL = [(642, 365)]          # pixel projection of world midpoint (52.5, 34.0)
CHECK_WORLD = [(52.5, 34.0)]        # near-centre of the pitch in world metres


def test_re_estimate_accepts_consistent_homography():
    """A freshly-estimated homography whose check-point reprojection error is
    small should be accepted (confidence >= 0.75, the threshold in football_config.yaml)."""
    candidate, conf = PlanarMapper.re_estimate(
        PIXEL_CORNERS, WORLD_CORNERS,
        check_pixel_pts=CHECK_PIXEL,
        check_world_pts=CHECK_WORLD,
        max_reproj_error_px=40.0,
    )
    assert conf >= 0.75, (
        f"Expected high confidence for a consistent homography, got {conf:.3f}"
    )
    # The returned object must actually be a PlanarMapper (not None / raw tuple)
    assert hasattr(candidate, "to_world") and hasattr(candidate, "to_pixel")


def test_re_estimate_rejects_inconsistent_homography():
    """A homography fitted to corners that are inconsistent with the check points
    (simulating a bad auto-detection) should score low confidence (< 0.75)."""
    # Deliberately broken corners — wildly different from the pixel quad above.
    bad_corners = [(0, 0), (10, 0), (10, 10), (0, 10)]
    candidate, conf = PlanarMapper.re_estimate(
        bad_corners, WORLD_CORNERS,
        check_pixel_pts=CHECK_PIXEL,
        check_world_pts=CHECK_WORLD,
        max_reproj_error_px=40.0,
    )
    assert conf < 0.75, (
        f"Expected low confidence for an inconsistent homography, got {conf:.3f}"
    )


def test_re_estimate_without_check_points_caps_at_half():
    """Without independent check points, re_estimate() cannot confirm the homography
    is correct and must cap confidence at 0.5 per the docstring contract."""
    _, conf = PlanarMapper.re_estimate(PIXEL_CORNERS, WORLD_CORNERS)
    assert conf == 0.5, (
        f"Expected confidence capped at 0.5 with no check points, got {conf}"
    )
