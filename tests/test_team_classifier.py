"""Pure-logic tests for common/team_classifier.py per TECHSPEC.md Section 8.

Tests use synthetic, well-separated HSV cluster data so no video fixture is needed
to run in CI — consistent with the overall test strategy defined in TECHSPEC.md §8.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.team_classifier import TeamClassifier, is_referee


# ---------------------------------------------------------------------------
#  Shared test config: minimal but sufficient for the classifier
# ---------------------------------------------------------------------------
_TC_KWARGS = dict(
    calibration_frames=0,   # fit immediately (no calibration wait period)
    min_calib_samples=2,    # only need a couple of samples to proceed
    min_sat_for_cluster=40,
    vote_buffer_len=8,
    vote_lock_thresh=5,
)

# Two well-separated hue clusters: "Team A" near hue=10 (red), "Team B" near hue=120 (blue).
# Each sample is (hue, sat, gray_mean, gray_std).
_TEAM_A_SAMPLES = [(10.0, 180.0, 120.0, 25.0)] * 10   # red jerseys
_TEAM_B_SAMPLES = [(120.0, 180.0, 90.0, 20.0)] * 10   # blue jerseys

# Referee-like sample: low saturation, high contrast
_REF_SAMPLE = (60.0, 20.0, 180.0, 60.0)   # near-grey high-contrast


def _build_fitted_classifier():
    """Return a TeamClassifier that has been calibrated on the two synthetic clusters."""
    tc = TeamClassifier(**_TC_KWARGS)
    for stats in _TEAM_A_SAMPLES + _TEAM_B_SAMPLES:
        tc.add_sample(stats)
    tc.maybe_fit(frame_idx=0)   # calibration_frames=0, so fits immediately
    assert tc.centers is not None, "Classifier did not fit — check min_calib_samples"
    return tc


def test_two_cluster_split_assigns_different_teams():
    """Feed two clearly separated hue clusters; they must be assigned to different teams."""
    tc = _build_fitted_classifier()

    # Feed enough votes to lock both tracks
    for _ in range(8):
        result_a = tc.classify(tid=1, stats=_TEAM_A_SAMPLES[0])
        result_b = tc.classify(tid=2, stats=_TEAM_B_SAMPLES[0])

    assert result_a in ("A", "B"), f"Unexpected team for A-cluster: {result_a}"
    assert result_b in ("A", "B"), f"Unexpected team for B-cluster: {result_b}"
    # The two clusters must get opposite assignments
    assert result_a != result_b, (
        f"Both clusters assigned the same team ({result_a}); "
        "k-means should have split them apart"
    )


def test_vote_lock_stabilises_classification():
    """Once a track has vote_lock_thresh consistent votes it should return the same team
    every subsequent call, even if the cluster centroid drifted slightly."""
    tc = _build_fitted_classifier()

    # Drive track 1 to lock on Team A's cluster
    for _ in range(8):
        tc.classify(tid=1, stats=_TEAM_A_SAMPLES[0])

    assert 1 in tc.locked, "Track 1 should have been locked by now"
    locked_team = tc.locked[1]

    # Now classify the same track many more times — must stay locked
    for _ in range(20):
        result = tc.classify(tid=1, stats=_TEAM_A_SAMPLES[0])
        assert result == locked_team, (
            f"Vote lock broken: got {result} after locking to {locked_team}"
        )


def test_referee_rejection_by_is_referee():
    """Low-saturation / high-contrast sample must be identified as a referee."""
    assert is_referee(_REF_SAMPLE, referee_max_mean_sat=55, referee_min_contrast=42), (
        "Referee sample was not flagged as referee"
    )


def test_non_referee_not_flagged():
    """A saturated, low-contrast player sample must NOT be flagged as referee."""
    player_sample = (30.0, 160.0, 100.0, 10.0)   # orange jersey, low contrast
    assert not is_referee(player_sample, referee_max_mean_sat=55, referee_min_contrast=42), (
        "Player sample incorrectly flagged as referee"
    )


def test_low_saturation_sample_excluded_from_clustering():
    """Samples below min_sat_for_cluster should not be added to the sample pool."""
    tc = TeamClassifier(**_TC_KWARGS)
    low_sat = (10.0, 30.0, 150.0, 10.0)   # sat=30 < min_sat_for_cluster=40
    tc.add_sample(low_sat)
    assert len(tc.samples) == 0, "Low-saturation sample should have been rejected"


def test_classify_returns_none_before_fit():
    """Before calibration completes, classify() must return None not crash."""
    tc = TeamClassifier(**_TC_KWARGS)
    # Add a sample but don't call maybe_fit
    tc.add_sample(_TEAM_A_SAMPLES[0])
    result = tc.classify(tid=1, stats=_TEAM_A_SAMPLES[0])
    assert result is None, f"Expected None before fit, got {result}"
