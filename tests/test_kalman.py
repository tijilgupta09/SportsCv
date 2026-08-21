"""Pure-logic tests for common/kalman.py per TECHSPEC.md Section 8."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.kalman import Kalman2D


def test_predicts_straight_line_through_short_gap():
    kf = Kalman2D(process_noise=0.01, measurement_noise=0.1)
    for x in range(8):
        kf.predict()
        kf.correct(float(x), float(2 * x))

    predicted = []
    for _ in range(3):
        predicted.append(kf.predict())

    expected = [(8.0, 16.0), (9.0, 18.0), (10.0, 20.0)]
    for state, (exp_x, exp_y) in zip(predicted, expected):
        x, y, vx, vy = state
        assert abs(x - exp_x) < 1.5
        assert abs(y - exp_y) < 3.0
        assert vx > 0.5
        assert vy > 1.0