"""Pure-logic tests for common/config_loader.py per TECHSPEC.md Section 8."""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config_loader import load_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MINIMAL_VALID_FOOTBALL_CFG = {
    "pitch": {
        "length_m": 105.0, "width_m": 68.0, "goal_width_m": 7.32,
        "penalty_box_length_m": 16.5, "penalty_box_width_m": 40.32,
        "center_circle_radius_m": 9.15,
    },
    "calibration": {
        "pixel_corners_fallback": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "recalibration_interval_frames": 150, "min_calibration_confidence": 0.75,
        "pitch_hue_range": [35, 85], "pitch_min_saturation": 40,
    },
    "detection": {"det_conf_person": 0.35, "det_conf_ball": 0.35, "input_size": 640},
    "team_classification": {
        "calibration_frames": 50, "min_calib_samples": 25, "referee_max_mean_sat": 55,
        "referee_min_contrast": 42, "min_sat_for_cluster": 60, "vote_buffer_len": 8,
        "vote_lock_thresh": 5,
    },
    "ball": {"gap_predict_frames": 6, "max_realistic_speed_kmh": 38.0},
    "sprint": {"sprint_kmh": 20.0, "sprint_min_frames": 5},
    "possession": {"radius_m": 2.2, "pass_min_distance_m": 5.0},
    "shots": {"velocity_threshold_ms": 15.0, "angle_tolerance_deg": 25.0, "max_distance_m": 35.0},
    "ocr": {"enabled": True, "interval_frames": 10, "min_votes": 3},
    "formation": {"enabled": True, "update_interval_frames": 90},
    "goal_detection": {"cooldown_frames": 90},
    "colors": {"team_a": [230, 230, 230], "team_b": [40, 90, 235], "ball": [0, 215, 255], "goal_flash": [0, 255, 90]},
}


def test_real_football_config_loads():
    cfg = load_config("football", path=os.path.join(REPO_ROOT, "config", "football_config.yaml"))
    assert cfg["pitch"]["length_m"] == 105.0
    assert cfg["detection"]["det_conf_person"] == 0.35


def test_valid_minimal_config_loads(tmp_path):
    p = tmp_path / "football_config.yaml"
    p.write_text(yaml.safe_dump(MINIMAL_VALID_FOOTBALL_CFG))
    cfg = load_config("football", path=str(p))
    assert cfg["pitch"]["width_m"] == 68.0


def test_missing_required_section_raises_keyerror(tmp_path):
    bad = dict(MINIMAL_VALID_FOOTBALL_CFG)
    del bad["team_classification"]
    p = tmp_path / "football_config.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(KeyError):
        load_config("football", path=str(p))


def test_missing_required_key_raises_keyerror(tmp_path):
    bad = {k: dict(v) if isinstance(v, dict) else v for k, v in MINIMAL_VALID_FOOTBALL_CFG.items()}
    del bad["pitch"]["goal_width_m"]
    p = tmp_path / "football_config.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(KeyError):
        load_config("football", path=str(p))


def test_missing_file_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config("football", path=str(tmp_path / "does_not_exist.yaml"))


def test_unknown_sport_raises_valueerror(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(MINIMAL_VALID_FOOTBALL_CFG))
    with pytest.raises(ValueError):
        load_config("rugby", path=str(p))
