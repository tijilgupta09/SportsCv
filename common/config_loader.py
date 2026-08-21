"""Loads and validates a sport's YAML config, raising clear errors on missing required keys."""
import os
import yaml

REQUIRED_KEYS = {
    "football": {
        "pitch": ["length_m", "width_m", "goal_width_m", "penalty_box_length_m",
                  "penalty_box_width_m", "center_circle_radius_m"],
        "calibration": ["pixel_corners_fallback", "recalibration_interval_frames",
                        "min_calibration_confidence", "pitch_hue_range", "pitch_min_saturation"],
        "detection": ["det_conf_person", "det_conf_ball", "input_size"],
        "team_classification": ["calibration_frames", "min_calib_samples", "referee_max_mean_sat",
                                 "referee_min_contrast", "min_sat_for_cluster",
                                 "vote_buffer_len", "vote_lock_thresh"],
        "ball": ["gap_predict_frames", "max_realistic_speed_kmh"],
        "sprint": ["sprint_kmh", "sprint_min_frames"],
        "possession": ["radius_m", "pass_min_distance_m"],
        "shots": ["velocity_threshold_ms", "angle_tolerance_deg", "max_distance_m"],
        "ocr": ["enabled", "interval_frames", "min_votes"],
        "formation": ["enabled", "update_interval_frames"],
        "goal_detection": ["cooldown_frames"],
        "colors": ["team_a", "team_b", "ball", "goal_flash"],
    },
    "cricket": {
        "pitch": ["length_m", "stump_width_m", "popping_crease_offset_m"],
        "calibration": ["camera_angle", "stump_pixel_points_far", "stump_pixel_points_near",
                        "auto_detect_attempt"],
        "detection": ["det_conf_person", "det_conf_ball", "ball_input_size"],
        "kalman": ["process_noise", "measurement_noise", "max_predict_gap_frames"],
        "delivery": ["release_zone_pixel_radius", "release_velocity_threshold_ms",
                     "delivery_timeout_frames", "bounce_zone_y_fraction"],
        "speed_gun": ["display_duration_frames"],
        "wagon_wheel": ["enabled", "contact_velocity_delta_threshold"],
        "scoreboard_ocr": ["enabled", "roi_pixels", "read_interval_frames"],
        "colors": ["ball_trail", "bounce_flash", "speed_gun_text"],
    },
}


def _default_config_path(sport):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "config", f"{sport}_config.yaml")


def load_config(sport, path=None):
    """Load config/<sport>_config.yaml, validating it against the required-key schema
    from SCHEMA.md. Raises a clear, actionable error naming the missing key/section
    and expected shape rather than a raw KeyError."""
    if sport not in REQUIRED_KEYS:
        raise ValueError(f"[CONFIG] Unknown sport '{sport}'. Expected one of: {list(REQUIRED_KEYS)}")

    cfg_path = path or _default_config_path(sport)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"[ERROR] Config file not found: {cfg_path}\n"
            f"See SCHEMA.md for the expected config/{sport}_config.yaml layout and example values."
        )

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    for section, keys in REQUIRED_KEYS[sport].items():
        if section not in cfg:
            raise KeyError(
                f"[CONFIG] Missing required section '{section}' in {cfg_path}. "
                f"See SCHEMA.md config/{sport}_config.yaml for the expected layout."
            )
        for key in keys:
            if key not in cfg[section]:
                raise KeyError(
                    f"[CONFIG] Missing required key '{section}.{key}' in {cfg_path}. "
                    f"See SCHEMA.md config/{sport}_config.yaml for the expected type/value."
                )

    return cfg

