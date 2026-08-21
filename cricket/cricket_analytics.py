import csv
import json
import math
import os
import subprocess
import sys

# pip package name -> actual importable module name
PKG_TO_MODULE = {
    "opencv-python": "cv2",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "ultralytics": "ultralytics",
    "Pillow": "PIL",
    "tqdm": "tqdm",
    "lapx": "lap",
    "PyYAML": "yaml",
}


def install_deps():
    print("[SETUP] Checking dependencies...", flush=True)
    for pkg, mod in PKG_TO_MODULE.items():
        try:
            __import__(mod)
        except ImportError:
            print(f"[INSTALL] {pkg} (missing) ...", flush=True)
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg, "-q", "--disable-pip-version-check"]
                )
            except subprocess.CalledProcessError as exc:
                print(f"[ERROR] Failed to install {pkg}: {exc}", flush=True)
                sys.exit(1)
    print("[SETUP] All dependencies ready", flush=True)


install_deps()

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config_loader import load_config
from common.dashboard_utils import (
    make_dark_figure, apply_dark_theme, style_table,
    DARK_BG, DARK_EDGE, DARK_LABEL, ACCENT_CYAN, ACCENT_BLUE,
)
from common.detection import run_detect
from common.draw_utils import txt
from common.homography import PlanarMapper
from common.kalman import Kalman2D
from common.video_io import make_writer
from common import ocr as _ocr_mod

# ============================================================================
#  CRICKET CAMERA ASSUMPTION
#  This MVP assumes a mostly static bowler's-end wide shot. The calibration in
#  Phase 2 maps configured stump/pitch points to world coordinates; clips from
#  side-on, handheld, or heavy pan/zoom cameras should be treated as unsupported
#  until their config points are verified via cricket_pitch_preview.png.
# ============================================================================

# ---------------------------------------------------------------------------
#  Delivery state machine states (Task 2.7)
# ---------------------------------------------------------------------------
STATE_IDLE = "IDLE"
STATE_RELEASED = "RELEASED"
STATE_IN_FLIGHT = "IN_FLIGHT"
STATE_BOUNCED = "BOUNCED"
STATE_COMPLETE = "COMPLETE"


# ---------------------------------------------------------------------------
#  Calibration helpers (Tasks 2.3)
# ---------------------------------------------------------------------------

def build_pitch_mapper(cfg):
    far_left, far_right = [tuple(pt) for pt in cfg["calibration"]["stump_pixel_points_far"]]
    near_left, near_right = [tuple(pt) for pt in cfg["calibration"]["stump_pixel_points_near"]]
    length_m = cfg["pitch"]["length_m"]
    stump_width_m = cfg["pitch"]["stump_width_m"]
    half_width = stump_width_m / 2.0
    pixel_pts = [far_left, far_right, near_right, near_left]
    world_pts = [
        (-half_width, 0.0),
        (half_width, 0.0),
        (half_width, length_m),
        (-half_width, length_m),
    ]
    return PlanarMapper(pixel_pts, world_pts), pixel_pts


def draw_pitch_preview(frame, pixel_pts):
    preview = frame.copy()
    pts = np.array(pixel_pts, dtype=np.int32)
    cv2.polylines(preview, [pts], True, (0, 210, 255), 2, cv2.LINE_AA)
    labels = ["far L", "far R", "near R", "near L"]
    for idx, (x, y) in enumerate(pixel_pts):
        cv2.circle(preview, (int(x), int(y)), 6, (0, 60, 255), -1)
        txt(preview, labels[idx], int(x) + 8, int(y) - 8, 0.55, (255, 255, 255), 2)
    return preview


# ---------------------------------------------------------------------------
#  Ball detection pass (Task 2.4)
# ---------------------------------------------------------------------------

def run_ball_detection_pass(video_path: str, cfg: dict) -> dict:
    ball_conf = cfg["detection"]["det_conf_ball"]
    default_conf = cfg["detection"]["det_conf_person"]
    input_size = cfg["detection"]["ball_input_size"]

    print("[YOLO] Loading detection model for cricket ball pass...", flush=True)
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    low_conf_frames = 0
    default_conf_frames = 0
    detections = []
    frame_idx = 0

    print(
        f"[YOLO] Ball-only pass: class=sports ball(32), conf={ball_conf:.2f}, imgsz={input_size}",
        flush=True,
    )
    with tqdm(total=total, unit="fr", ncols=80, colour="cyan") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets = run_detect(
                model, frame,
                classes=[32],
                conf=ball_conf,
                imgsz=input_size,
            )
            frame_candidates = [
                {"frame": frame_idx, "conf": d["conf"], "box": d["box"]}
                for d in dets
            ]
            if frame_candidates:
                low_conf_frames += 1
                detections.extend(frame_candidates)
            if any(c["conf"] >= default_conf for c in frame_candidates):
                default_conf_frames += 1
            frame_idx += 1
            pbar.update(1)
    cap.release()

    print(
        f"[STATS] Ball frames @ {ball_conf:.2f}: {low_conf_frames}/{frame_idx} | "
        f"@ {default_conf:.2f}: {default_conf_frames}/{frame_idx}",
        flush=True,
    )
    return {
        "frames_processed": frame_idx,
        "low_conf_frames": low_conf_frames,
        "default_conf_frames": default_conf_frames,
        "detections": detections,
    }


# ---------------------------------------------------------------------------
#  Task 2.6 — Kalman-backed trajectory buffer
# ---------------------------------------------------------------------------

class BallTrajectory:
    """Maintains a rolling Kalman-backed ball state for one delivery.
    On each frame call `update(cx, cy)` if the ball is detected, or
    `update(None, None)` to predict through a gap (up to max_gap frames).
    """

    def __init__(self, process_noise: float, measurement_noise: float, max_gap: int):
        self.kf = Kalman2D(process_noise=process_noise, measurement_noise=measurement_noise)
        self.max_gap = max_gap
        # Trail history: list of (x, y, is_predicted)
        self.trail: list[tuple[float, float, bool]] = []
        self._gap_count = 0

    def update(self, cx: float | None, cy: float | None) -> tuple[float, float, float, float, bool] | None:
        """Feed a detection (or None for gap frame).
        Returns (x, y, vx, vy, is_predicted) or None if gap exceeded max_gap.
        """
        if cx is not None and cy is not None:
            self._gap_count = 0
            self.kf.predict()
            x, y, vx, vy = self.kf.correct(cx, cy)
            is_predicted = False
        else:
            self._gap_count += 1
            if self._gap_count > self.max_gap:
                return None   # trajectory lost
            x, y, vx, vy = self.kf.predict()
            is_predicted = True

        self.trail.append((x, y, is_predicted))
        return x, y, vx, vy, is_predicted

    def reset(self):
        self.kf = Kalman2D(
            process_noise=self.kf.filter.processNoiseCov[0, 0],
            measurement_noise=self.kf.filter.measurementNoiseCov[0, 0],
        )
        self.trail.clear()
        self._gap_count = 0


# ---------------------------------------------------------------------------
#  Task 2.7 — Delivery state machine
# ---------------------------------------------------------------------------

class DeliveryStateMachine:
    """IDLE → RELEASED → IN_FLIGHT → BOUNCED → COMPLETE per TECHSPEC.md 6.5."""

    def __init__(self, cfg: dict):
        dc = cfg["delivery"]
        self.release_zone_r = dc["release_zone_pixel_radius"]
        self.release_vel_ms = dc["release_velocity_threshold_ms"]
        self.timeout_frames = dc["delivery_timeout_frames"]
        bounce_fracs = dc["bounce_zone_y_fraction"]
        self.bounce_zone_y_frac_lo = bounce_fracs[0]
        self.bounce_zone_y_frac_hi = bounce_fracs[1]
        # TECHSPEC.md 6.5: ball entering batsman zone also completes the delivery.
        # Default 0.30 means the top 30% of the frame is the "batsman zone".
        self.batsman_zone_y_frac = dc.get("batsman_zone_y_fraction", 0.30)

        self.state = STATE_IDLE
        self.delivery_no = 0
        self.records: list[dict] = []          # completed DeliveryRecord dicts
        self._active: dict | None = None       # in-progress record scratch dict
        self._prev_vy: float | None = None
        self._idle_frames = 0
        self._last_ball_frame = -999
        # Near-stump pixel y used to define release zone centre (set externally)
        self.release_zone_centre: tuple[float, float] | None = None

    def _new_record(self, frame_idx: int) -> dict:
        return {
            "delivery_no": self.delivery_no,
            "release_frame": frame_idx,
            "bounce_frame": None,
            "bounce_x_m": None,
            "bounce_y_m": None,
            "complete_frame": None,
            "speed_kmh": 0.0,
            "predicted_frames": 0,
            # Phase 3 fields (Task 3.5 fast/spin, Task 3.1 contact)
            "delivery_type": "unknown",
            "contact_frame": None,
            "contact_x_m": None,
            "contact_y_m": None,
        }

    def update(self, frame_idx: int, fps: float,
               ball_state: tuple[float, float, float, float, bool] | None,
               mapper: "PlanarMapper | None",
               frame_h: int) -> str:
        """Drive one frame of the state machine.
        Returns the current state string (for HUD / logging).
        ball_state: (x, y, vx, vy, is_predicted) in pixel space, or None if ball lost.
        """
        if ball_state is None:
            # No ball visible at all
            if self.state not in (STATE_IDLE, STATE_COMPLETE):
                frames_since = frame_idx - self._last_ball_frame
                if frames_since > self.timeout_frames:
                    self._finalize(frame_idx)
            return self.state

        x, y, vx, vy, is_predicted = ball_state
        speed_px_per_frame = (vx ** 2 + vy ** 2) ** 0.5

        # Convert px/frame velocity to m/s via homography if available
        speed_ms = 0.0
        if mapper is not None and speed_px_per_frame > 0:
            wx1, wy1 = mapper.to_world(x, y)
            wx2, wy2 = mapper.to_world(x + vx, y + vy)
            speed_ms = ((wx2 - wx1) ** 2 + (wy2 - wy1) ** 2) ** 0.5 * fps

        if not is_predicted:
            self._last_ball_frame = frame_idx

        if self._active and is_predicted:
            self._active["predicted_frames"] += 1

        # --- State transitions ---
        if self.state == STATE_IDLE:
            # Check if ball has left the release zone at sufficient speed
            if self._ball_in_release_zone(x, y) and speed_ms >= self.release_vel_ms:
                self.delivery_no += 1
                self._active = self._new_record(frame_idx)
                self.state = STATE_RELEASED
                self._prev_vy = vy
                print(f"[DELIVERY] frame={frame_idx} → RELEASED (delivery #{self.delivery_no})", flush=True)

        elif self.state == STATE_RELEASED:
            # Immediately move to IN_FLIGHT
            self.state = STATE_IN_FLIGHT
            self._prev_vy = vy
            print(f"[DELIVERY] frame={frame_idx} → IN_FLIGHT", flush=True)

        elif self.state == STATE_IN_FLIGHT:
            # Task 2.8 — Bounce detection: vy sign changes + in lower pitch zone
            if self._prev_vy is not None and self._in_bounce_zone(y, frame_h):
                if self._prev_vy > 0 and vy < 0:   # downward → upward (image coords)
                    wx, wy = (None, None)
                    if mapper is not None:
                        wx, wy = mapper.to_world(x, y)
                    self._active["bounce_frame"] = frame_idx
                    self._active["bounce_x_m"] = wx
                    self._active["bounce_y_m"] = wy
                    self.state = STATE_BOUNCED
                    print(
                        f"[DELIVERY] frame={frame_idx} → BOUNCED @ world=({wx},{wy})",
                        flush=True,
                    )
            self._prev_vy = vy

            # Minor fix: also complete in IN_FLIGHT if ball enters batsman zone
            # (covers full-toss/yorker that never bounces — BUG-11 gap).
            if self._in_batsman_zone(y, frame_h):
                self._finalize(frame_idx)
            elif frame_idx - self._last_ball_frame > self.timeout_frames:
                self._finalize(frame_idx)

        elif self.state == STATE_BOUNCED:
            # After bounce, wait for timeout or ball to disappear;
            # also trigger COMPLETE if ball enters the batsman zone (upper frame).
            if self._in_batsman_zone(y, frame_h):
                self._finalize(frame_idx)
            elif frame_idx - self._last_ball_frame > self.timeout_frames:
                self._finalize(frame_idx)

        elif self.state == STATE_COMPLETE:
            # Start watching for next delivery with a short dead-time
            self._idle_frames += 1
            if self._idle_frames > 30:
                self.state = STATE_IDLE
                self._idle_frames = 0
                self._prev_vy = None

        return self.state

    def _ball_in_release_zone(self, x: float, y: float) -> bool:
        """Ball is near the bowler's release area (far-stump end of pitch)."""
        if self.release_zone_centre is None:
            return True   # conservative: always allow if not configured
        cx, cy = self.release_zone_centre
        return (x - cx) ** 2 + (y - cy) ** 2 <= self.release_zone_r ** 2

    def _in_bounce_zone(self, y: float, frame_h: int) -> bool:
        """Vertical bounce zone: lower fraction of the frame."""
        rel = y / frame_h if frame_h > 0 else 0.5
        return self.bounce_zone_y_frac_lo <= rel <= self.bounce_zone_y_frac_hi

    def _in_batsman_zone(self, y: float, frame_h: int) -> bool:
        """Batsman zone: upper portion of the frame where the striker stands.
        TECHSPEC.md 6.5 specifies COMPLETE fires here rather than only on timeout.
        Threshold is config-driven via delivery.batsman_zone_y_fraction (default 0.30).
        """
        rel = y / frame_h if frame_h > 0 else 0.5
        return rel < self.batsman_zone_y_frac

    def _finalize(self, frame_idx: int):
        """Close the active delivery record and compute speed."""
        if self._active is None:
            self.state = STATE_COMPLETE
            return
        self._active["complete_frame"] = frame_idx
        # Task 2.9 — speed: stored at release, finalized here
        # Speed was being accumulated externally; if not yet set, leave 0.0
        self.records.append(self._active)
        self._active = None
        self.state = STATE_COMPLETE
        self._idle_frames = 0
        print(f"[DELIVERY] frame={frame_idx} → COMPLETE (delivery #{self.delivery_no})", flush=True)

    # Task 2.9 helper — set speed on active record
    def set_release_speed(self, speed_kmh: float):
        if self._active is not None:
            self._active["speed_kmh"] = speed_kmh

    # Task 3.5 helper — set delivery type on active record
    def set_delivery_type(self, delivery_type: str):
        if self._active is not None:
            self._active["delivery_type"] = delivery_type

    # Task 3.1 helper — record bat-contact on active record
    def set_contact(self, frame_idx: int, wx: float | None, wy: float | None):
        if self._active is not None:
            self._active["contact_frame"] = frame_idx
            self._active["contact_x_m"] = wx
            self._active["contact_y_m"] = wy


# ---------------------------------------------------------------------------
#  Task 2.9 — Speed estimation helper
# ---------------------------------------------------------------------------

class SpeedEstimator:
    """Tracks ball position across IN_FLIGHT to estimate release→arrival speed."""

    def __init__(self, fps: float, mapper: "PlanarMapper | None"):
        self.fps = fps
        self.mapper = mapper
        self._release_world: tuple[float, float] | None = None
        self._release_frame: int | None = None
        self._last_speed_kmh: float = 0.0
        self._active = False

    def on_released(self, frame_idx: int, px: float, py: float):
        if self.mapper:
            self._release_world = self.mapper.to_world(px, py)
        self._release_frame = frame_idx
        self._active = True

    def on_complete(self, frame_idx: int, px: float, py: float) -> float:
        if not self._active or self._release_frame is None or self.mapper is None:
            return 0.0
        arrival_world = self.mapper.to_world(px, py)
        rx, ry = self._release_world
        ax, ay = arrival_world
        dist_m = ((ax - rx) ** 2 + (ay - ry) ** 2) ** 0.5
        elapsed_s = (frame_idx - self._release_frame) / max(self.fps, 1.0)
        if elapsed_s <= 0:
            return 0.0
        speed_ms = dist_m / elapsed_s
        self._last_speed_kmh = speed_ms * 3.6
        self._active = False
        return self._last_speed_kmh

    def last_speed(self) -> float:
        return self._last_speed_kmh

    def reset(self, mapper: "PlanarMapper | None"):
        self.mapper = mapper
        self._release_world = None
        self._release_frame = None
        self._active = False


# ---------------------------------------------------------------------------
#  Task 2.10 — Speed-gun HUD
# ---------------------------------------------------------------------------

def draw_speed_gun(frame: np.ndarray, speed_kmh: float, display_frames_left: int,
                   color_bgr: tuple) -> None:
    """Draw a speed-gun HUD at bottom-centre of the frame."""
    if display_frames_left <= 0 or speed_kmh <= 0:
        return
    h, w = frame.shape[:2]
    label = f"SPEED: {speed_kmh:.1f} km/h"
    panel_w, panel_h = 320, 52
    px = (w - panel_w) // 2
    py = h - panel_h - 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (8, 12, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    txt(frame, "SPEED GUN", px + 12, py + 16, 0.45, (160, 160, 160), 1)
    txt(frame, label, px + 12, py + 38, 0.75, color_bgr, 2)


# ---------------------------------------------------------------------------
#  Task 2.10 — Draw ball trail on frame
# ---------------------------------------------------------------------------

def draw_ball_trail(frame: np.ndarray, trail: list[tuple[float, float, bool]],
                    color_bgr: tuple, max_trail: int = 30) -> None:
    """Draw the Kalman ball trail (last max_trail points)."""
    recent = trail[-max_trail:]
    for i, (x, y, is_pred) in enumerate(recent):
        alpha = (i + 1) / len(recent)
        radius = max(2, int(4 * alpha))
        color = color_bgr if not is_pred else (80, 80, 80)
        cv2.circle(frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
#  HUD helpers
# ---------------------------------------------------------------------------

def draw_top_banner(frame: np.ndarray, delivery_no: int, state: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 36), (10, 8, 90), -1)
    txt(frame, "CRICKET DELIVERY ANALYTICS", 10, 24, 0.65, (255, 255, 255), 2)
    right_label = f"Delivery #{delivery_no}  [{state}]"
    ts, _ = cv2.getTextSize(right_label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    txt(frame, right_label, w - ts[0] - 12, 24, 0.55, (0, 210, 255), 2)


def draw_pitch_inset(frame: np.ndarray, pixel_pts: list, trail: list, bounce_world: tuple | None,
                     mapper: "PlanarMapper | None", cfg: dict) -> None:
    """Draw a miniature pitch map inset in the bottom-right corner."""
    h, w = frame.shape[:2]
    iw, ih = 140, 90
    margin = 10
    ox = w - iw - margin
    oy = h - ih - margin - 56   # above speed-gun panel

    overlay = frame.copy()
    cv2.rectangle(overlay, (ox - 4, oy - 4), (ox + iw + 4, oy + ih + 4), (8, 12, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Draw pitch rectangle
    cv2.rectangle(frame, (ox, oy), (ox + iw, oy + ih), (0, 120, 0), 1)
    txt(frame, "PITCH", ox + 4, oy + 12, 0.35, (160, 160, 160), 1)

    # Plot recent trail on minimap using homography
    if mapper is not None:
        length_m = cfg["pitch"]["length_m"]
        for x, y, is_pred in trail[-20:]:
            try:
                wx, wy = mapper.to_world(x, y)
                # map world → inset pixel
                ipx = int(ox + (wx / (cfg["pitch"]["stump_width_m"] + 2.0) + 0.5) * iw)
                ipy = int(oy + (wy / length_m) * ih)
                col = (0, 210, 255) if not is_pred else (80, 80, 80)
                cv2.circle(frame, (ipx, ipy), 2, col, -1)
            except Exception:
                pass

        # Bounce marker
        if bounce_world is not None:
            bwx, bwy = bounce_world
            bipx = int(ox + (bwx / (cfg["pitch"]["stump_width_m"] + 2.0) + 0.5) * iw)
            bipy = int(oy + (bwy / length_m) * ih)
            cv2.circle(frame, (bipx, bipy), 4, (0, 60, 255), -1)


# ---------------------------------------------------------------------------
#  Task 3.5 — Fast/spin delivery heuristic
#  Rule: speed > 120 km/h → "fast"; speed < 90 km/h → "spin"; else "medium".
#  Clearly a heuristic/approximation — documented as such per RULES.md §8.
# ---------------------------------------------------------------------------

# Speed band thresholds (km/h) — tunable via config if needed in future;
# currently hardcoded as physics-based constants unlikely to differ between clips.
_FAST_THRESHOLD_KMH = 120.0
_SPIN_THRESHOLD_KMH = 90.0


def classify_delivery_type(speed_kmh: float) -> str:
    """Heuristic (est.) fast/medium/spin classification by speed only.
    Not a broadcast-grade classifier — labeled 'est.' wherever displayed.
    """
    if speed_kmh <= 0:
        return "unknown"
    if speed_kmh >= _FAST_THRESHOLD_KMH:
        return "fast"
    if speed_kmh <= _SPIN_THRESHOLD_KMH:
        return "spin"
    return "medium"


# ---------------------------------------------------------------------------
#  Task 3.1 — Bat-contact point detection
#  Heuristic: sudden velocity/direction change in the ball trajectory frames
#  following the bounce, in the batsman zone (upper portion of pitch image).
#  This mirrors the football shot/spike detection pattern per TECHSPEC.md 6.6.
# ---------------------------------------------------------------------------

_CONTACT_VEL_DELTA_THRESHOLD = 20.0   # px/frame magnitude change — falls back to
                                       # wagon_wheel.contact_velocity_delta_threshold in config


class BatContactDetector:
    """Detects approximate bat-contact frame by watching for a sudden change in
    ball velocity/direction magnitude after the bounce. Explicitly a heuristic;
    results logged as 'est.' (RULES.md §8). Enabled only when wagon_wheel.enabled."""

    def __init__(self, vel_delta_threshold: float):
        self.threshold = vel_delta_threshold
        self._prev_speed: float | None = None
        self._active = False

    def reset(self):
        self._prev_speed = None
        self._active = False

    def arm(self):
        """Call after BOUNCED state to start watching for contact."""
        self._active = True
        self._prev_speed = None

    def update(self, vx: float, vy: float) -> bool:
        """Returns True on the frame where contact is estimated."""
        if not self._active:
            return False
        speed = math.hypot(vx, vy)
        triggered = False
        if self._prev_speed is not None:
            delta = abs(speed - self._prev_speed)
            if delta >= self.threshold:
                triggered = True
                self._active = False   # fire once per delivery
        self._prev_speed = speed
        return triggered


# ---------------------------------------------------------------------------
#  Task 3.2 — Wagon wheel PNG export
# ---------------------------------------------------------------------------

def save_wagon_wheel(records: list[dict], cfg: dict, out_path: str) -> None:
    """Generate cricket_wagon_wheel.png — circular field diagram with direction
    lines from bat-contact point for deliveries where contact was detected.
    Per DESIGN.md and TECHSPEC.md 6.6. Only called when wagon_wheel.enabled."""
    length_m = cfg["pitch"]["length_m"]
    field_r = 60.0   # approximate half-field radius for display (meters)

    contact_records = [
        r for r in records
        if r.get("contact_x_m") is not None and r.get("contact_y_m") is not None
    ]

    fig, ax = make_dark_figure(figsize=(8, 8), subplot_kw={"projection": "polar"})
    apply_dark_theme(ax)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, field_r)
    ax.set_yticks([20, 40, 60])
    ax.set_yticklabels(["20m", "40m", "60m"], color=DARK_LABEL, fontsize=7)
    ax.grid(color=DARK_EDGE, linewidth=0.5)
    ax.spines["polar"].set_color(DARK_EDGE)

    cmap = plt.cm.get_cmap("RdYlGn_r")

    if contact_records:
        for rec in contact_records:
            # Direction vector: from pitch centre toward contact point (rough estimate).
            # The contact world coords are in the pitch frame (x=lateral, y=length).
            # Project to field angle (N = straight, 90° = off-side etc.)
            cx_m = rec["contact_x_m"]
            cy_m = rec["contact_y_m"] - length_m   # relative to batsman end
            angle_rad = math.atan2(cx_m, -cy_m)    # bearing from batsman toward ball
            speed = rec["speed_kmh"]
            color = cmap(min(max((speed - 60.0) / 100.0, 0.0), 1.0))
            ax.annotate("",
                        xy=(angle_rad, field_r * 0.85),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
            ax.text(angle_rad, field_r * 0.9,
                    f"#{rec['delivery_no']}",
                    color="white", fontsize=6, ha="center")
    else:
        ax.text(0, 0.5,
                "No bat-contact data\n(run with real cricket footage",
                color="#6a8aaa", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)

    fig.suptitle(
        f"Wagon Wheel (est.) — {len(contact_records)} contacts detected\n| dev: abhinav.phi",
        color="#00e0ff", fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[DONE] Saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
#  Task 3.3 / 3.4 — Scoreboard OCR integration + cricket_events.json
# ---------------------------------------------------------------------------

class ScoreboardReader:
    """Rate-limited scoreboard OCR reader (Task 3.3/3.4).
    Reads the broadcast scoreboard ROI every `read_interval_frames` frames.
    Results are tagged source=\"ocr\" per SCHEMA.md 2.4 — never merged silently
    with vision-inferred fields (RULES.md §8)."""

    def __init__(self, roi_pixels: list, read_interval: int, enabled: bool):
        self.roi = tuple(roi_pixels)   # (x1, y1, x2, y2)
        self.interval = read_interval
        self.enabled = enabled
        self._reads: list[dict] = []  # list of SCHEMA.md 2.4 scoreboard_read dicts
        self._last_read_frame = -9999

    def update(self, frame: np.ndarray, frame_idx: int) -> dict | None:
        """Call every frame. Returns a new scoreboard read dict if one fired, else None."""
        if not self.enabled:
            return None
        if frame_idx - self._last_read_frame < self.interval:
            return None
        self._last_read_frame = frame_idx
        raw = _ocr_mod.read_scoreboard(frame, self.roi)
        if raw is None:
            return None
        # Best-effort parse: extract digits for runs/wickets/overs
        import re
        nums = re.findall(r'\d+\.?\d*', raw)
        read_dict = {
            "frame": frame_idx,
            "source": "ocr",
            "raw_text": raw,
            "runs": int(nums[0]) if len(nums) > 0 else None,
            "wickets": int(nums[1]) if len(nums) > 1 else None,
            "overs": nums[2] if len(nums) > 2 else None,
        }
        self._reads.append(read_dict)
        print(
            f"[OCR] frame={frame_idx} scoreboard raw='{raw}' parsed={read_dict}",
            flush=True,
        )
        return read_dict

    def all_reads(self) -> list[dict]:
        return self._reads


def save_cricket_events_json(video_path: str, scoreboard_reads: list[dict], out_path: str) -> None:
    """Write cricket_events.json per SCHEMA.md 2.4.
    Only written when scoreboard OCR is enabled (Task 3.4)."""
    payload = {
        "video": os.path.basename(video_path),
        "generated_by": "abhinav.phi",
        "scoreboard_reads": scoreboard_reads,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[DONE] Saved {out_path} ({len(scoreboard_reads)} scoreboard reads)", flush=True)


# ---------------------------------------------------------------------------
#  Task 2.11 — CSV export  (updated to include delivery_type, Task 3.5)
# ---------------------------------------------------------------------------

def save_deliveries_csv(records: list[dict], out_path: str) -> None:
    """Write cricket_deliveries.csv per SCHEMA.md 2.3, plus delivery_type (Task 3.5)."""
    fieldnames = [
        "delivery_no", "release_frame", "bounce_frame",
        "bounce_x_m", "bounce_y_m", "complete_frame",
        "speed_kmh", "predicted_frames",
        "delivery_type",   # Task 3.5 — fast/medium/spin/unknown (est.)
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})
    print(f"[DONE] Saved {out_path} ({len(records)} deliveries)", flush=True)


# ---------------------------------------------------------------------------
#  Task 2.12 — Pitch map PNG
# ---------------------------------------------------------------------------

def _speed_to_color(speed_kmh: float):
    """Map speed to a matplotlib color: green (slow) → yellow → red (fast)."""
    lo, hi = 60.0, 160.0
    t = min(max((speed_kmh - lo) / (hi - lo), 0.0), 1.0)
    cmap = plt.cm.get_cmap("RdYlGn_r")
    return cmap(t)


def save_pitch_map(records: list[dict], cfg: dict, out_path: str) -> None:
    """Generate cricket_pitch_map.png — top-down pitch with bounce markers.
    This is the highest-priority visual deliverable per IMPLEMENTATIONPLAN.md 2.12.
    """
    length_m = cfg["pitch"]["length_m"]
    width_m = cfg["pitch"]["stump_width_m"]
    vis_width_m = 3.05  # standard visualisation lane width

    fig, ax = make_dark_figure(figsize=(4, 10))
    apply_dark_theme(ax)

    # Draw pitch rectangle
    pitch_rect = mpatches.FancyBboxPatch(
        (-vis_width_m / 2, 0), vis_width_m, length_m,
        boxstyle="round,pad=0.1",
        linewidth=1.5, edgecolor="#4a6a8a", facecolor="#1a2840",
    )
    ax.add_patch(pitch_rect)

    # Stump lines
    for y_val, label in [(0.0, "Bowler's end"), (length_m, "Batsman's end")]:
        ax.axhline(y_val, color="#6a8aaa", linewidth=0.8, linestyle="--")
        ax.text(-vis_width_m / 2 - 0.1, y_val, label, color="#6a8aaa",
                fontsize=6, va="center", ha="right")

    # Popping crease lines
    popping = cfg["pitch"].get("popping_crease_offset_m", 1.22)
    ax.axhline(popping, color="#8aaa6a", linewidth=0.5, linestyle=":")
    ax.axhline(length_m - popping, color="#8aaa6a", linewidth=0.5, linestyle=":")

    # Scatter bounce points
    bounced = [r for r in records if r.get("bounce_x_m") is not None and r.get("bounce_y_m") is not None]
    if bounced:
        for rec in bounced:
            col = _speed_to_color(rec["speed_kmh"])
            ax.scatter(rec["bounce_x_m"], rec["bounce_y_m"], color=col, s=60,
                       edgecolors="white", linewidths=0.5, zorder=5)
            ax.text(rec["bounce_x_m"] + 0.05, rec["bounce_y_m"],
                    f"#{rec['delivery_no']}",
                    color="white", fontsize=5, va="center")
    else:
        ax.text(0, length_m / 2, "No bounce data yet\n(run with real cricket footage)",
                color="#6a8aaa", ha="center", va="center", fontsize=7)

    # Colorbar legend
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(vmin=60, vmax=160))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.02, fraction=0.03)
    cbar.set_label("Speed (km/h)", color=DARK_LABEL, fontsize=7)
    cbar.ax.tick_params(colors=DARK_LABEL, labelsize=6)

    ax.set_xlim(-vis_width_m / 2 - 0.3, vis_width_m / 2 + 0.3)
    ax.set_ylim(-0.5, length_m + 0.5)
    ax.set_xlabel("Width (m)", color=DARK_LABEL, fontsize=7)
    ax.set_ylabel("Length (m)", color=DARK_LABEL, fontsize=7)
    ax.tick_params(colors="#6a8aaa", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a2840")

    fig.suptitle(
        f"Cricket Pitch Map — {len(records)} deliveries\n| dev: abhinav.phi",
        color="#00e0ff", fontsize=9, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[DONE] Saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
#  Task 2.13 — Dashboard PNG
# ---------------------------------------------------------------------------

def save_dashboard(records: list[dict], cfg: dict, out_path: str,
                   pitch_map_path: str | None) -> None:
    """Generate cricket_dashboard.png per DESIGN.md 6.2 — 2×2 grid.
    Updated by Task 3.5 to show fast/spin split in summary table.
    """
    speeds = [r["speed_kmh"] for r in records if r["speed_kmh"] > 0]
    n_deliveries = len(records)
    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    top_speed = float(max(speeds)) if speeds else 0.0
    n_fast = sum(1 for r in records if r.get("delivery_type") == "fast")
    n_spin = sum(1 for r in records if r.get("delivery_type") == "spin")
    n_medium = sum(1 for r in records if r.get("delivery_type") == "medium")
    fast_spin_label = f"Fast {n_fast} / Medium {n_medium} / Spin {n_spin} (est.)"

    fig, axes = make_dark_figure(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Cricket Analytics Dashboard — {n_deliveries} deliveries | dev: abhinav.phi",
        color=ACCENT_CYAN, fontsize=13,
    )

    # ---- Top-left: Speed histogram ----
    ax = axes[0, 0]
    apply_dark_theme(ax)
    if speeds:
        ax.hist(speeds, bins=max(5, n_deliveries // 2 + 1), color="#d47800",
                edgecolor="#1a2840", alpha=0.85)
    else:
        ax.text(0.5, 0.5, "No speed data yet", color="#6a8aaa",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_title("Speed Distribution (km/h)", color=ACCENT_BLUE, fontsize=9)
    ax.set_xlabel("Speed (km/h)", color=DARK_LABEL, fontsize=8)
    ax.set_ylabel("Deliveries", color=DARK_LABEL, fontsize=8)
    ax.tick_params(colors="#6a8aaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a2840")

    # ---- Top-right: Outcome breakdown ----
    ax = axes[0, 1]
    apply_dark_theme(ax)
    n_bounced = sum(1 for r in records if r.get("bounce_frame") is not None)
    n_no_bounce = n_deliveries - n_bounced
    if n_deliveries > 0:
        ax.pie(
            [n_bounced, n_no_bounce] if n_bounced > 0 or n_no_bounce > 0 else [1],
            labels=["Bounce detected", "No bounce / timeout"] if n_deliveries > 0 else ["No data"],
            colors=["#0078d4", "#6a8aaa"],
            autopct="%1.0f%%" if n_deliveries > 0 else None,
            textprops={"color": "white", "fontsize": 8},
            startangle=90,
        )
    else:
        ax.text(0.5, 0.5, "No deliveries yet", color=DARK_LABEL,
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_title("Delivery Outcomes", color=ACCENT_BLUE, fontsize=9)

    # ---- Bottom-left: Pitch map thumbnail ----
    ax = axes[1, 0]
    apply_dark_theme(ax)
    ax.axis("off")
    if pitch_map_path and os.path.isfile(pitch_map_path):
        try:
            img = plt.imread(pitch_map_path)
            ax.imshow(img, aspect="auto")
            ax.set_title("Pitch Map (thumbnail)", color="#00c8ff", fontsize=9)
        except Exception:
            ax.text(0.5, 0.5, "Pitch map unavailable", color="#6a8aaa",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
    else:
        ax.text(0.5, 0.5, "Pitch map not yet generated", color="#6a8aaa",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)

    # ---- Bottom-right: Summary table ----
    ax = axes[1, 1]
    apply_dark_theme(ax)
    ax.axis("off")
    table_data = [
        ["Deliveries", str(n_deliveries)],
        ["Avg Speed (km/h)", f"{avg_speed:.1f}" if avg_speed > 0 else "—"],
        ["Top Speed (km/h)", f"{top_speed:.1f}" if top_speed > 0 else "—"],
        ["Bounces detected", str(sum(1 for r in records if r.get("bounce_frame") is not None))],
        ["Fast/Spin (est.)", fast_spin_label],
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=["Stat", "Value"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    style_table(table)
    ax.set_title("Match Summary", color=ACCENT_BLUE, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[DONE] Saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
#  Main pipeline (process)
# ---------------------------------------------------------------------------

def process(video_path: str, config_path: str | None = None) -> bool:
    print("+------------------------------------------------+", flush=True)
    print("|   CRICKET DELIVERY ANALYTICS  v1.0            |", flush=True)
    print("|   dev: abhinav.phi                            |", flush=True)
    print("+------------------------------------------------+", flush=True)

    cfg = load_config("cricket", path=config_path)
    print(f"[CALIB] camera_angle={cfg['calibration']['camera_angle']}", flush=True)

    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found: {video_path}", flush=True)
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}", flush=True)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        print(f"[ERROR] Could not read first frame from {video_path}", flush=True)
        return False

    # ---- Calibration (Task 2.3) ----
    mapper, pixel_pts = build_pitch_mapper(cfg)
    out_dir = os.path.dirname(os.path.abspath(video_path))

    # Release zone centre = midpoint of near stumps (bowler's end in this camera)
    near_l = tuple(cfg["calibration"]["stump_pixel_points_near"][0])
    near_r = tuple(cfg["calibration"]["stump_pixel_points_near"][1])
    release_centre = ((near_l[0] + near_r[0]) / 2, (near_l[1] + near_r[1]) / 2)

    preview_path = os.path.join(out_dir, "cricket_pitch_preview.png")
    cv2.imwrite(preview_path, draw_pitch_preview(frame0, pixel_pts))
    print(f"[INFO] {width}x{height} @ {fps:.1f}fps | {total} frames", flush=True)
    print(f"[CALIB] Saved {preview_path}", flush=True)

    # ---- Task 2.4: Ball detection pass to pre-gather detections ----
    cap.release()
    ball_stats = run_ball_detection_pass(video_path, cfg)
    if ball_stats["low_conf_frames"] == 0:
        print(
            "[WARN] No cricket ball detections on this clip; use a real bowler's-end cricket clip for meaningful analytics.",
            flush=True,
        )

    # Build per-frame lookup of best ball detection
    ball_conf = cfg["detection"]["det_conf_ball"]
    per_frame_ball: dict[int, tuple[float, float]] = {}
    for det in ball_stats["detections"]:
        fi = det["frame"]
        x1, y1, x2, y2 = det["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # Keep highest-conf detection per frame
        if fi not in per_frame_ball or det["conf"] > ball_conf:
            per_frame_ball[fi] = (cx, cy)

    # ---- Task 2.6: Kalman trajectory ----
    kc = cfg["kalman"]
    traj = BallTrajectory(
        process_noise=kc["process_noise"],
        measurement_noise=kc["measurement_noise"],
        max_gap=kc["max_predict_gap_frames"],
    )

    # ---- Task 2.7: Delivery state machine ----
    fsm = DeliveryStateMachine(cfg)
    fsm.release_zone_centre = release_centre

    # ---- Task 2.9: Speed estimator ----
    speed_est = SpeedEstimator(fps=fps, mapper=mapper)

    # ---- Task 3.1: Bat-contact detector (enabled when wagon_wheel.enabled) ----
    ww_cfg = cfg.get("wagon_wheel", {})
    wagon_wheel_enabled = ww_cfg.get("enabled", False)
    contact_vel_threshold = ww_cfg.get("contact_velocity_delta_threshold",
                                        _CONTACT_VEL_DELTA_THRESHOLD)
    bat_contact = BatContactDetector(vel_delta_threshold=contact_vel_threshold)

    # ---- Task 3.3: Scoreboard OCR reader ----
    sb_cfg = cfg.get("scoreboard_ocr", {})
    scoreboard_reader = ScoreboardReader(
        roi_pixels=sb_cfg.get("roi_pixels", [20, 20, 300, 90]),
        read_interval=sb_cfg.get("read_interval_frames", 60),
        enabled=sb_cfg.get("enabled", False),
    )

    # ---- Video writer ----
    video_out = os.path.join(out_dir, "cricket_output.mp4")
    writer = make_writer(video_out, fps, width, height)

    # ---- Speed-gun state ----
    speed_gun_color = tuple(cfg["colors"]["speed_gun_text"])
    speed_display_frames = cfg["speed_gun"]["display_duration_frames"]
    speed_display_left = 0
    last_speed_kmh = 0.0

    # ---- Bounce tracking for inset ----
    current_bounce_world: tuple[float, float] | None = None

    # ---- Main frame loop ----
    cap = cv2.VideoCapture(video_path)
    prev_state = STATE_IDLE
    # BUG FIX (Task 2.9): track last frame where ball was ACTUALLY detected (not Kalman-predicted).
    # Used as arrival position fallback when timeout fires after Kalman has given up.
    last_real_ball_px: tuple[float, float] | None = None

    print("[PROC] Starting main frame loop...", flush=True)
    with tqdm(total=total, unit="fr", ncols=80, colour="green") as pbar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- Kalman update (Task 2.6) ---
            raw_ball = per_frame_ball.get(frame_idx)
            ball_state = traj.update(
                raw_ball[0] if raw_ball else None,
                raw_ball[1] if raw_ball else None,
            )
            # Track last real (non-predicted) ball position for speed fallback
            if ball_state is not None and not ball_state[4]:   # is_predicted == False
                last_real_ball_px = (ball_state[0], ball_state[1])

            # --- State machine (Tasks 2.7, 2.8) ---
            cur_state = fsm.update(
                frame_idx=frame_idx,
                fps=fps,
                ball_state=ball_state,
                mapper=mapper,
                frame_h=height,
            )

            # --- Speed transitions (Task 2.9) ---
            if prev_state != STATE_RELEASED and cur_state == STATE_RELEASED:
                if ball_state:
                    speed_est.on_released(frame_idx, ball_state[0], ball_state[1])
                traj.reset()
                bat_contact.reset()   # Task 3.1: disarm on new delivery

            if cur_state == STATE_COMPLETE and prev_state != STATE_COMPLETE:
                # BUG FIX (Task 2.9): ball_state is None by the time timeout fires because
                # Kalman gives up after max_predict_gap_frames (8) frames. Use the last
                # real (non-predicted) ball position as the arrival point instead.
                arrival_pos = None
                if ball_state and not ball_state[4]:   # real detection this frame
                    arrival_pos = (ball_state[0], ball_state[1])
                elif last_real_ball_px is not None:    # fallback: last known real position
                    arrival_pos = last_real_ball_px
                if arrival_pos is not None:
                    speed_kmh = speed_est.on_complete(frame_idx, arrival_pos[0], arrival_pos[1])
                else:
                    speed_kmh = 0.0
                if speed_kmh > 0:
                    fsm.set_release_speed(speed_kmh)
                    last_speed_kmh = speed_kmh
                    speed_display_left = speed_display_frames
                    # Task 3.5: classify delivery type immediately when speed is known
                    delivery_type = classify_delivery_type(speed_kmh)
                    fsm.set_delivery_type(delivery_type)
                    print(
                        f"[EVENT] delivery #{fsm.delivery_no} type={delivery_type} (est.) "
                        f"speed={speed_kmh:.1f}km/h",
                        flush=True,
                    )

            if cur_state == STATE_BOUNCED and prev_state != STATE_BOUNCED:
                if fsm._active and fsm._active.get("bounce_x_m") is not None:
                    current_bounce_world = (
                        fsm._active["bounce_x_m"],
                        fsm._active["bounce_y_m"],
                    )
                # Task 3.1: arm bat-contact detector after bounce
                if wagon_wheel_enabled:
                    bat_contact.arm()

            # Task 3.1: check for bat-contact
            if ball_state and cur_state in (STATE_BOUNCED,) and wagon_wheel_enabled:
                vx, vy = ball_state[2], ball_state[3]
                if bat_contact.update(vx, vy):
                    wx, wy = (None, None)
                    if mapper is not None:
                        wx, wy = mapper.to_world(ball_state[0], ball_state[1])
                    fsm.set_contact(frame_idx, wx, wy)
                    print(
                        f"[EVENT] frame={frame_idx} bat-contact (est.) @ world=({wx},{wy})",
                        flush=True,
                    )

            # Reset bounce flash when idle
            if cur_state == STATE_IDLE:
                current_bounce_world = None

            prev_state = cur_state

            # --- Scoreboard OCR (Task 3.3) --- rate-limited, runs only when enabled
            scoreboard_reader.update(frame, frame_idx)

            # --- Speed-gun countdown ---
            if speed_display_left > 0:
                speed_display_left -= 1

            # --- Bounce flash on frame ---
            if ball_state and not ball_state[4]:  # actual detection, not predicted
                raw_cx, raw_cy = ball_state[0], ball_state[1]
                cv2.circle(frame, (int(raw_cx), int(raw_cy)), 8,
                           tuple(cfg["colors"]["ball_trail"]), -1, cv2.LINE_AA)

            # --- Draw ball trail (Task 2.6) ---
            draw_ball_trail(frame, traj.trail, tuple(cfg["colors"]["ball_trail"]))

            # --- Draw bounce flash circle (Task 2.8) ---
            if cur_state in (STATE_BOUNCED, STATE_COMPLETE) and current_bounce_world is not None:
                try:
                    bpx, bpy = mapper.to_pixel(*current_bounce_world)
                    cv2.circle(frame, (int(bpx), int(bpy)), 14,
                               tuple(cfg["colors"]["bounce_flash"]), 2, cv2.LINE_AA)
                    cv2.circle(frame, (int(bpx), int(bpy)), 6,
                               tuple(cfg["colors"]["bounce_flash"]), -1, cv2.LINE_AA)
                except Exception:
                    pass

            # --- HUD drawing ---
            draw_top_banner(frame, fsm.delivery_no, cur_state)

            # Pitch mini inset (Task 2.10 — live mini pitch map)
            draw_pitch_inset(
                frame, pixel_pts, traj.trail, current_bounce_world, mapper, cfg
            )

            # Speed gun (Task 2.10)
            draw_speed_gun(frame, last_speed_kmh, speed_display_left, speed_gun_color)

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"[DONE] Wrote {video_out}", flush=True)

    # ---- Post-run exports ----

    # Finalize any open delivery
    if fsm.state not in (STATE_IDLE, STATE_COMPLETE):
        fsm._finalize(frame_idx - 1)

    all_records = fsm.records

    # Task 2.11 — CSV
    csv_path = os.path.join(out_dir, "cricket_deliveries.csv")
    save_deliveries_csv(all_records, csv_path)

    # Task 2.12 — Pitch map PNG (highest priority)
    pitch_map_path = os.path.join(out_dir, "cricket_pitch_map.png")
    save_pitch_map(all_records, cfg, pitch_map_path)

    # Task 2.13 — Dashboard PNG
    dashboard_path = os.path.join(out_dir, "cricket_dashboard.png")
    save_dashboard(all_records, cfg, dashboard_path, pitch_map_path)

    # Task 3.2 — Wagon wheel PNG (only if enabled in config)
    if wagon_wheel_enabled:
        wagon_path = os.path.join(out_dir, "cricket_wagon_wheel.png")
        save_wagon_wheel(all_records, cfg, wagon_path)
    else:
        print("[INFO] Wagon wheel disabled (set wagon_wheel.enabled=true in config to enable)",
              flush=True)

    # Task 3.4 — cricket_events.json (only if scoreboard OCR enabled)
    if sb_cfg.get("enabled", False):
        events_path = os.path.join(out_dir, "cricket_events.json")
        save_cricket_events_json(video_path, scoreboard_reader.all_reads(), events_path)

    print(
        f"\n[DONE] Cricket analytics complete — {len(all_records)} deliveries processed.",
        flush=True,
    )
    return True


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args(argv):
    if len(argv) < 1:
        print("Usage: python cricket_analytics.py video.mp4 [--config config/cricket_config.yaml]")
        sys.exit(1)
    video_path = argv[0]
    config_path = None
    if "--config" in argv:
        idx = argv.index("--config")
        if idx + 1 >= len(argv):
            print("[ERROR] --config requires a path argument")
            sys.exit(1)
        config_path = argv[idx + 1]
    return video_path, config_path


if __name__ == "__main__":
    video_arg, config_arg = parse_args(sys.argv[1:])
    ok = process(video_arg, config_arg)
    sys.exit(0 if ok else 1)