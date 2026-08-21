import subprocess, sys, os

# pip package name -> actual importable module name
PKG_TO_MODULE = {
    "opencv-python": "cv2",
    "numpy":         "numpy",
    "matplotlib":    "matplotlib",
    "scipy":         "scipy",
    "ultralytics":   "ultralytics",
    "Pillow":        "PIL",
    "tqdm":          "tqdm",
    "lapx":          "lap",       # required by ultralytics' ByteTrack/BoT-SORT associator
    "PyYAML":        "yaml",      # config file loading (common/config_loader.py)
    "easyocr":       "easyocr",   # jersey number OCR (common/ocr.py) - CPU-only, no system Tesseract needed
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
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to install {pkg}: {e}", flush=True)
                sys.exit(1)
    print("[SETUP] All dependencies ready ✓", flush=True)

install_deps()

import cv2, numpy as np, warnings, math, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, deque
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO
from tqdm import tqdm
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.homography import PlanarMapper
from common.team_classifier import TeamClassifier, torso_patch, analyze_patch, is_referee
from common.draw_utils import txt, label_block, FONT
from common.video_io import make_writer
from common.config_loader import load_config
from common.ocr import read_jersey_number
from common.dashboard_utils import (
    make_dark_figure, apply_dark_theme, style_table,
    DARK_BG, DARK_EDGE, DARK_LABEL, ACCENT_CYAN, ACCENT_BLUE,
)
from common.detection import run_track

# ============================================================================
#  PITCH CALIBRATION
#  Football requires real-world coordinates to give accurate distance/speed
#  (a flat pixels-per-meter constant is wrong because of camera perspective).
#  This script uses a homography: 4 pitch-corner pixel points -> real meters.
#
#  IMPORTANT: this assumes a mostly STATIC wide/tactical camera (the whole
#  pitch visible, camera not panning/zooming). A free-panning broadcast feed
#  will make the homography drift and distances will be wrong. If your
#  footage pans a lot, treat the speed/distance numbers as rough estimates.
#
#  TO CALIBRATE: run once, open football_pitch_preview.png, and adjust the
#  4 points in config/football_config.yaml's calibration.pixel_corners_fallback
#  (in pixel coordinates) so the outline in the preview image sits exactly on
#  the four pitch corners: top-left, top-right, bottom-right, bottom-left (as
#  seen on screen, going clockwise from top-left).
# ============================================================================

# All tunable values below are placeholders populated from config/football_config.yaml
# at the start of process() (see RULES.md Section 2 - config-driven mandate).
PITCH_LENGTH_M = None
PITCH_WIDTH_M  = None
GOAL_WIDTH_M   = None
PENALTY_BOX_LEN_M  = None
PENALTY_BOX_WID_M  = None
CENTER_CIRCLE_R_M  = None

C_TEAM_A       = None
C_TEAM_B       = None
C_BALL         = None
C_WHITE        = (255, 255, 255)
C_GOAL_FLASH   = None

# Cosmetic fill/background shades derived from the team accent colors above -
# not part of SCHEMA.md's config (theming constants, not per-video-variable
# values per RULES.md Section 2), so these remain hardcoded.
C_TEAM_A_FILL  = (255, 255, 255)
C_TEAM_A_BG    = (50, 50, 70)
C_TEAM_B_FILL  = (25, 60, 200)
C_TEAM_B_BG    = (10, 20, 65)

LABEL_SCALE = 0.62
LABEL_THICK = 2


def team_color(team):
    if team == 'A': return C_TEAM_A
    if team == 'B': return C_TEAM_B
    return (120, 120, 120)


# ----------------------------------------------------------------------------
#  Ball tracking with short-gap prediction (keeps the trail alive through
#  brief occlusions / missed detections instead of just vanishing)
# ----------------------------------------------------------------------------
class BallTracker:
    def __init__(self, mapper, gap_predict_frames):
        self.mapper = mapper
        self.gap_predict_frames = gap_predict_frames
        self.trail_px = deque(maxlen=90)
        self.velocity_px = (0.0, 0.0)
        self.last_seen_frame = -999
        self.predicted = False

    def update(self, pos_px, frame_idx):
        if pos_px is not None:
            if self.trail_px:
                lp = self.trail_px[-1]
                self.velocity_px = (pos_px[0] - lp[0], pos_px[1] - lp[1])
            self.trail_px.append(pos_px)
            self.last_seen_frame = frame_idx
            self.predicted = False
            return pos_px
        gap = frame_idx - self.last_seen_frame
        if self.trail_px and gap <= self.gap_predict_frames:
            lp = self.trail_px[-1]
            pred = (lp[0] + self.velocity_px[0], lp[1] + self.velocity_px[1])
            self.trail_px.append(pred)
            self.predicted = True
            return pred
        return None

    def world_pos(self):
        if not self.trail_px:
            return None
        return self.mapper.to_world(*self.trail_px[-1])


# ----------------------------------------------------------------------------
#  Per-player world-space stats: distance, speed, sprints
# ----------------------------------------------------------------------------
class PlayerStats:
    def __init__(self, fps, max_realistic_speed_kmh, sprint_kmh, sprint_min_frames):
        self.fps = fps
        self.max_realistic_speed_kmh = max_realistic_speed_kmh
        self.sprint_kmh = sprint_kmh
        self.sprint_min_frames = sprint_min_frames
        self.world_pos = {}      # tid -> (x,y) last frame
        self.dist_m = defaultdict(float)
        self.speed_kmh = defaultdict(float)
        self.top_speed = defaultdict(float)
        self.sprint_run = defaultdict(int)
        self.sprint_count = defaultdict(int)
        self.smooth = defaultdict(lambda: deque(maxlen=6))

    def update(self, tid, wpos):
        if tid in self.world_pos:
            px, py = self.world_pos[tid]
            d = math.hypot(wpos[0]-px, wpos[1]-py)
            spd = min(d * self.fps * 3.6, self.max_realistic_speed_kmh)
            self.smooth[tid].append(spd)
            spd_s = float(np.mean(self.smooth[tid]))
            self.dist_m[tid] += d
            self.speed_kmh[tid] = spd_s
            self.top_speed[tid] = max(self.top_speed[tid], spd_s)
            if spd_s >= self.sprint_kmh:
                self.sprint_run[tid] += 1
                if self.sprint_run[tid] == self.sprint_min_frames:
                    self.sprint_count[tid] += 1
            else:
                self.sprint_run[tid] = 0
        self.world_pos[tid] = wpos


# ----------------------------------------------------------------------------
#  Jersey number resolution (TECHSPEC.md 5.3): rate-limited OCR per track,
#  majority vote across reads, displayed once ocr_min_votes consistent reads
#  are collected - avoids flashing a wrong number from one noisy read.
# ----------------------------------------------------------------------------
class JerseyResolver:
    def __init__(self, interval_frames, min_votes):
        self.interval_frames = interval_frames
        self.min_votes = min_votes
        self.votes = defaultdict(lambda: defaultdict(int))   # tid -> {number_str: count}
        self.resolved = {}                                    # tid -> number_str, once locked
        self.last_attempt_frame = defaultdict(lambda: -10**9)

    def maybe_read(self, tid, patch, frame_idx):
        """Call at most once per track per frame; internally rate-limited to
        once per interval_frames per track (RULES.md Section 9 - never OCR
        every frame). Returns the resolved number if/once locked, else None."""
        if tid in self.resolved:
            return self.resolved[tid]
        if frame_idx - self.last_attempt_frame[tid] < self.interval_frames:
            return None
        self.last_attempt_frame[tid] = frame_idx
        number = read_jersey_number(patch)
        if number:
            self.votes[tid][number] += 1
            best_num, best_count = max(self.votes[tid].items(), key=lambda kv: kv[1])
            if best_count >= self.min_votes:
                self.resolved[tid] = best_num
                return best_num
        return None

    def label_for(self, tid):
        return self.resolved.get(tid)


# ----------------------------------------------------------------------------
#  Possession + pass detection (world-space, team-aware)
# ----------------------------------------------------------------------------
class PossessionTracker:
    def __init__(self, possession_radius_m, pass_min_dist_m):
        self.possession_radius_m = possession_radius_m
        self.pass_min_dist_m = pass_min_dist_m
        self.total = defaultdict(int)
        self.last_holder = None
        self.last_holder_team = None
        self.last_holder_pos = None
        self.passes = defaultdict(int)     # team -> completed passes
        self.turnovers = 0
        self.pass_events = []       # dicts matching SCHEMA.md 2.2 "pass" events
        self.turnover_events = []   # dicts matching SCHEMA.md 2.2 "turnover" events

    def update(self, world_positions, teams, ball_world, frame_idx):
        """world_positions: {tid:(x,y)}  teams: {tid:'A'/'B'/None}"""
        if ball_world is None:
            return None
        best_tid, best_d = None, self.possession_radius_m
        for tid, wp in world_positions.items():
            d = math.hypot(wp[0]-ball_world[0], wp[1]-ball_world[1])
            if d < best_d:
                best_d = d; best_tid = tid
        if best_tid is None:
            return None
        team = teams.get(best_tid)
        if team in ('A', 'B'):
            self.total[team] += 1

        if self.last_holder is not None and best_tid != self.last_holder and self.last_holder_pos:
            travel = math.hypot(ball_world[0]-self.last_holder_pos[0], ball_world[1]-self.last_holder_pos[1])
            if travel >= self.pass_min_dist_m:
                if team == self.last_holder_team and team in ('A', 'B'):
                    self.passes[team] += 1
                    self.pass_events.append({
                        "type": "pass", "frame": frame_idx, "team": team,
                        "from_track_id": self.last_holder, "to_track_id": best_tid,
                        "distance_m": round(travel, 2),
                    })
                elif team in ('A', 'B') and self.last_holder_team in ('A', 'B') and team != self.last_holder_team:
                    self.turnovers += 1
                    self.turnover_events.append({
                        "type": "turnover", "frame": frame_idx,
                        "from_team": self.last_holder_team, "to_team": team,
                        "world_pos": [round(ball_world[0], 2), round(ball_world[1], 2)],
                    })

        self.last_holder = best_tid
        self.last_holder_team = team
        self.last_holder_pos = ball_world
        return best_tid

    def pct(self):
        tot = sum(self.total.values())
        if tot == 0: return 50.0, 50.0
        return (round(self.total.get('A', 0)/tot*100, 1),
                round(self.total.get('B', 0)/tot*100, 1))


# ----------------------------------------------------------------------------
#  Formation / shape estimate (TECHSPEC.md 5.4) - a rough heuristic, NOT true
#  tactical-line detection: bucket each team's players into thirds of the
#  pitch length (their own defense/mid/attack direction) and count per third.
#  Labeled "Estimated Shape" everywhere it's shown, never "Formation", to
#  avoid overclaiming accuracy (RULES.md Section 8 / DESIGN.md Section 1).
# ----------------------------------------------------------------------------
class FormationEstimator:
    def __init__(self, update_interval_frames, pitch_length_m):
        self.update_interval_frames = update_interval_frames
        self.pitch_length_m = pitch_length_m
        self.shape = {'A': None, 'B': None}
        self.last_update_frame = -10**9

    def maybe_update(self, world_positions, teams, frame_idx):
        if frame_idx - self.last_update_frame < self.update_interval_frames:
            return
        self.last_update_frame = frame_idx
        third = self.pitch_length_m / 3.0
        buckets = {'A': [0, 0, 0], 'B': [0, 0, 0]}
        for tid, wp in world_positions.items():
            team = teams.get(tid)
            if team not in ('A', 'B'):
                continue
            idx = min(2, max(0, int(wp[0] // third)))
            buckets[team][idx] += 1
        # Team A defends the x=0 end (attacks toward x=length; see GoalDetector),
        # so its own bucket order [near-0, mid, near-length] already reads as
        # defense->mid->attack. Team B defends the x=length end, so its buckets
        # are reversed to read in ITS OWN defense->mid->attack direction.
        if sum(buckets['A']) > 0:
            a = buckets['A']
            self.shape['A'] = f"{a[0]}-{a[1]}-{a[2]}"
        if sum(buckets['B']) > 0:
            b = list(reversed(buckets['B']))
            self.shape['B'] = f"{b[0]}-{b[1]}-{b[2]}"

    def summary_pair(self):
        a = self.shape['A'] or "?"
        b = self.shape['B'] or "?"
        return a, b

    def hud_line(self):
        if not self.shape['A'] and not self.shape['B']:
            return None
        a, b = self.summary_pair()
        return f"Estimated Shape  A: {a}    B: {b}"


# ----------------------------------------------------------------------------
#  Goal detection (ball crosses either goal line within the goal mouth width).
#  World-coordinate-correct: works off the live, auto-recalibrated `mapper`
#  (Tasks 1.1/1.2), not a one-shot homography, so a pan/zoom mid-match that
#  triggers a re-calibration doesn't silently corrupt where "the goal line" is.
# ----------------------------------------------------------------------------
class GoalDetector:
    def __init__(self, pitch_length_m, pitch_width_m, goal_width_m, cooldown_frames):
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.goal_width_m = goal_width_m
        self.cooldown_frames = cooldown_frames
        self.last_goal_frame = -99999
        self.goals = {'A': 0, 'B': 0}

    def check(self, ball_world, frame_idx, scorer_track_id=None):
        """Returns an event dict matching SCHEMA.md 2.2's "goal" shape, or None."""
        if ball_world is None:
            return None
        if frame_idx - self.last_goal_frame < self.cooldown_frames:
            return None
        x, y = ball_world
        y_lo = self.pitch_width_m/2 - self.goal_width_m/2
        y_hi = self.pitch_width_m/2 + self.goal_width_m/2
        if not (y_lo <= y <= y_hi):
            return None
        team = None
        if x <= 0.3:
            team = 'B'   # team attacking toward x=0 scored
        elif x >= self.pitch_length_m - 0.3:
            team = 'A'
        if team is None:
            return None
        self.last_goal_frame = frame_idx
        self.goals[team] += 1
        return {
            "type": "goal", "frame": frame_idx, "team": team,
            "track_id": scorer_track_id,
            "world_pos": [round(x, 2), round(y, 2)],
        }


# ----------------------------------------------------------------------------
#  Shot detection + xG(est.) (TECHSPEC.md 5.2). A "shot" fires when the ball's
#  world-space velocity exceeds a threshold, its direction points at a goal
#  mouth within an angle tolerance, and it's within range of that goal.
# ----------------------------------------------------------------------------
def estimate_xg(dist_to_goal_m, angle_diff_deg):
    """Rough xG (est.) approximation: logistic in (distance to goal, off-center
    angle) only. NOT a broadcast-grade model - no defender pressure, shot body
    part, or assist type considered. Always label "xG (est.)" wherever shown,
    never bare "xG" (RULES.md Section 8 / DESIGN.md Section 1)."""
    z = 1.2 - 0.14 * dist_to_goal_m - 0.02 * angle_diff_deg
    xg = 1.0 / (1.0 + math.exp(-z))
    return round(min(max(xg, 0.01), 0.95), 3)


class ShotDetector:
    def __init__(self, velocity_threshold_ms, angle_tolerance_deg, max_distance_m,
                 pitch_length_m, pitch_width_m, goal_width_m, fps, cooldown_frames=30):
        self.velocity_threshold_ms = velocity_threshold_ms
        self.angle_tolerance_deg = angle_tolerance_deg
        self.max_distance_m = max_distance_m
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.goal_width_m = goal_width_m
        self.fps = fps
        self.cooldown_frames = cooldown_frames
        self.last_shot_frame = -99999
        self.prev_ball_world = None
        self.prev_frame_idx = None

    def update(self, ball_world, frame_idx, holder_tid, holder_team):
        """Returns a shot event dict (SCHEMA.md 2.2) or None. Call once per frame
        regardless of whether the ball was seen this frame."""
        event = None
        if (ball_world is not None and self.prev_ball_world is not None
                and holder_team in ('A', 'B')
                and frame_idx - self.last_shot_frame >= self.cooldown_frames):
            dt = (frame_idx - self.prev_frame_idx) / self.fps if self.prev_frame_idx is not None else 0
            if dt > 0:
                dx = ball_world[0] - self.prev_ball_world[0]
                dy = ball_world[1] - self.prev_ball_world[1]
                travel = math.hypot(dx, dy)
                velocity_ms = travel / dt
                if velocity_ms >= self.velocity_threshold_ms and travel > 1e-6:
                    goal_x = self.pitch_length_m if holder_team == 'A' else 0.0
                    goal_y = self.pitch_width_m / 2
                    to_goal_x, to_goal_y = goal_x - ball_world[0], goal_y - ball_world[1]
                    dist_to_goal = math.hypot(to_goal_x, to_goal_y)
                    if dist_to_goal <= self.max_distance_m:
                        shot_ang = math.degrees(math.atan2(dy, dx))
                        goal_ang = math.degrees(math.atan2(to_goal_y, to_goal_x))
                        angle_diff = abs((shot_ang - goal_ang + 180) % 360 - 180)
                        if angle_diff <= self.angle_tolerance_deg:
                            # Project the current trajectory forward to the goal line
                            # to estimate whether it was heading on-target.
                            t = (goal_x - ball_world[0]) / dx if dx != 0 else 0
                            proj_y = ball_world[1] + dy * t
                            y_lo = self.pitch_width_m/2 - self.goal_width_m/2
                            y_hi = self.pitch_width_m/2 + self.goal_width_m/2
                            on_target = y_lo <= proj_y <= y_hi
                            self.last_shot_frame = frame_idx
                            event = {
                                "type": "shot", "frame": frame_idx, "team": holder_team,
                                "track_id": holder_tid,
                                "world_pos": [round(ball_world[0], 2), round(ball_world[1], 2)],
                                "velocity_ms": round(velocity_ms, 2),
                                "xg_estimate": estimate_xg(dist_to_goal, angle_diff),
                                "on_target": on_target,
                            }
        self.prev_ball_world = ball_world
        self.prev_frame_idx = frame_idx
        return event


# ----------------------------------------------------------------------------
#  Drawing helpers (football-specific HUD composition; shared primitives -
#  txt/label_block - come from common/draw_utils)
# ----------------------------------------------------------------------------
def draw_player(canvas, box, team, tid, spd, tc, jersey_number=None):
    x1, y1, x2, y2 = box
    color = team_color(team)
    fill = C_TEAM_A_FILL if team == 'A' else (C_TEAM_B_FILL if team == 'B' else (110,110,110))
    ov = canvas.copy()
    iy2 = int(y1 + (y2-y1)*0.82)
    cv2.rectangle(ov, (x1+3, y1+3), (x2-3, iy2), fill, -1)
    cv2.addWeighted(ov, 0.22, canvas, 0.78, 0, canvas)
    cx = (x1+x2)//2
    rx = max((x2-x1)//2 + 10, 20); ry = max(int(rx*0.32), 7)
    cv2.ellipse(canvas, (cx, y2), (rx, ry), 0, 0, 360, color, 2, cv2.LINE_AA)
    bg = C_TEAM_A_BG if team == 'A' else (C_TEAM_B_BG if team == 'B' else (30,30,30))
    # Once OCR resolves a confident jersey number (TECHSPEC.md 5.3), the label
    # switches from the anonymous track id to it - track id remains internal only.
    id_label = f"#{jersey_number}" if jersey_number else f"#{tid}"
    label_block(canvas, [f"{id_label}  {spd:.1f} km/h"], [C_WHITE], cx, y2+ry+8, bg)


def draw_ball(canvas, ball):
    if not ball.trail_px:
        return
    pts = list(ball.trail_px)[-40:]
    n = len(pts)
    for i in range(1, n):
        a = i/n
        col = tuple(int(c*a) for c in C_BALL)
        cv2.line(canvas, tuple(map(int, pts[i-1])), tuple(map(int, pts[i])), col, 2, cv2.LINE_AA)
    bx, by = map(int, pts[-1])
    for r, a in [(15,35),(10,80),(5,190)]:
        ov = canvas.copy()
        cv2.circle(ov, (bx,by), r, C_BALL, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, a/255.0, canvas, 1-a/255.0, 0, canvas)
    ring = (120,120,255) if ball.predicted else C_WHITE
    cv2.circle(canvas, (bx,by), 6, ring, 2, cv2.LINE_AA)


def draw_minimap(canvas, mapper, world_positions, teams, tc, ball, W, H, mw=260, mh=180):
    mx, my = W-mw-16, H-mh-70
    ov = canvas.copy()
    cv2.rectangle(ov, (mx,my), (mx+mw,my+mh), (10,60,15), -1)
    cv2.rectangle(ov, (mx,my), (mx+mw,my+mh), (255,255,255), 2)
    cv2.line(ov, (mx+mw//2,my), (mx+mw//2,my+mh), (255,255,255), 1)
    cv2.circle(ov, (mx+mw//2,my+mh//2), int(CENTER_CIRCLE_R_M/PITCH_WIDTH_M*mh), (255,255,255), 1)
    cv2.addWeighted(ov, 0.55, canvas, 0.45, 0, canvas)

    def w2m(wx, wy):
        px = mx + int((wx/PITCH_LENGTH_M)*mw)
        py = my + int((wy/PITCH_WIDTH_M)*mh)
        return np.clip(px, mx+2, mx+mw-2), np.clip(py, my+2, my+mh-2)

    for tid, wp in world_positions.items():
        px, py = w2m(*wp)
        col = team_color(teams.get(tid))
        cv2.circle(canvas, (px,py), 4, col, -1, cv2.LINE_AA)
    bw = ball.world_pos() if ball else None
    if bw:
        px, py = w2m(*bw)
        cv2.circle(canvas, (px,py), 4, C_BALL, -1, cv2.LINE_AA)
    txt(canvas, "PITCH MAP", mx+4, my-6, 0.42, (170,220,180), 1)


def draw_bottom_hud(canvas, pa, pb, avg_s, n_players, passes_a, passes_b, W, H):
    bar_h = 56; y0 = H-bar_h-22
    ov = canvas.copy()
    cv2.rectangle(ov, (0,y0), (W,y0+bar_h), (8,12,20), -1)
    cv2.addWeighted(ov, 0.80, canvas, 0.20, 0, canvas)
    cv2.line(canvas, (0,y0), (W,y0), (50,50,80), 1)

    bw, bx, by, bh2 = 300, W//2-150, y0+8, 16
    af = int(bw*pa/100)
    cv2.rectangle(canvas, (bx,by), (bx+bw,by+bh2), (40,40,40), -1)
    if af>0: cv2.rectangle(canvas, (bx,by), (bx+af,by+bh2), C_TEAM_A, -1)
    if af<bw: cv2.rectangle(canvas, (bx+af,by), (bx+bw,by+bh2), C_TEAM_B, -1)
    cv2.rectangle(canvas, (bx,by), (bx+bw,by+bh2), (80,80,100), 1)
    pt_ = f"A {pa}%  |  POSSESSION  |  {pb}% B"
    (pw,_),_ = cv2.getTextSize(pt_, FONT, 0.56, 1)
    txt(canvas, pt_, W//2-pw//2, by+bh2+20, 0.56, C_WHITE, 1)

    s1 = f"AVG SPD {avg_s:.1f} km/h"
    (sw,_),_ = cv2.getTextSize(s1, FONT, 0.66, 2)
    txt(canvas, s1, bx-sw-40, y0+bar_h//2+10, 0.66, (0,220,130), 2)
    txt(canvas, f"PLAYERS {n_players}   PASSES A:{passes_a} B:{passes_b}",
        bx+bw+20, y0+bar_h//2+10, 0.60, (0,200,255), 2)


def draw_top_banner(canvas, W, goals_a, goals_b):
    title = f"FOOTBALL MATCH ANALYTICS  |  A {goals_a} - {goals_b} B"
    (tw, th), _ = cv2.getTextSize(title, FONT, 0.8, 2)
    p = 12
    cv2.rectangle(canvas, (0,0), (tw+p*2, th+p*2), (90,10,10), -1)
    cv2.putText(canvas, title, (p, th+p), FONT, 0.8, C_WHITE, 2, cv2.LINE_AA)
    return th + p*2  # bottom y of the banner, so callers can stack below it


def draw_formation_hud(canvas, formation, banner_bottom_y):
    """Small, low-weight "Estimated Shape" line (TECHSPEC.md 5.4 / DESIGN.md
    Section 1 principle 4 - clearly marked as an estimate, never "Formation")."""
    line = formation.hud_line()
    if not line:
        return
    txt(canvas, line, 12, banner_bottom_y + 18, 0.5, (170, 170, 170), 1)


def draw_goal_flash(canvas, team, W, H):
    ov = canvas.copy()
    cv2.rectangle(ov, (0,0), (W,H), C_GOAL_FLASH, -1)
    cv2.addWeighted(ov, 0.18, canvas, 0.82, 0, canvas)
    text = f"GOAL!  TEAM {team}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 2.2, 5)
    txt(canvas, text, W//2-tw//2, H//2, 2.2, C_GOAL_FLASH, 5)


# ----------------------------------------------------------------------------
#  Pitch-line auto-detection (TECHSPEC.md Section 5.1): HSV green mask ->
#  Canny -> Hough line transform -> intersect boundary lines -> 4 corners,
#  plus (if visible) the halfway line as an independent confidence check
#  point for PlanarMapper.re_estimate's reprojection-error scoring.
# ----------------------------------------------------------------------------
def _line_angle_deg(x1, y1, x2, y2):
    return abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180


def _line_intersect(l1, l2):
    """Infinite-line intersection (ax+by=c form); None if near-parallel."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    a1, b1, c1 = (y2 - y1), (x1 - x2), (y2 - y1) * x1 + (x1 - x2) * y1
    a2, b2, c2 = (y4 - y3), (x3 - x4), (y4 - y3) * x3 + (x3 - x4) * y3
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    return ((c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det)


def _fit_representative_line(segs):
    """Average a cluster of near-parallel Hough segments into one representative
    infinite line by averaging endpoints (simple, adequate at this line count)."""
    xs1 = np.mean([s[0] for s in segs]); ys1 = np.mean([s[1] for s in segs])
    xs2 = np.mean([s[2] for s in segs]); ys2 = np.mean([s[3] for s in segs])
    return (xs1, ys1, xs2, ys2)


def detect_pitch_corners(frame, cfg, margin_frac=0.15):
    """Attempt to auto-detect the 4 pitch boundary corners from visible pitch
    lines. Returns (corners, check_pixel_pts, check_world_pts) on a plausible
    detection, or (None, None, None) if no usable pitch boundary is found this
    frame (e.g. a tight crop with no lines visible, or a non-green surface) -
    this is a normal, expected outcome, not an error; the caller falls back
    to the last known-good homography or the config seed (TECHSPEC.md 5.1)."""
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_lo, hue_hi = cfg["calibration"]["pitch_hue_range"]
    min_sat = cfg["calibration"]["pitch_min_saturation"]
    mask = cv2.inRange(hsv, (hue_lo, min_sat, 30), (hue_hi, 255, 255))
    # NOTE: deliberately no morphological closing here - pitch lines are thin
    # gaps *within* the green mask (this is how they're found at all: Canny
    # picks up the mask boundary around each line); closing with a kernel wider
    # than the line stroke erases the very feature we're trying to detect.

    edges = cv2.Canny(mask, 50, 150)
    min_line_len = int(min(W, H) * 0.25)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=min_line_len, maxLineGap=20)
    if lines is None or len(lines) < 4:
        return None, None, None
    lines = lines.reshape(-1, 4)  # OpenCV returns (N,1,4) on some builds, (N,4) on others

    horiz, vert = [], []
    for l in lines:
        x1, y1, x2, y2 = [float(v) for v in l]
        ang = _line_angle_deg(x1, y1, x2, y2)
        length = math.hypot(x2 - x1, y2 - y1)
        if ang < 30 or ang > 150:
            horiz.append((length, (x1, y1, x2, y2)))
        elif 60 < ang < 120:
            vert.append((length, (x1, y1, x2, y2)))
    if len(horiz) < 2 or len(vert) < 2:
        return None, None, None

    # Top/bottom touchlines: cluster horiz lines by mean-y, take the extremes.
    horiz.sort(key=lambda t: (t[1][1] + t[1][3]) / 2)
    top_cluster = [s for _, s in horiz[:max(1, len(horiz) // 3)]]
    bottom_cluster = [s for _, s in horiz[-max(1, len(horiz) // 3):]]
    top_line = _fit_representative_line(top_cluster)
    bottom_line = _fit_representative_line(bottom_cluster)
    if bottom_line[1] - top_line[1] < H * 0.15:
        return None, None, None  # touchlines too close together to be real

    # Left/right boundary lines: cluster vert lines by mean-x, take the extremes.
    vert.sort(key=lambda t: (t[1][0] + t[1][2]) / 2)
    left_cluster = [s for _, s in vert[:max(1, len(vert) // 3)]]
    right_cluster = [s for _, s in vert[-max(1, len(vert) // 3):]]
    left_line = _fit_representative_line(left_cluster)
    right_line = _fit_representative_line(right_cluster)
    if right_line[0] - left_line[0] < W * 0.15:
        return None, None, None  # boundary lines too close together to be real

    tl = _line_intersect(top_line, left_line)
    tr = _line_intersect(top_line, right_line)
    br = _line_intersect(bottom_line, right_line)
    bl = _line_intersect(bottom_line, left_line)
    if None in (tl, tr, br, bl):
        return None, None, None

    corners = [tl, tr, br, bl]
    mx, my = W * margin_frac, H * margin_frac
    for (cx, cy) in corners:
        if not (-mx <= cx <= W + mx and -my <= cy <= H + my):
            return None, None, None  # corner projects wildly outside the frame

    area = abs(cv2.contourArea(np.array(corners, dtype=np.float32)))
    if area < 0.10 * W * H:
        return None, None, None  # implausibly small quadrilateral

    # Halfway line (optional): a vertical-ish long line strictly between the
    # left/right boundary lines' x-position, not itself part of either cluster.
    boundary_segs = set(left_cluster + right_cluster)
    mid_lo_x, mid_hi_x = W * 0.35, W * 0.65
    halfway_candidates = [
        (length, s) for length, s in vert
        if s not in boundary_segs and mid_lo_x <= (s[0] + s[2]) / 2 <= mid_hi_x
    ]
    check_pixel_pts, check_world_pts = None, None
    if halfway_candidates:
        halfway_candidates.sort(key=lambda t: -t[0])
        halfway_line = halfway_candidates[0][1]
        hp_top = _line_intersect(top_line, halfway_line)
        hp_bottom = _line_intersect(bottom_line, halfway_line)
        if hp_top and hp_bottom:
            pitch_len = cfg["pitch"]["length_m"]
            pitch_wid = cfg["pitch"]["width_m"]
            check_pixel_pts = [hp_top, hp_bottom]
            check_world_pts = [(pitch_len / 2, 0.0), (pitch_len / 2, pitch_wid)]

    return corners, check_pixel_pts, check_world_pts


def draw_pitch_preview(frame, pixel_corners):
    pv = frame.copy()
    pts = np.array(pixel_corners, dtype=np.int32)
    cv2.polylines(pv, [pts], True, (0,255,255), 2, cv2.LINE_AA)
    for i,(x,y) in enumerate(pixel_corners):
        cv2.circle(pv, (int(x),int(y)), 6, (0,0,255), -1)
        txt(pv, str(i), int(x)+8, int(y)-8, 0.6, C_WHITE, 2)
    return pv


# ----------------------------------------------------------------------------
#  Dashboards (proper pitch-shaped heatmap + summary charts)
# ----------------------------------------------------------------------------
def draw_pitch_outline(ax):
    ax.set_facecolor("#0b3d17")
    ax.add_patch(mpatches.Rectangle((0,0), PITCH_LENGTH_M, PITCH_WIDTH_M, fill=False, color="white", lw=1.5))
    ax.axvline(PITCH_LENGTH_M/2, color="white", lw=1)
    ax.add_patch(mpatches.Circle((PITCH_LENGTH_M/2, PITCH_WIDTH_M/2), CENTER_CIRCLE_R_M, fill=False, color="white", lw=1))
    for x0 in (0, PITCH_LENGTH_M-PENALTY_BOX_LEN_M):
        y0 = (PITCH_WIDTH_M-PENALTY_BOX_WID_M)/2
        ax.add_patch(mpatches.Rectangle((x0,y0), PENALTY_BOX_LEN_M, PENALTY_BOX_WID_M, fill=False, color="white", lw=1))
    ax.set_xlim(-2, PITCH_LENGTH_M+2); ax.set_ylim(-2, PITCH_WIDTH_M+2)
    ax.invert_yaxis(); ax.axis("off")


def save_heatmap(pos_a, pos_b, path):
    fig, axes = make_dark_figure(1, 2, figsize=(16, 7))
    apply_dark_theme(axes)
    for ax, pos, title in zip(axes, [pos_a, pos_b], ["TEAM A — HEATMAP", "TEAM B — HEATMAP"]):
        draw_pitch_outline(ax)
        if pos:
            xs = np.array([p[0] for p in pos]); ys = np.array([p[1] for p in pos])
            hm = np.zeros((68,105), dtype=np.float32)
            for x,y in zip(xs,ys):
                xi, yi = int(np.clip(x,0,104)), int(np.clip(y,0,67))
                hm[yi,xi] += 1
            hm = gaussian_filter(hm, sigma=2.2)
            if hm.max()>0: hm/=hm.max()
            ax.imshow(hm, cmap="inferno", alpha=0.75, extent=[0,PITCH_LENGTH_M,PITCH_WIDTH_M,0])
        ax.set_title(title, color="#00e0ff", fontsize=13, fontweight="bold", pad=10)
    fig.suptitle("FOOTBALL MATCH ANALYTICS — HEATMAPS | dev: abhinav.phi", color="#00e0ff", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(); print(f"[DONE] {path} ✓")


def save_pitch_map(events, path):
    fig, ax = make_dark_figure(figsize=(11, 7))
    apply_dark_theme(ax)
    draw_pitch_outline(ax)
    ax.set_title("Shot Map - xG (est.)", color=ACCENT_BLUE, fontsize=13, fontweight="bold", pad=12)

    shot_count = 0
    goal_count = 0
    for event in events:
        etype = event.get("type")
        if etype == "shot":
            wx, wy = event["world_pos"]
            xg = float(event.get("xg_estimate", 0.0))
            on_target = bool(event.get("on_target", False))
            color = plt.cm.inferno(float(np.clip(xg, 0.0, 1.0)))
            edge = "#00ff5a" if on_target else "#ff4d4d"
            marker = "o" if on_target else "X"
            size = 80 + 420 * float(np.clip(xg, 0.0, 1.0))
            ax.scatter(wx, wy, s=size, marker=marker, c=[color], edgecolors=edge,
                       linewidths=1.8, alpha=0.9, zorder=5)
            ax.text(wx + 1.2, wy - 1.2, f"{xg:.2f}", color="#deeeff", fontsize=8,
                    bbox={"boxstyle": "round,pad=0.18", "facecolor": "#141428", "edgecolor": edge, "alpha": 0.78},
                    zorder=6)
            shot_count += 1
        elif etype == "goal":
            wx, wy = event["world_pos"]
            ax.scatter(wx, wy, s=320, marker="*", c="#00ff5a", edgecolors="white",
                       linewidths=1.4, alpha=0.95, zorder=7)
            goal_count += 1

    if shot_count == 0 and goal_count == 0:
        ax.text(PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2, "No shots detected",
                ha="center", va="center", color="#8aa4c0", fontsize=13, fontweight="bold")
    else:
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#fdbb2d",
                       markeredgecolor="#00ff5a", markersize=9, label="On target"),
            plt.Line2D([0], [0], marker="X", color="none", markerfacecolor="#fdbb2d",
                       markeredgecolor="#ff4d4d", markersize=9, label="Off target"),
            plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#00ff5a",
                       markeredgecolor="white", markersize=13, label="Goal"),
        ]
        leg = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
                        ncol=3, frameon=True, facecolor="#101a2c", edgecolor="#2a3a5a")
        for txt_obj in leg.get_texts():
            txt_obj.set_color("#deeeff")

    fig.suptitle("FOOTBALL MATCH ANALYTICS - PITCH MAP | dev: abhinav.phi",
                 color=ACCENT_CYAN, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(); print(f"[DONE] {path} saved")

def save_dashboard(hist_spd, poss, ps, goals, passes, turnovers, frame_idx, path, formation=None, events=None):
    fig, axes = make_dark_figure(2, 2, figsize=(14, 8))
    apply_dark_theme(axes)
    fig.suptitle("FOOTBALL MATCH ANALYTICS — REPORT | dev: abhinav.phi", color=ACCENT_CYAN, fontsize=14, fontweight="bold", y=0.98)

    fx = np.arange(len(hist_spd))
    axes[0,0].fill_between(fx, hist_spd, alpha=0.35, color="#00dc64")
    axes[0,0].plot(fx, hist_spd, color="#00dc64", lw=1.2)
    axes[0,0].set_title("Avg Player Speed (km/h)", color=ACCENT_BLUE, fontsize=11)
    axes[0,0].tick_params(colors=DARK_LABEL)

    pa, pb = poss.pct()
    axes[0,1].pie([pa,pb], labels=[f"Team A\n{pa}%", f"Team B\n{pb}%"],
                  colors=[(0.86,0.86,0.86),(0.16,0.35,0.92)],
                  textprops={"color":"#e0eeff","fontsize":11},
                  wedgeprops={"edgecolor":"#080c12","linewidth":2}, startangle=90)
    axes[0,1].set_title("Possession %", color=ACCENT_BLUE, fontsize=11)

    tids = list(ps.top_speed.keys())
    top = sorted(tids, key=lambda t: ps.top_speed[t], reverse=True)[:8]
    if top:
        axes[1,0].barh([f"#{t}" for t in top][::-1], [ps.top_speed[t] for t in top][::-1], color="#ffa000")
    axes[1,0].set_title("Top Speed Leaderboard (km/h)", color=ACCENT_BLUE, fontsize=11)
    axes[1,0].tick_params(colors=DARK_LABEL)

    axes[1,1].axis("off")
    total_dist = sum(ps.dist_m.values())
    total_sprints = sum(ps.sprint_count.values())
    shot_counts = {"A": 0, "B": 0}
    xg_totals = {"A": 0.0, "B": 0.0}
    for event in events or []:
        if event.get("type") == "shot" and event.get("team") in shot_counts:
            team = event["team"]
            shot_counts[team] += 1
            xg_totals[team] += float(event.get("xg_estimate", 0.0))
    rows = [
        ["Frames Processed", str(frame_idx)],
        ["Goals  A : B", f"{goals['A']} : {goals['B']}"],
        ["Shots  A : B", f"{shot_counts['A']} : {shot_counts['B']}"],
        ["xG (est.) A : B", f"{xg_totals['A']:.2f} : {xg_totals['B']:.2f}"],
        ["Possession  A : B", f"{pa}% : {pb}%"],
    ]
    if formation is not None:
        shape_a, shape_b = formation.summary_pair()
        rows.append(["Estimated Shape A : B", f"{shape_a} : {shape_b}"])
    rows.extend([
        ["Completed Passes A : B", f"{passes.get('A',0)} : {passes.get('B',0)}"],
        ["Turnovers", str(turnovers)],
        ["Total Distance (all players)", f"{total_dist:.0f} m"],
        ["Total Sprints (>%.0f km/h)"%ps.sprint_kmh, str(total_sprints)],
    ])
    tbl = axes[1,1].table(cellText=rows, colLabels=["Metric","Value"], loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.5)
    style_table(tbl)
    axes[1,1].set_title("Match Summary", color=ACCENT_BLUE, fontsize=11)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(); print(f"[DONE] {path} ✓")


# ----------------------------------------------------------------------------
#  Main pipeline
# ----------------------------------------------------------------------------
def process(video_path, config_path=None):
    global PITCH_LENGTH_M, PITCH_WIDTH_M, GOAL_WIDTH_M, PENALTY_BOX_LEN_M, \
        PENALTY_BOX_WID_M, CENTER_CIRCLE_R_M, C_TEAM_A, C_TEAM_B, C_BALL, C_GOAL_FLASH

    print("╔" + "═"*48 + "╗", flush=True)
    print("║   FOOTBALL MATCH ANALYTICS  v1.0            ║", flush=True)
    print("║   dev: abhinav.phi                          ║", flush=True)
    print("╚" + "═"*48 + "╝", flush=True)

    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found: {video_path}", flush=True)
        return

    cfg = load_config("football", path=config_path)

    PITCH_LENGTH_M = cfg["pitch"]["length_m"]
    PITCH_WIDTH_M  = cfg["pitch"]["width_m"]
    GOAL_WIDTH_M   = cfg["pitch"]["goal_width_m"]
    PENALTY_BOX_LEN_M = cfg["pitch"]["penalty_box_length_m"]
    PENALTY_BOX_WID_M = cfg["pitch"]["penalty_box_width_m"]
    CENTER_CIRCLE_R_M = cfg["pitch"]["center_circle_radius_m"]

    C_TEAM_A = tuple(cfg["colors"]["team_a"])
    C_TEAM_B = tuple(cfg["colors"]["team_b"])
    C_BALL   = tuple(cfg["colors"]["ball"])
    C_GOAL_FLASH = tuple(cfg["colors"]["goal_flash"])

    pitch_pixel_corners = [tuple(pt) for pt in cfg["calibration"]["pixel_corners_fallback"]]
    pitch_world_corners = [(0.0, 0.0), (PITCH_LENGTH_M, 0.0),
                            (PITCH_LENGTH_M, PITCH_WIDTH_M), (0.0, PITCH_WIDTH_M)]

    min_calib_confidence = cfg["calibration"]["min_calibration_confidence"]
    recalibration_interval = cfg["calibration"]["recalibration_interval_frames"]

    det_conf = cfg["detection"]["det_conf_person"]
    det_conf_ball = cfg["detection"]["det_conf_ball"]   # BUG FIX: was defined in config but never consumed
    input_size = cfg["detection"]["input_size"]

    print("[YOLO] Loading detection model (yolov8n, first run downloads weights)...", flush=True)
    model = YOLO("yolov8n.pt")
    print("[YOLO] loaded ✓  |  tracker: ByteTrack (persistent IDs)", flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}", flush=True)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {W}x{H} @ {fps:.1f}fps | {total} frames", flush=True)
    print("[NOTE] Homography assumes a mostly static wide/tactical camera. "
          "Heavy panning will make distance/speed numbers unreliable.", flush=True)

    out_dir = os.path.dirname(os.path.abspath(video_path))
    ret0, frame0 = cap.read()
    if ret0:
        # Initial auto-calibration attempt (TECHSPEC.md 5.1 / APPFLOW.md 5): try
        # pitch-line detection first; only ever fall back to the config-seeded
        # corners if detection fails or its confidence is below threshold. This
        # fallback path must never be removed - it's the guaranteed last resort.
        corners0, chk_px0, chk_wd0 = detect_pitch_corners(frame0, cfg)
        if corners0 is not None:
            candidate0, conf0 = PlanarMapper.re_estimate(
                corners0, pitch_world_corners, chk_px0, chk_wd0)
            if conf0 >= min_calib_confidence:
                pitch_pixel_corners = corners0
                print(f"[HOMOGRAPHY] frame 0: auto-calibration accepted "
                      f"(confidence={conf0:.2f} >= {min_calib_confidence:.2f})", flush=True)
            else:
                print(f"[HOMOGRAPHY] frame 0: auto-calibration confidence too low "
                      f"({conf0:.2f} < {min_calib_confidence:.2f}), using config fallback corners", flush=True)
        else:
            print("[HOMOGRAPHY] frame 0: no pitch lines detected, using config fallback corners", flush=True)

        cv2.imwrite(os.path.join(out_dir, "football_pitch_preview.png"),
                    draw_pitch_preview(frame0, pitch_pixel_corners))
        print("[CALIB] Saved football_pitch_preview.png — check the yellow outline "
              "matches the 4 pitch corners; edit config/football_config.yaml's "
              "calibration.pixel_corners_fallback if not.", flush=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    mapper = PlanarMapper(pitch_pixel_corners, pitch_world_corners)
    out_path = os.path.join(out_dir, "football_output.mp4")
    writer = make_writer(out_path, fps, W, H)

    tc = TeamClassifier(
        calibration_frames=cfg["team_classification"]["calibration_frames"],
        min_calib_samples=cfg["team_classification"]["min_calib_samples"],
        min_sat_for_cluster=cfg["team_classification"]["min_sat_for_cluster"],
        vote_buffer_len=cfg["team_classification"]["vote_buffer_len"],
        vote_lock_thresh=cfg["team_classification"]["vote_lock_thresh"],
    )
    ps = PlayerStats(
        fps,
        max_realistic_speed_kmh=cfg["ball"]["max_realistic_speed_kmh"],
        sprint_kmh=cfg["sprint"]["sprint_kmh"],
        sprint_min_frames=cfg["sprint"]["sprint_min_frames"],
    )
    ball = BallTracker(mapper, gap_predict_frames=cfg["ball"]["gap_predict_frames"])
    poss = PossessionTracker(
        possession_radius_m=cfg["possession"]["radius_m"],
        pass_min_dist_m=cfg["possession"]["pass_min_distance_m"],
    )
    goal_det = GoalDetector(
        pitch_length_m=PITCH_LENGTH_M,
        pitch_width_m=PITCH_WIDTH_M,
        goal_width_m=GOAL_WIDTH_M,
        cooldown_frames=cfg["goal_detection"]["cooldown_frames"],
    )
    shot_det = ShotDetector(
        velocity_threshold_ms=cfg["shots"]["velocity_threshold_ms"],
        angle_tolerance_deg=cfg["shots"]["angle_tolerance_deg"],
        max_distance_m=cfg["shots"]["max_distance_m"],
        pitch_length_m=PITCH_LENGTH_M,
        pitch_width_m=PITCH_WIDTH_M,
        goal_width_m=GOAL_WIDTH_M,
        fps=fps,
    )
    jersey_ocr_enabled = cfg["ocr"]["enabled"]
    jersey = JerseyResolver(
        interval_frames=cfg["ocr"]["interval_frames"],
        min_votes=cfg["ocr"]["min_votes"],
    )
    formation_enabled = cfg["formation"]["enabled"]
    formation = FormationEstimator(
        update_interval_frames=cfg["formation"]["update_interval_frames"],
        pitch_length_m=PITCH_LENGTH_M,
    )

    referee_max_mean_sat = cfg["team_classification"]["referee_max_mean_sat"]
    referee_min_contrast = cfg["team_classification"]["referee_min_contrast"]

    pos_a, pos_b = [], []
    hist_spd = []
    events = []
    goal_flash_frames = 0
    goal_flash_team = None
    frame_idx = 0

    print("[PROC] Processing...", flush=True)
    with tqdm(total=total, unit="fr", ncols=80, colour="cyan") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            canvas = frame.copy()

            # Rolling re-calibration (TECHSPEC.md 5.1 / 1.2): re-attempt pitch-line
            # detection every recalibration_interval_frames. Never silently apply a
            # low-confidence homography - accept/reject/keep-previous, always logged.
            if frame_idx > 0 and frame_idx % recalibration_interval == 0:
                corners_i, chk_px_i, chk_wd_i = detect_pitch_corners(frame, cfg)
                if corners_i is not None:
                    candidate_i, conf_i = PlanarMapper.re_estimate(
                        corners_i, pitch_world_corners, chk_px_i, chk_wd_i)
                    if conf_i >= min_calib_confidence:
                        mapper = candidate_i
                        ball.mapper = mapper
                        print(f"[HOMOGRAPHY] frame {frame_idx}: re-calibration accepted "
                              f"(confidence={conf_i:.2f} >= {min_calib_confidence:.2f})", flush=True)
                    else:
                        print(f"[HOMOGRAPHY] frame {frame_idx}: low confidence "
                              f"({conf_i:.2f} < {min_calib_confidence:.2f}), keeping previous homography", flush=True)
                else:
                    print(f"[HOMOGRAPHY] frame {frame_idx}: no pitch lines detected, "
                          f"keeping previous homography", flush=True)

            tracks = run_track(
                model, frame,
                classes=[0, 32],
                conf=det_conf,
                imgsz=input_size,
            )

            world_positions = {}
            teams = {}
            ball_px = None; best_ball_conf = 0.0

            for tr in tracks:
                tid, cls, conf = tr["tid"], tr["cls"], tr["conf"]
                x1, y1, x2, y2 = (int(v) for v in tr["box"])
                if cls == 0:  # person
                    patch = torso_patch(frame, (x1,y1,x2,y2))
                    stats = analyze_patch(patch)
                    if is_referee(stats, referee_max_mean_sat, referee_min_contrast):
                        continue
                    tc.add_sample(stats)
                    team = tc.classify(tid, stats)
                    teams[tid] = team
                    foot_x, foot_y = (x1+x2)/2, y2
                    wpos = mapper.to_world(foot_x, foot_y)
                    ps.update(tid, wpos)
                    world_positions[tid] = wpos
                    jersey_num = jersey.maybe_read(tid, patch, frame_idx) if jersey_ocr_enabled else None
                    draw_player(canvas, (x1,y1,x2,y2), team, tid, ps.speed_kmh[tid], tc, jersey_num)
                    if team == 'A': pos_a.append(wpos)
                    elif team == 'B': pos_b.append(wpos)
                elif cls == 32 and conf >= det_conf_ball:  # sports ball (config-gated threshold)
                    if conf > best_ball_conf:
                        best_ball_conf = conf
                        ball_px = ((x1+x2)/2, (y1+y2)/2)
            tc.maybe_fit(frame_idx)
            resolved_px = ball.update(ball_px, frame_idx)
            ball_world = ball.world_pos()

            holder_tid = poss.update(world_positions, teams, ball_world, frame_idx)
            holder_team = teams.get(holder_tid) if holder_tid is not None else None

            shot_event = shot_det.update(ball_world, frame_idx, holder_tid, holder_team)
            if shot_event:
                events.append(shot_event)

            goal_event = goal_det.check(ball_world, frame_idx, scorer_track_id=holder_tid)
            if goal_event:
                events.append(goal_event)
                goal_flash_frames = 20; goal_flash_team = goal_event["team"]

            if formation_enabled:
                formation.maybe_update(world_positions, teams, frame_idx)

            draw_ball(canvas, ball)
            draw_minimap(canvas, mapper, world_positions, teams, tc, ball, W, H)

            pa, pb = poss.pct()
            spds = [ps.speed_kmh[t] for t in world_positions]
            avg_s = float(np.mean(spds)) if spds else 0.0
            hist_spd.append(avg_s)

            banner_bottom_y = draw_top_banner(canvas, W, goal_det.goals['A'], goal_det.goals['B'])
            if formation_enabled:
                draw_formation_hud(canvas, formation, banner_bottom_y)
            draw_bottom_hud(canvas, pa, pb, avg_s, len(world_positions),
                             poss.passes.get('A',0), poss.passes.get('B',0), W, H)

            if goal_flash_frames > 0:
                draw_goal_flash(canvas, goal_flash_team, W, H)
                goal_flash_frames -= 1

            writer.write(canvas)
            frame_idx += 1
            pbar.update(1)

    cap.release(); writer.release()
    print(f"[DONE] {out_path} ✓", flush=True)
    print(f"[STATS] Goals A:{goal_det.goals['A']} B:{goal_det.goals['B']} | "
          f"Possession {poss.pct()} | Passes A:{poss.passes.get('A',0)} B:{poss.passes.get('B',0)} | "
          f"Turnovers {poss.turnovers}", flush=True)

    all_events = sorted(events + poss.pass_events + poss.turnover_events, key=lambda e: e["frame"])
    events_path = os.path.join(out_dir, "football_events.json")
    with open(events_path, "w") as f:
        json.dump({
            "video": os.path.basename(video_path),
            "generated_by": "abhinav.phi",
            "events": all_events,
        }, f, indent=2)
    print(f"[EVENT] {len(all_events)} events -> {events_path} ✓", flush=True)

    save_heatmap(pos_a, pos_b, os.path.join(out_dir, "football_heatmap.png"))
    save_pitch_map(all_events, os.path.join(out_dir, "football_pitch_map.png"))
    save_dashboard(hist_spd, poss, ps, goal_det.goals, poss.passes, poss.turnovers,
                    frame_idx, os.path.join(out_dir, "football_dashboard.png"),
                    formation if formation_enabled else None, all_events)


def parse_args(argv):
    if len(argv) < 1:
        print("Usage: python football_analytics.py video.mp4 [--config config/football_config.yaml]")
        sys.exit(1)
    video_path = argv[0]
    config_path = None
    if "--config" in argv:
        idx = argv.index("--config")
        if idx + 1 >= len(argv):
            print("[ERROR] --config requires a path argument"); sys.exit(1)
        config_path = argv[idx + 1]
    return video_path, config_path


if __name__ == "__main__":
    video_path, config_path = parse_args(sys.argv[1:])
    process(video_path, config_path)
