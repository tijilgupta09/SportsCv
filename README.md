# Sports Computer Vision Analytics Suite — Football & Cricket

**Author / Maintainer:** abhinav.phi
**Repo:** https://github.com/abhinav-phi/sports_cv
**Status:** Football = flagship-complete (Python). Cricket = MVP + V2 complete (Python). Neither has real-footage QA yet (see [Known Limitations](#known-limitations--open-bugs)). No C++ ports yet.

This README is the single entry point for understanding the entire project — architecture, every file's job, every feature, every config value, every known bug, and exactly what's left to do. If you're an AI agent or a new contributor picking this up cold, read this file fully before touching code; it supersedes memory/assumptions.

---

## 1. What this project actually is

A YOLOv8-based computer-vision analytics pipeline that takes a **football (soccer) match video** or a **cricket bowling-end video** and outputs an annotated video + data files (CSV/JSON) + chart/dashboard images — fully automatic, no manual labeling.

It originally started as a repo of independent sport scripts (Basketball, Hockey, Volleyball — pre-existing, untouched reference implementations) plus an empty Cricket placeholder and a basic Football script. **This project's scope is exclusively: refactor Football into a flagship pipeline, build Cricket from scratch, and extract genuinely shared logic into a `common/` package.** Basketball/Hockey/Volleyball are explicitly out of scope and must never be modified (see [Section 8](#8-hard-rules-that-govern-this-project)).

---

## 2. Repository layout

```
sports_cv/
├── football/football_analytics.py     # Football pipeline (entrypoint script)
├── cricket/cricket_analytics.py       # Cricket pipeline (entrypoint script)
├── common/                            # Shared logic used by both pipelines
│   ├── homography.py                  # PlanarMapper — pixel <-> real-world coordinate mapping
│   ├── team_classifier.py             # K-means jersey-color team classifier + vote-locking
│   ├── kalman.py                      # Kalman2D — constant-velocity 2D filter (cricket ball trajectory)
│   ├── ocr.py                         # EasyOCR wrappers: jersey numbers + scoreboard reading
│   ├── draw_utils.py                  # txt()/label_block() — shared on-video drawing primitives
│   ├── dashboard_utils.py             # make_dark_figure()/apply_dark_theme()/style_table() — shared matplotlib dark theme
│   ├── detection.py                   # run_detect()/run_track() — normalized YOLO predict/track wrappers
│   ├── tracking.py                    # parse_track_results() — normalizes ByteTrack output across ultralytics versions
│   ├── video_io.py                    # make_writer() — codec-fallback VideoWriter
│   └── config_loader.py               # load_config() — YAML loader + required-key validator
├── config/
│   ├── football_config.yaml           # All football tunables (pitch, calibration, thresholds, colors)
│   └── cricket_config.yaml            # All cricket tunables (pitch, calibration, thresholds, colors)
├── tests/                             # pytest suite — pure-logic tests, no video needed to run
│   ├── test_homography.py
│   ├── test_kalman.py
│   ├── test_team_classifier.py
│   └── test_config_loader.py
├── basketball/, hockey/, volleyball/  # PRE-EXISTING reference sports. Out of scope. Gitignored locally.
├── 01_prd.md … 08_rules.md            # Planning docs (product spec, tech spec, app flow, design, schema,
│                                       #   implementation plan, live tracker, binding rules) — currently
│                                       #   gitignored, NOT on GitHub (see Known Limitations — fix this).
├── pytest.ini
├── LICENSE
└── README.md                          # this file
```

**Note on the 8 planning docs:** these are the actual source of truth for requirements/status (especially `07_tracker.md`, the live task tracker) but are currently excluded from git via `.gitignore`. This is a known gap, not intentional design — they should be un-ignored and committed so project history/status isn't stranded on one local machine.

---

## 3. How to run it

```bash
python football/football_analytics.py path/to/match.mp4 [--config config/football_config.yaml]
python cricket/cricket_analytics.py  path/to/clip.mp4   [--config config/cricket_config.yaml]
```

- First run auto-installs missing pip packages (`opencv-python`, `numpy`, `matplotlib`, `scipy`, `ultralytics`, `Pillow`, `tqdm`, `lapx`, `PyYAML`, `easyocr`) and auto-downloads `yolov8n.pt` COCO weights via ultralytics.
- CPU-only works (slow); GPU auto-used if `torch.cuda.is_available()`.
- **First thing every run does:** save a calibration preview PNG (`football_pitch_preview.png` / `cricket_pitch_preview.png`) from frame 0. **Always check this image before trusting any output numbers** — if the drawn corners/stump-points don't line up with your actual footage, edit the pixel coordinates in the config YAML and re-run.
- If `--config` is omitted, defaults to `config/<sport>_config.yaml`. Missing config file = hard fail with an actionable error (never silently uses undocumented defaults).

### Outputs produced

**Football** (all in the input video's directory):
| File | What it is |
|---|---|
| `football_output.mp4` | Annotated video: player boxes+labels, ball trail, minimap, top banner, bottom HUD |
| `football_pitch_preview.png` | Calibration check image (see above) |
| `football_events.json` | Every shot/goal/pass/turnover event with frame, team, track id, world position |
| `football_heatmap.png` | Per-team position density on a to-scale pitch outline |
| `football_pitch_map.png` | Shot map: markers sized/colored by xG (est.), on/off-target, goals |
| `football_dashboard.png` | 2×2 report: speed-over-time, possession pie, top-speed leaderboard, summary table |

**Cricket** (all in the input video's directory):
| File | What it is |
|---|---|
| `cricket_output.mp4` | Annotated video: ball trail, speed-gun HUD, bounce flash, live pitch mini-inset |
| `cricket_pitch_preview.png` | Calibration check image |
| `cricket_deliveries.csv` | One row per delivery: release/bounce/complete frame, bounce (x,y) in meters, speed, delivery_type |
| `cricket_pitch_map.png` | Top-down pitch diagram, bounce point per delivery, colored by speed bucket — **the single most important visual output** |
| `cricket_dashboard.png` | 2×2 report: speed histogram, outcome breakdown, pitch-map thumbnail, summary table |
| `cricket_wagon_wheel.png` | *(only if `wagon_wheel.enabled: true`)* polar field diagram of post-contact ball direction |
| `cricket_events.json` | *(only if `scoreboard_ocr.enabled: true`)* OCR-read scoreboard snapshots, tagged `"source":"ocr"` |

---

## 4. Football — full feature list

1. **Real-world coordinate mapping (homography), not a flat pixel-to-meter constant.** `common/homography.PlanarMapper` maps 4 pitch-corner pixels to real meters (105×68m standard pitch) via `cv2.getPerspectiveTransform`. This is why distances/speeds are accurate despite camera perspective distortion.
2. **Auto pitch-line detection + rolling re-calibration.** HSV green mask → Canny → Hough lines → intersect boundary lines → 4 corners, plus the halfway line as an independent confidence check point. Re-attempted every `recalibration_interval_frames` (default 150). A candidate homography is only swapped in if its reprojection-error-derived confidence ≥ `min_calibration_confidence` (default 0.75); otherwise the previous good homography is kept and a `[HOMOGRAPHY]` warning is logged. **Never silently applies a bad calibration.**
3. **Guaranteed manual fallback.** If auto-detection ever fails (no lines visible, tight crop, non-green surface), falls back to `calibration.pixel_corners_fallback` from the config YAML. This path is always available, never removable.
4. **Player tracking via ByteTrack** (`model.track(persist=True, tracker="bytetrack.yaml")`, wrapped by `common.detection.run_track`) — persistent IDs across frames, far more robust than a hand-rolled centroid tracker.
5. **Team classification**: generic k-means clustering on jersey hue/saturation (`common/team_classifier.py`), referee-rejection (low saturation + high grayscale contrast = ref, excluded), and vote-locking (a track's team locks only after `vote_lock_thresh` consistent votes) to prevent flicker.
6. **Ball tracking with short-gap prediction**: if the ball isn't detected for up to `gap_predict_frames` (default 6), its position is linearly extrapolated from last known velocity so the trail doesn't just vanish.
7. **Jersey number OCR**: EasyOCR on the same torso-crop already used for team-color sampling, rate-limited to once per `ocr_interval_frames` per track, majority-voted, only displayed once `ocr_min_votes` consistent reads accumulate. Label switches from `#track_id` to `#jersey_number` once confident.
8. **Shot detection**: fires when ball world-space velocity ≥ `shots.velocity_threshold_ms`, direction points at a goal mouth within `angle_tolerance_deg`, and it's within `max_distance_m` of that goal.
9. **xG (est.)**: simple logistic function of (distance to goal, angle) — explicitly labeled "(est.)" everywhere, never presented as a real xG model. Field name is `xg_estimate`, never bare `xg`.
10. **Goal detection**: ball crosses either goal line (world x ≈ 0 or ≈ pitch length) within the goal-mouth y-range, with a cooldown to prevent double-counting. On-video "GOAL!" flash triggers.
11. **Possession & passes**: nearest player within `possession.radius_m` of the ball "has" it; a possession change to the same team over ≥ `pass_min_distance_m` = completed pass; to the other team = turnover.
12. **Formation / "Estimated Shape"**: every `formation.update_interval_frames`, buckets each team's players into defense/mid/attack thirds of the pitch. Explicitly labeled "Estimated Shape", never "Formation" — it's a rough heuristic, not tactical-line detection.
13. **Sprint counting**: sustained speed above `sprint.sprint_kmh` for ≥ `sprint_min_frames` counts as one sprint; speeds are smoothed (6-frame rolling average) and clipped to `ball.max_realistic_speed_kmh` to avoid ID-swap speed spikes.
14. **On-video HUD**: top banner (live score), bottom bar (possession split bar, avg speed, player count, pass counts), minimap (pitch-shaped, live player+ball dots), player labels (id/jersey + speed), ball trail with glow, goal flash overlay.
15. **Exports**: `football_events.json` (all events merged+sorted by frame), pitch-shaped heatmap PNG, shot-map PNG, 2×2 dashboard PNG — see [Section 3](#3-how-to-run-it) table.

---

## 5. Cricket — full feature list

**Camera assumption (hard constraint, documented up front):** a mostly static bowler's-end (or batsman's-end) wide camera. Side-on/handheld/heavy-pan footage is unsupported until manually re-verified via the preview PNG.

1. **Stump-based calibration**: user-marked (or config-set) far-end and near-end stump pixel pairs map to a real pitch quadrilateral (20.12m × stump width) via the same `PlanarMapper` class football uses — this is the concrete proof the shared homography module generalizes beyond football.
2. **Tuned ball detection**: separate, lower confidence threshold for the `sports ball` COCO class vs. person (`detection.det_conf_ball` = 0.20 vs `det_conf_person` = 0.40 by default) since a cricket ball is small, fast, and motion-blurred — YOLOv8n was never trained specifically on it, so expect some false negatives (this is exactly why the Kalman layer exists).
3. **Kalman-filtered ball trajectory** (`common/kalman.Kalman2D`, constant-velocity model): on each frame, `correct()` if detected, else `predict()` for up to `kalman.max_predict_gap_frames` (default 8) frames, keeping the trail alive through brief occlusions. Predicted (vs. real) points are visually distinguished (gray, vs. colored).
4. **Delivery state machine**: `IDLE → RELEASED → IN_FLIGHT → BOUNCED → COMPLETE`, logged via `[DELIVERY]` console tags on every transition.
   - `RELEASED`: ball detected leaving the bowler's release zone (config pixel-radius around near-stump midpoint) above `release_velocity_threshold_ms`.
   - `BOUNCED`: vertical pixel-velocity sign change (downward→upward) while inside `bounce_zone_y_fraction` of the frame.
   - `COMPLETE`: fires on **either** of two triggers — the ball entering the batsman-zone (`batsman_zone_y_fraction`, upper portion of frame, checked during both `IN_FLIGHT` and `BOUNCED`), **or** `delivery_timeout_frames` (default 90) of no real detection. Whichever fires first.
5. **Speed estimation**: world-space distance from release point to arrival point ÷ elapsed time. Arrival point uses the last known **real** (non-predicted) ball pixel position as a fallback if the completing frame itself has no fresh detection — this was a real bug (see [Known Limitations](#known-limitations--open-bugs), BUG-09) that's now fixed.
6. **Speed-gun HUD**: broadcast-style "SPEED: 132.4 km/h" panel, appears after a delivery completes, holds for `speed_gun.display_duration_frames`, then clears — deliberately not live-updated mid-flight (avoids a jittery, unrealistic number).
7. **Length/line zone classification** happens implicitly via bounce world-coordinates recorded per delivery (used in the pitch map, not a separate on-HUD label).
8. **Fast/spin classification (est.)**: speed > 120 km/h → "fast", < 90 km/h → "spin", else "medium" — a simple heuristic, always labeled as an estimate.
9. **Bat-contact detection (P1, off by default)**: only active when `wagon_wheel.enabled: true`. Watches for a sudden ball-velocity magnitude change (≥ `contact_velocity_delta_threshold`) after the bounce — heuristic, labeled "(est.)".
10. **Wagon wheel export (P1, off by default)**: polar field diagram, direction from pitch-center toward each detected contact point, colored by delivery speed.
11. **Scoreboard OCR (P1, off by default)**: `common/ocr.read_scoreboard` on a configured broadcast-graphic ROI (`scoreboard_ocr.roi_pixels`), rate-limited to once per `read_interval_frames`, best-effort regex-parsed for runs/wickets/overs, always tagged `"source":"ocr"` — never silently merged with vision-inferred data.
12. **Exports**: `cricket_deliveries.csv`, `cricket_pitch_map.png` (highest-priority visual — bounce points colored by speed bucket via `RdYlGn_r` colormap), 2×2 `cricket_dashboard.png`, optional wagon wheel PNG, optional `cricket_events.json`.

**Explicit honesty note baked into the code and docs:** this is a single-camera 2D system, not a multi-camera triangulating Hawk-Eye-style setup. Speed/bounce-point/six-vs-four-style calls are estimates for a fun broadcast-style overlay and rough analytics — not umpiring-grade measurements.

---

## 6. `common/` module reference (what each file does, exactly)

| Module | Responsibility |
|---|---|
| `homography.py` | `PlanarMapper(pixel_pts, world_pts)` — 4-point perspective transform, `to_world()`/`to_pixel()`, `reprojection_error()`, and the static factory `re_estimate()` used for confidence-gated rolling re-calibration. |
| `team_classifier.py` | `torso_patch()`, `analyze_patch()` (HSV+grayscale stats), `is_referee()`, and `TeamClassifier` (k-means 2-cluster fit + per-track vote-locking). Sport-agnostic — used identically by football (cricket does not currently use team classification). |
| `kalman.py` | `Kalman2D` — thin `cv2.KalmanFilter` wrapper (`predict()`, `correct()`, `get_state()`), constant-velocity 4-state model. Used by cricket's ball trajectory. |
| `ocr.py` | `read_jersey_number(patch)` and `read_scoreboard(frame, roi_pixels)`, both backed by a lazily-loaded singleton EasyOCR reader (CPU-only, no system Tesseract dependency). |
| `draw_utils.py` | `txt()` (outline-then-fill text) and `label_block()` (translucent background label) — the two primitives shared verbatim across both sports' HUD rendering. |
| `dashboard_utils.py` | `make_dark_figure()`, `apply_dark_theme()`, `style_table()`, plus shared color constants (`DARK_BG`, `DARK_PANEL`, `ACCENT_CYAN`, etc.) — the dark-theme matplotlib boilerplate factored out of both sports' dashboard/heatmap/pitch-map/wagon-wheel functions. **Now actually wired into both scripts** (previously existed but was unused — fixed). |
| `detection.py` | `run_detect()` (YOLO `predict()`, normalized detection dicts) and `run_track()` (YOLO `track()` w/ ByteTrack, normalized track dicts incl. persistent `tid`). **Now actually called** by both scripts instead of inline `model.track()`/`model.predict()` calls — fixed. |
| `tracking.py` | `parse_track_results()` — normalizes a raw ultralytics `Results` object into track dicts; isolates future ultralytics API changes to one place. |
| `video_io.py` | `make_writer(path, fps, W, H)` — tries codecs `mp4v → avc1 → H264 → h264` in order, returns the first that opens successfully. |
| `config_loader.py` | `load_config(sport, path=None)` — loads `config/<sport>_config.yaml`, validates every required section/key is present, raises a named `KeyError`/`FileNotFoundError`/`ValueError` (never a raw `KeyError` with no context). |

---

## 7. Config reference (every tunable value lives in YAML, never hardcoded)

Full schema lives in `05_schema.md`; summarized here.

**`config/football_config.yaml` sections:** `pitch` (real dimensions: length/width/goal width/penalty box/center circle), `calibration` (fallback pixel corners, recalibration interval, min confidence, pitch HSV hue/sat range), `detection` (`det_conf_person`, `det_conf_ball` — now both actually used, `input_size`), `team_classification` (calibration frame count, min samples, referee thresholds, vote-lock params), `ball` (gap-predict frames, max realistic speed), `sprint` (kmh threshold, min frames), `possession` (radius, pass min distance), `shots` (velocity/angle/distance thresholds), `ocr` (enabled, interval, min votes), `formation` (enabled, update interval), `goal_detection` (cooldown frames), `colors` (team A/B, ball, goal flash — BGR tuples).

**`config/cricket_config.yaml` sections:** `pitch` (length, stump width, popping crease offset), `calibration` (camera angle documentation string, far/near stump pixel points, auto-detect flag), `detection` (`det_conf_person`, `det_conf_ball`, `ball_input_size`), `kalman` (process/measurement noise, max predict gap), `delivery` (release zone radius/velocity threshold, timeout frames, bounce zone y-fraction, **`batsman_zone_y_fraction`** — added as part of the BUG-09/BUG-11 fix), `speed_gun` (display duration), `wagon_wheel` (enabled flag, contact velocity delta threshold), `scoreboard_ocr` (enabled flag, ROI pixels, read interval), `colors` (ball trail, bounce flash, speed gun text — BGR tuples).

**Golden rule (RULES.md §2):** if a value could plausibly differ between two different video clips of the same sport, it lives in config, never as a Python constant. True physical constants (pitch length, etc.) may be hardcoded only if also mirrored in the schema doc so source and docs never disagree.

---

## 8. Hard rules that govern this project

(Full detail in `08_rules.md`; the ones that matter most if you're extending this:)

- **Only** `football/`, `cricket/`, `common/`, `config/`, `tests/`, and the planning docs may be created/modified. **Never touch** `basketball_match_analytics.cpp`, `basketball.py`, `hockey_analytics.cpp`, `hockey.py`, `volleyball.py`, `volleyball_analytics.cpp` — reference-only.
- Every estimated/heuristic value (xG, formation shape, delivery type, wagon-wheel contact, OCR scoreboard) must be labeled as an estimate everywhere it's shown — HUD, dashboard, and field naming (`xg_estimate` not `xg`).
- Never let a low-confidence calibration silently replace a good one — accept/reject/keep-previous, always logged.
- Must run on CPU-only hardware (GPU is an accelerant, never a requirement). No expensive per-frame operation (OCR, etc.) without rate-limiting.
- Credit string in all new/touched files: exactly `abhinav.phi`.
- A task isn't `Done` in the tracker until its acceptance criteria are actually verified (test run / output checked) — not assumed because the code compiles.

---

## 9. Testing

```bash
pytest -q     # 18 tests, pure-logic only (homography math, Kalman prediction, team-clustering, config validation) — no video file needed
```

All 18 currently pass. These test the **math/logic layer only** — homography round-trips + `re_estimate()` accept/reject confidence scoring, Kalman straight-line-through-a-gap prediction, k-means team clustering + referee rejection + vote-locking, and config loader error paths. **They do not exercise the full video pipeline** (detection accuracy, real delivery/shot detection, HUD rendering) — that requires an actual end-to-end run on a real video clip, which has only ever been done against a synthetic placeholder clip (see below).

---

## 10. Known limitations / open items

1. **No real football or cricket footage has ever been run through either pipeline.** Everything has been validated against `synthetic_test_clip.mp4` (flat colored rectangles, no real people/ball) — which correctly proves the *pipeline runs end-to-end without crashing* and *produces all expected output files*, but cannot prove detection accuracy, real shot/goal/delivery counts, or realistic speed values. **This is the single most important next step** — run both scripts against real footage and check the preview PNG + output numbers make sense.
2. **8 planning docs (`01_prd.md`–`08_rules.md`) are gitignored, not on GitHub.** Should be un-ignored and committed — `07_tracker.md` is the project's live status source of truth and currently only exists on one local machine.
3. **Cricket runs ball detection as a separate full video pre-pass**, then decodes the video a second time in the main loop — works correctly but is ~2x slower than necessary on real footage. Deferred; not fixed.
4. Previously-fixed and now resolved (kept here for history — see `07_tracker.md` BUG-09/10/11 for full detail):
   - Cricket speed was always recorded as `0.0 km/h` on any delivery that completed via the ball-going-undetected path (the common case) — **fixed**: last known real ball position is now used as the arrival-point fallback.
   - `common/dashboard_utils.py`, `detection.py`, `tracking.py` existed but were never called by either script (dead code, duplicated dark-theme/YOLO-call boilerplate remained inline) — **fixed**: both scripts now import and call these shared helpers.
   - Delivery `COMPLETE` only ever fired via 90-frame timeout, never via the documented "ball reaches batsman zone" trigger — **fixed**: `_in_batsman_zone()` now checked in both `IN_FLIGHT` and `BOUNCED` states.
   - `football_config.yaml`'s `det_conf_ball` was defined but never read (both person and ball detections shared one threshold) — **fixed**: now actually applied as the ball-class filter threshold.
5. No C++ ports exist yet for either sport (`football_analytics.cpp` / `cricket_analytics.cpp` — planned, not started, low priority until Python side has real-footage QA).

---

## 11. Glossary / naming conventions worth knowing

- **"(est.)"** anywhere in output (xG, delivery type, formation shape, wagon-wheel direction) = a heuristic approximation, not a precise measurement. Never remove this label when touching that code.
- **`tid`** = track ID (persistent per-player/object ID from ByteTrack), distinct from a resolved jersey number.
- **World coordinates** = real-world meters (pitch-plane for football, pitch-strip-plane for cricket), as opposed to raw pixel coordinates — always the basis for any distance/speed/velocity calculation, never raw pixels.
- **BGR**, not RGB — all color config values follow OpenCV's convention.

---

*If anything in this README ever disagrees with the actual code or with `07_tracker.md`, the code + tracker win — update this README in the same change that causes the disagreement.*