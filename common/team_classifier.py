"""Generic, sport-agnostic two-team jersey-color classifier via k-means clustering with vote-locking."""
import cv2
import numpy as np
from collections import defaultdict, deque


def torso_patch(frame, box):
    x1, y1, x2, y2 = box
    h = y2 - y1
    ty1 = int(y1 + h * 0.15); ty2 = int(y1 + h * 0.50)
    tx1 = int(x1 + (x2 - x1) * 0.22); tx2 = int(x2 - (x2 - x1) * 0.22)
    ty1, ty2 = max(0, ty1), max(ty1 + 1, ty2)
    tx1, tx2 = max(0, tx1), max(tx1 + 1, tx2)
    patch = frame[ty1:ty2, tx1:tx2]
    return patch if patch.size else None


def analyze_patch(patch):
    if patch is None:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return (float(np.mean(hsv[:, 0])), float(np.mean(hsv[:, 1])),
            float(np.mean(gray)), float(np.std(gray)))


def is_referee(stats, referee_max_mean_sat, referee_min_contrast):
    if stats is None:
        return False
    _, sat, _, contrast = stats
    return sat < referee_max_mean_sat and contrast > referee_min_contrast


class TeamClassifier:
    """Learns 2 jersey-color clusters via k-means, then locks each track's team
    once enough consistent votes accumulate (avoids per-frame flicker)."""

    def __init__(self, calibration_frames, min_calib_samples, min_sat_for_cluster,
                 vote_buffer_len, vote_lock_thresh):
        self.calibration_frames = calibration_frames
        self.min_calib_samples = min_calib_samples
        self.min_sat_for_cluster = min_sat_for_cluster
        self.vote_buffer_len = vote_buffer_len
        self.vote_lock_thresh = vote_lock_thresh

        self.samples = []
        self.centers = None
        self.votes = defaultdict(lambda: deque(maxlen=vote_buffer_len))
        self.locked = {}

    def add_sample(self, stats):
        if self.centers is not None or stats is None:
            return
        hue, sat, _, _ = stats
        if sat >= self.min_sat_for_cluster:
            self.samples.append((hue, sat))

    def maybe_fit(self, frame_idx):
        if self.centers is not None:
            return
        if frame_idx >= self.calibration_frames and len(self.samples) >= self.min_calib_samples:
            data = np.array(self.samples, dtype=np.float32)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.4)
            _, _, centers = cv2.kmeans(data, 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)
            self.centers = centers
            print(f"[TEAMS] Learned 2 jersey clusters from {len(data)} samples ✓", flush=True)

    def classify(self, tid, stats):
        if tid in self.locked:
            return self.locked[tid]
        if self.centers is None or stats is None:
            return None
        hue, sat, _, _ = stats
        if sat < self.min_sat_for_cluster:
            return None
        v = np.array([hue, sat], dtype=np.float32)
        d0 = np.linalg.norm(v - self.centers[0])
        d1 = np.linalg.norm(v - self.centers[1])
        vote = 'A' if d0 <= d1 else 'B'
        buf = self.votes[tid]
        buf.append(vote)
        if len(buf) >= self.vote_lock_thresh:
            a = buf.count('A'); b = buf.count('B')
            if a >= self.vote_lock_thresh: self.locked[tid] = 'A'; return 'A'
            if b >= self.vote_lock_thresh: self.locked[tid] = 'B'; return 'B'
        return vote

