"""Minimal constant-velocity 2D Kalman filter helper for smoothing/predicting ball trajectories."""
import cv2
import numpy as np


class Kalman2D:
    def __init__(self, process_noise: float = 0.03, measurement_noise: float = 0.5):
        self.filter = cv2.KalmanFilter(4, 2)
        self.filter.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.filter.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )
        self.filter.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.filter.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def correct(self, x: float, y: float) -> tuple[float, float, float, float]:
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            self.filter.statePre = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
            self.filter.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
            self.initialized = True
            return self.get_state()
        self.filter.correct(measurement)
        return self.get_state()

    def predict(self) -> tuple[float, float, float, float]:
        if not self.initialized:
            return 0.0, 0.0, 0.0, 0.0
        predicted = self.filter.predict()
        self.filter.statePost = predicted
        return self.get_state()

    def get_state(self) -> tuple[float, float, float, float]:
        state = self.filter.statePost
        return float(state[0, 0]), float(state[1, 0]), float(state[2, 0]), float(state[3, 0])