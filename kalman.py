import cv2
import numpy as np
import logging
from config import (
    MAX_FPS,
    KALMAN_PROCESS_NOISE,
    KALMAN_MEASUREMENT_NOISE,
    KALMAN_ERROR_COV_INIT,
)

logger = logging.getLogger(__name__)


class KalmanTargetTracker:
    """
    Kalman фильтр 4D для сглаживания позиции и скорости цели.

    Состояние вектора: [x, y, vx, vy]
      x, y   — позиция в пикселях
      vx, vy — скорость в пикселях/кадр

    predict_with_velocity() возвращает (x, y, vx, vy) для
    расчёта точки упреждения (predictive intercept).
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_w = frame_width
        self.frame_h = frame_height
        dt = 1.0 / MAX_FPS

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.errorCovPost     = np.eye(4, dtype=np.float32) * KALMAN_ERROR_COV_INIT
        self.kf.processNoiseCov  = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
        self.kf.statePost = np.array(
            [[frame_width / 2.0], [frame_height / 2.0], [0.0], [0.0]],
            dtype=np.float32,
        )
        self._predicted_this_step = False

    def _ensure_predict(self):
        if not self._predicted_this_step:
            self.kf.predict()
            self._predicted_this_step = True

    def predict_only(self):
        """Возвращает (x, y) прогнозную позицию."""
        self._ensure_predict()
        return (float(self.kf.statePost[0]), float(self.kf.statePost[1]))

    def predict_with_velocity(self):
        """
        Возвращает (x, y, vx, vy).
        vx, vy в пикселях/кадр — используются для упреждения.
        """
        self._ensure_predict()
        s = self.kf.statePost
        return (
            float(s[0]), float(s[1]),
            float(s[2]), float(s[3]),
        )

    def update(self, measurement):
        """Обновить фильтр измерением. Возвращает (x, y)."""
        self._ensure_predict()
        meas = np.array(
            [[np.float32(measurement[0])], [np.float32(measurement[1])]]
        )
        self.kf.correct(meas)
        self._predicted_this_step = False
        return (float(self.kf.statePost[0]), float(self.kf.statePost[1]))

    def begin_step(self):
        """Сбросить флаг предсказания — вызывать в начале каждого кадра."""
        self._predicted_this_step = False

    def reset(self):
        """Сбросить в центр кадра."""
        self.kf.statePost = np.array(
            [[self.frame_w / 2.0], [self.frame_h / 2.0], [0.0], [0.0]],
            dtype=np.float32,
        )
        self.kf.errorCovPost      = np.eye(4, dtype=np.float32) * KALMAN_ERROR_COV_INIT
        self._predicted_this_step = False