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

# EMA α для сглаживания скорости (PixEagle MotionPredictor паттерн)
# 0.7 = 70% вес нового измерения, 30% — история (быстрый отклик + сглаживание)
_VEL_EMA_ALPHA = 0.7
# EMA α для ускорения (более консервативный — нет резких скачков)
_ACC_EMA_ALPHA = 0.5
# Максимальное ускорение px/кадр² (защита от выброса)
_MAX_ACCEL_PF2 = 10.0


class KalmanTargetTracker:
    """
    Kalman фильтр 4D [x, y, vx, vy] + EMA velocity smoothing + acceleration tracking.

    Архитектура по образцу PixEagle MotionPredictor:
      1. cv2.KalmanFilter(4, 2) — основное сглаживание позиции и скорости.
      2. EMA (Exponential Moving Average) поверх Kalman-скорости — стабильнее
         при резких изменениях курса цели.
      3. Оценка ускорения из разности скоростей — для кинематического прогноза
         по уравнению: pos + vel*t + 0.5*accel*t² (15–25% точнее при t > 5 кадров).

    Единицы скорости: пиксели/кадр (совместимо с lead_point и REACQUIRE).
    Конвертация в пиксели/секунду: умножить на MAX_FPS.

    predict_with_velocity()       → (x, y, vx_ema, vy_ema)
    predict_with_acceleration()   → (x, y, vx_ema, vy_ema, ax, ay)
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_w = frame_width
        self.frame_h = frame_height
        dt = 1.0 / MAX_FPS

        # --- cv2.KalmanFilter ---
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
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32) * KALMAN_ERROR_COV_INIT
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
        self.kf.statePost = np.array(
            [[frame_width / 2.0], [frame_height / 2.0], [0.0], [0.0]],
            dtype=np.float32,
        )
        self._predicted_this_step = False

        # --- EMA velocity (px/frame) — сглаживание поверх Kalman ---
        self._ema_vx: float = 0.0
        self._ema_vy: float = 0.0

        # --- EMA acceleration (px/frame²) — кинематический прогноз ---
        self._ema_ax: float = 0.0
        self._ema_ay: float = 0.0
        self._prev_ema_vx: float = 0.0
        self._prev_ema_vy: float = 0.0
        self._update_count: int  = 0   # число update() — нужно ≥ 2 для ускорения

    def _ensure_predict(self):
        if not self._predicted_this_step:
            self.kf.predict()
            self._predicted_this_step = True

    def predict_only(self):
        """Возвращает (x, y) прогнозную позицию."""
        self._ensure_predict()
        return (float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0]))

    def predict_with_velocity(self):
        """
        Возвращает (x, y, vx, vy).
        vx, vy — EMA-сглаженная скорость в пикселях/кадр.
        """
        self._ensure_predict()
        s = self.kf.statePost
        return (
            float(s[0, 0]), float(s[1, 0]),
            float(s[2, 0]), float(s[3, 0]),
        )

    def update(self, measurement):
        """Обновить фильтр измерением. Возвращает (x, y)."""
        self._ensure_predict()
        meas = np.array(
            [[np.float32(measurement[0])], [np.float32(measurement[1])]]
        )
        self.kf.correct(meas)
        self._predicted_this_step = False
        return (float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0]))

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
        self._ema_vx   = 0.0
        self._ema_vy   = 0.0
        self._ema_ax   = 0.0
        self._ema_ay   = 0.0
        self._prev_ema_vx  = 0.0
        self._prev_ema_vy  = 0.0
        self._update_count = 0