"""
TrackerEngine — алгоритм перехвата цели (PixEagle архитектура).

Поддерживаемые направления атаки (6 DOF):
  Горизонталь:  цель справа → yaw вправо, цель слева → yaw влево
  Вертикаль:    цель выше → throttle вверх, цель ниже → throttle вниз + pitch
  Диагонали:    любая комбинация (дрон выходит на траекторию перехвата)
  Позади выше:  yaw разворот + throttle вверх
  Позади ниже:  yaw разворот + pitch пикирование

Плавное ускорение:
  - Throttle нарастает через THROTTLE_RAMP_SEC (нет рывков)
  - Pitch нарастает через RAMP_DURATION_SEC
  - Dead reckoning при потере цели до 0.4 сек

State машина:
  IDLE → engage() → ACQUIRING → (детекция) → TRACKING
  TRACKING → (потеря <0.4с) → DEAD_RECKON → (потеря >0.4с) → LOST
  TRACKING → (рампа 100%) → STRIKING
  LOST / disengage() → IDLE
"""

import time
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

from config import (
    FRAME_WIDTH, FRAME_HEIGHT,
    RC_MID, RC_RELEASE, RC_SAFE_MIN, RC_SAFE_MAX,
    RC_THROTTLE_MIN, RC_THROTTLE_MAX,
    PITCH_NEAR, PITCH_DIVE,
    RAMP_DURATION_SEC, THROTTLE_RAMP_SEC,
    DEAD_RECKONING_SEC,
    KP_YAW, KI_YAW, KD_YAW,
    KP_ALT, KI_ALT, KD_ALT,
    TARGET_CLASS_ID, CONF_THRESHOLD,
    MAX_RC_DELTA_PER_FRAME,
)
from types_enum import TrackerState
from pid import PIDController
from vision_tracker import VisionTracker

logger = logging.getLogger(__name__)


@dataclass
class TrackResult:
    """Результат одного шага трекера — передаётся в ControlManager."""
    state:          TrackerState = TrackerState.IDLE
    target_x:       float = -1.0
    target_y:       float = -1.0
    lead_x:         float = -1.0   # точка упреждения X
    lead_y:         float = -1.0   # точка упреждения Y
    confidence:     float = 0.0
    err_x:          float = 0.0
    err_y:          float = 0.0
    # RC_RELEASE (65535) = passthrough = оператор управляет этим каналом
    rc_roll:        int   = RC_RELEASE   # Roll всегда оператору (крен для манёвра)
    rc_pitch:       int   = RC_RELEASE
    rc_throttle:    int   = RC_RELEASE
    rc_yaw:         int   = RC_RELEASE
    ramp_progress:  float = 0.0
    throttle_ramp:  float = 0.0


def _idle_result() -> TrackResult:
    return TrackResult(state=TrackerState.IDLE)


def _lost_result(ramp: float = 0.0) -> TrackResult:
    return TrackResult(state=TrackerState.LOST, ramp_progress=ramp)


def _rate_limit(new_val: int, prev_val: int) -> int:
    """Ограничить скорость изменения RC значения за один кадр."""
    delta = int(np.clip(new_val - prev_val,
                        -MAX_RC_DELTA_PER_FRAME, MAX_RC_DELTA_PER_FRAME))
    return prev_val + delta


class TrackerEngine:
    """
    Полный алгоритм перехвата цели.

    Использование:
        engine = TrackerEngine()
        engine.engage()
        result = engine.step(yolo_outputs, frame)
        engine.disengage()
    """

    def __init__(self):
        self._vision        = VisionTracker()
        self._pid_yaw       = PIDController(KP_YAW, KI_YAW, KD_YAW)
        self._pid_alt       = PIDController(KP_ALT, KI_ALT, KD_ALT)
        self._state         = TrackerState.IDLE
        self._ramp_start    = 0.0
        self._ramp_progress = 0.0
        self._throttle_ramp = 0.0   # отдельная рампа для throttle
        self._throttle_ramp_start = 0.0
        # Предыдущие RC значения для rate limiting (плавность без рывков)
        self._prev_yaw      = RC_MID
        self._prev_throttle = RC_MID
        self._prev_pitch    = RC_MID

    @property
    def state(self) -> TrackerState:
        return self._state

    def engage(self):
        """Начать перехват. Сброс всех состояний."""
        self._vision.reset()
        self._pid_yaw.reset()
        self._pid_alt.reset()
        self._ramp_start    = time.monotonic()
        self._ramp_progress = 0.0
        self._throttle_ramp = 0.0
        self._throttle_ramp_start = time.monotonic()
        self._prev_yaw      = RC_MID
        self._prev_throttle = RC_MID
        self._prev_pitch    = RC_MID
        self._state = TrackerState.ACQUIRING
        logger.info("TrackerEngine: ENGAGE → ACQUIRING")

    def disengage(self):
        """Остановить перехват. Возврат в IDLE."""
        self._vision.reset()
        self._pid_yaw.reset()
        self._pid_alt.reset()
        self._ramp_progress = 0.0
        self._throttle_ramp = 0.0
        self._prev_yaw      = RC_MID
        self._prev_throttle = RC_MID
        self._prev_pitch    = RC_MID
        self._state = TrackerState.IDLE
        logger.info("TrackerEngine: DISENGAGE → IDLE")

    def step(self, yolo_outputs, frame: np.ndarray) -> TrackResult:
        """
        Один шаг трекинга и управления.

        Args:
            yolo_outputs: результат YOLO post_process() или None
            frame:        BGR кадр (640×480) для CSRT

        Returns:
            TrackResult с RC значениями для ControlManager
        """
        if self._state == TrackerState.IDLE:
            return _idle_result()

        # VisionTracker: YOLO + CSRT + Kalman
        # ИСПРАВЛЕНО: frame — первый аргумент
        vision_result = self._vision.step(frame, yolo_outputs)

        if vision_result is None:
            return self._handle_lost()

        cx, cy, conf, bbox = vision_result
        lead_x, lead_y = self._vision.get_lead_point()

        # Используем точку упреждения для наводки (predictive intercept)
        # При малой скорости цели lead ≈ cx,cy (LEAD_FACTOR компенсирует)
        aim_x = lead_x
        aim_y = lead_y

        # Ошибки от центра кадра (к точке упреждения)
        err_x = aim_x - (FRAME_WIDTH  / 2.0)
        err_y = aim_y - (FRAME_HEIGHT / 2.0)

        # --- Обновляем рампы ---
        now = time.monotonic()

        # Pitch рампа (пикирование)
        elapsed_pitch = now - self._ramp_start
        self._ramp_progress = min(elapsed_pitch / RAMP_DURATION_SEC, 1.0)

        # Throttle рампа (плавное ускорение без рывков)
        elapsed_thr = now - self._throttle_ramp_start
        self._throttle_ramp = min(elapsed_thr / THROTTLE_RAMP_SEC, 1.0)

        # --- A. YAW — горизонтальная наводка ---
        raw_yaw = RC_MID + self._pid_yaw.update(err_x)
        rc_yaw = int(np.clip(raw_yaw, RC_SAFE_MIN, RC_SAFE_MAX))

        # --- B. THROTTLE — вертикальная наводка (с рампой для плавности) ---
        # err_y > 0 = цель ниже центра = нужно снижаться
        # err_y < 0 = цель выше центра = нужно подниматься
        raw_throttle_delta = -self._pid_alt.update(err_y)
        # Плавное нарастание throttle через рампу
        throttle_delta = raw_throttle_delta * self._throttle_ramp
        raw_throttle = RC_MID + throttle_delta
        rc_throttle = int(np.clip(raw_throttle, RC_THROTTLE_MIN, RC_THROTTLE_MAX))

        # --- C. PITCH — кинетический удар (рампа пикирования) ---
        # Нормированная Y позиция цели (0 = верх кадра, 1 = низ кадра)
        norm_y = float(np.clip(cy / FRAME_HEIGHT, 0.0, 1.0))
        # Целевой питч в зависимости от вертикальной позиции цели
        target_pitch = PITCH_NEAR - (PITCH_NEAR - PITCH_DIVE) * norm_y
        # Нарастание от нейтрали (RC_MID=1500) до target_pitch через рампу
        rc_pitch_raw = RC_MID - (RC_MID - target_pitch) * self._ramp_progress
        rc_pitch = int(np.clip(rc_pitch_raw, PITCH_DIVE, PITCH_NEAR))

        # --- D. ROLL — всегда passthrough (оператор управляет креном) ---
        rc_roll = RC_RELEASE

        # --- E. Rate limiting — плавность RC без рывков (MAX_RC_DELTA_PER_FRAME) ---
        rc_yaw      = _rate_limit(rc_yaw,      self._prev_yaw)
        rc_throttle = _rate_limit(rc_throttle, self._prev_throttle)
        rc_pitch    = _rate_limit(rc_pitch,    self._prev_pitch)

        self._prev_yaw      = rc_yaw
        self._prev_throttle = rc_throttle
        self._prev_pitch    = rc_pitch

        if self._ramp_progress >= 1.0:
            self._state = TrackerState.STRIKING
        else:
            self._state = TrackerState.TRACKING

        return TrackResult(
            state         = self._state,
            target_x      = cx,
            target_y      = cy,
            lead_x        = lead_x,
            lead_y        = lead_y,
            confidence    = conf,
            err_x         = err_x,
            err_y         = err_y,
            rc_roll       = rc_roll,
            rc_pitch      = rc_pitch,
            rc_throttle   = rc_throttle,
            rc_yaw        = rc_yaw,
            ramp_progress = self._ramp_progress,
            throttle_ramp = self._throttle_ramp,
        )

    def _handle_lost(self) -> TrackResult:
        """Обработка потери цели — dead reckoning → LOST."""
        if self._state in (TrackerState.TRACKING, TrackerState.STRIKING,
                           TrackerState.DEAD_RECKON):
            self._state = TrackerState.DEAD_RECKON
            # Dead reckoning: удерживаем последние RC значения
            # VisionTracker уже возвращает Kalman predict на DEAD_RECKONING_SEC
            # Если vision_result=None после DEAD_RECKONING_SEC → LOST
            logger.debug("TrackerEngine: DEAD_RECKON")

        elapsed = time.monotonic() - self._ramp_start
        if elapsed > RAMP_DURATION_SEC + DEAD_RECKONING_SEC:
            self._state = TrackerState.LOST
            logger.warning("TrackerEngine: LOST — цель потеряна окончательно")

        return TrackResult(
            state         = self._state,
            ramp_progress = self._ramp_progress,
            rc_roll       = RC_RELEASE,
            rc_pitch      = RC_RELEASE,
            rc_throttle   = RC_RELEASE,
            rc_yaw        = RC_RELEASE,
        )