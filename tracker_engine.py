"""
TrackerEngine — алгоритм перехвата цели (PixEagle архитектура).

Поддерживаемые направления атаки (6 DOF):
  Горизонталь:  цель справа → yaw вправо, цель слева → yaw влево
  Вертикаль:    цель выше → throttle вверх, цель ниже → throttle вниз + pitch
  Диагонали:    любая комбинация (дрон выходит на траекторию перехвата)
  Позади выше:  yaw разворот + throttle вверх + roll assist
  Позади ниже:  yaw разворот + pitch пикирование + roll assist

Плавное ускорение:
  - Throttle нарастает через THROTTLE_RAMP_SEC (нет рывков)
  - Pitch нарастает через RAMP_DURATION_SEC
  - STRIKING: throttle до RC_THROTTLE_STRIKING (1800) для максимального ускорения
  - Roll assist: при |err_x| > ROLL_ASSIST_THRESHOLD → крен для быстрого разворота

State машина:
  IDLE → engage() → ACQUIRING → (детекция) → TRACKING
  TRACKING → (потеря <0.25с) → DEAD_RECKON → REACQUIRE → LOST
  REACQUIRE: манёвр по Kalman vx,vy пока цель не найдена или таймаут
  TRACKING → (рампа 100%) → STRIKING
  LOST / disengage() → IDLE
"""

import time
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from config import (
    FRAME_WIDTH, FRAME_HEIGHT,
    RC_MID, RC_RELEASE, RC_SAFE_MIN, RC_SAFE_MAX,
    RC_THROTTLE_MIN, RC_THROTTLE_MAX, RC_THROTTLE_STRIKING,
    ROLL_ASSIST_THRESHOLD, ROLL_ASSIST_MIN, ROLL_ASSIST_MAX,
    PITCH_NEAR, PITCH_DIVE,
    RAMP_DURATION_SEC, THROTTLE_RAMP_SEC,
    DEAD_RECKONING_SEC, REACQUIRE_TIMEOUT,
    REACQUIRE_VELOCITY_DECAY, REACQUIRE_CONFIRM_SEC,
    MAX_FPS,
    KP_YAW, KI_YAW, KD_YAW,
    KP_ALT, KI_ALT, KD_ALT,
    TARGET_CLASS_ID, CONF_THRESHOLD,
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
        self._throttle_ramp = 0.0
        self._throttle_ramp_start = 0.0
        self._dead_reckon_start   = 0.0   # момент потери цели (DEAD_RECKON таймер)
        self._reacquire_start     = 0.0   # момент входа в REACQUIRE
        self._reacquire_pos: Tuple[float, float] = (0.0, 0.0)
        self._reacquire_vel: Tuple[float, float] = (0.0, 0.0)
        # Recovery confirmation (PixEagle паттерн):
        # цель должна стабильно отслеживаться REACQUIRE_CONFIRM_SEC до возврата в TRACKING
        self._reacquire_confirm_start: float = 0.0

    @property
    def state(self) -> TrackerState:
        return self._state

    def engage(self):
        """Начать перехват. Сброс всех состояний."""
        self._vision.reset()
        self._pid_yaw.reset()
        self._pid_alt.reset()
        self._ramp_start              = time.monotonic()
        self._ramp_progress           = 0.0
        self._throttle_ramp           = 0.0
        self._throttle_ramp_start     = time.monotonic()
        self._dead_reckon_start       = 0.0
        self._reacquire_start         = 0.0
        self._reacquire_pos           = (0.0, 0.0)
        self._reacquire_vel           = (0.0, 0.0)
        self._reacquire_confirm_start = 0.0
        self._state = TrackerState.ACQUIRING
        logger.info("TrackerEngine: ENGAGE → ACQUIRING")

    def disengage(self):
        """Остановить перехват. Возврат в IDLE."""
        self._vision.reset()
        self._pid_yaw.reset()
        self._pid_alt.reset()
        self._ramp_progress           = 0.0
        self._throttle_ramp           = 0.0
        self._dead_reckon_start       = 0.0
        self._reacquire_start         = 0.0
        self._reacquire_confirm_start = 0.0
        self._state = TrackerState.IDLE
        logger.info("TrackerEngine: DISENGAGE → IDLE")

    def _compute_roll_assist(self, err_x: float) -> int:
        """
        Roll assist для быстрого горизонтального разворота.

        Включается только когда |err_x| > ROLL_ASSIST_THRESHOLD (150 px).
        Линейно нарастает от RC_MID до ROLL_ASSIST_MAX/MIN.
        При err_x в диапазоне [-threshold, +threshold] → RC_RELEASE (оператор управляет).

        err_x > 0 → цель справа → крен вправо  (rc_roll > RC_MID)
        err_x < 0 → цель слева  → крен влево   (rc_roll < RC_MID)
        """
        if abs(err_x) <= ROLL_ASSIST_THRESHOLD:
            return RC_RELEASE
        max_range = max(FRAME_WIDTH / 2.0 - ROLL_ASSIST_THRESHOLD, 1.0)
        factor = min((abs(err_x) - ROLL_ASSIST_THRESHOLD) / max_range, 1.0)
        if err_x > 0:
            return int(RC_MID + (ROLL_ASSIST_MAX - RC_MID) * factor)
        else:
            return int(RC_MID - (RC_MID - ROLL_ASSIST_MIN) * factor)

    def step(self, yolo_outputs, frame: np.ndarray) -> TrackResult:
        """
        Один шаг трекинга и управления.

        Args:
            yolo_outputs: результат YOLO post_process() или None
            frame:        BGR кадр (640×480) для CSRT/KCF

        Returns:
            TrackResult с RC значениями для ControlManager
        """
        if self._state == TrackerState.IDLE:
            return _idle_result()

        # VisionTracker: YOLO + CSRT/KCF + Kalman
        # frame — первый аргумент
        vision_result = self._vision.step(frame, yolo_outputs)

        if vision_result is None:
            # Нет детекции → потеря цели
            self._reacquire_confirm_start = 0.0   # сброс подтверждения recovery
            return self._handle_lost()

        # --- Цель найдена ---
        # Recovery confirmation (PixEagle паттерн):
        # При возврате из REACQUIRE требуем стабильного трекинга REACQUIRE_CONFIRM_SEC
        if self._state == TrackerState.REACQUIRE:
            now = time.monotonic()
            if self._reacquire_confirm_start == 0.0:
                # Первый кадр возврата — начинаем отсчёт подтверждения
                self._reacquire_confirm_start = now
                logger.debug("TrackerEngine: REACQUIRE — начало recovery confirmation")
            elif (now - self._reacquire_confirm_start) < REACQUIRE_CONFIRM_SEC:
                # Ещё не подтверждено — ждём стабильности REACQUIRE_CONFIRM_SEC.
                # Состояние остаётся REACQUIRE; RC-команды считаются ниже
                # с _in_reacquire_confirm=True чтобы не перезаписать состояние.
                pass
            else:
                # Подтверждено: цель стабильно отслеживается REACQUIRE_CONFIRM_SEC
                self._state = TrackerState.TRACKING
                self._reacquire_confirm_start = 0.0
                logger.info(f"TrackerEngine: REACQUIRE → TRACKING (подтверждено "
                            f"{REACQUIRE_CONFIRM_SEC}с стабильного трекинга)")
        elif self._state == TrackerState.DEAD_RECKON:
            # Возврат из DEAD_RECKON — сразу восстанавливаем TRACKING (быстрый переход)
            self._state = TrackerState.TRACKING

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
        throttle_delta = raw_throttle_delta * self._throttle_ramp
        raw_throttle   = RC_MID + throttle_delta

        # --- C. PITCH — кинетический удар (рампа пикирования) ---
        norm_y = float(np.clip(cy / FRAME_HEIGHT, 0.0, 1.0))
        target_pitch = PITCH_NEAR - (PITCH_NEAR - PITCH_DIVE) * norm_y
        rc_pitch_raw = RC_MID - (RC_MID - target_pitch) * self._ramp_progress
        rc_pitch = int(np.clip(rc_pitch_raw, PITCH_DIVE, PITCH_NEAR))

        # --- D. ROLL — assist для быстрого разворота при большом err_x ---
        # При |err_x| <= ROLL_ASSIST_THRESHOLD → RC_RELEASE (оператор управляет)
        # При |err_x|  > ROLL_ASSIST_THRESHOLD → пропорциональный крен
        rc_roll = self._compute_roll_assist(err_x)

        # --- Состояние + throttle cap ---
        # _in_reacquire_confirm: цель найдена, но подтверждение ещё не завершено.
        # В этом случае НЕ переводим в TRACKING/STRIKING — ждём REACQUIRE_CONFIRM_SEC.
        # После подтверждения состояние уже переведено в TRACKING выше (строки 197–202).
        _in_reacquire_confirm = (
            self._state == TrackerState.REACQUIRE
            and self._reacquire_confirm_start > 0.0
        )

        if self._ramp_progress >= 1.0 and not _in_reacquire_confirm:
            self._state = TrackerState.STRIKING
            rc_throttle = int(np.clip(raw_throttle, RC_THROTTLE_MIN, RC_THROTTLE_STRIKING))
        elif not _in_reacquire_confirm:
            # TRACKING (нормальный полёт) или только что подтверждённый TRACKING
            self._state = TrackerState.TRACKING
            rc_throttle = int(np.clip(raw_throttle, RC_THROTTLE_MIN, RC_THROTTLE_MAX))
        else:
            # REACQUIRE подтверждение в процессе: безопасный throttle, состояние = REACQUIRE
            rc_throttle = int(np.clip(raw_throttle, RC_THROTTLE_MIN, RC_THROTTLE_MAX))

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
        """
        Обработка потери цели.

        Цепочка переходов:
          TRACKING / STRIKING → DEAD_RECKON  (фиксируем _dead_reckon_start)
          DEAD_RECKON          → REACQUIRE   (через DEAD_RECKONING_SEC)
          REACQUIRE            → LOST        (через REACQUIRE_TIMEOUT)
          ACQUIRING            → LOST        (через RAMP_DURATION_SEC+DEAD_RECKONING_SEC)

        REACQUIRE — продолжение манёвра по последнему вектору Kalman vx,vy:
          Дрон продолжает лететь в направлении последней скорости цели,
          применяя yaw и roll-assist, пока цель не будет найдена снова.

        КРИТИЧНО — RC значения при DEAD_RECKON / REACQUIRE:
          RC_MID (1500) для pitch/throttle, НЕ RC_RELEASE (65535).
          RC_RELEASE = UINT16_MAX = ArduPilot игнорирует поле → старый override
          (PITCH_DIVE=1280) остаётся активным → дрон продолжает пикировать!
        """
        now = time.monotonic()

        # TRACKING / STRIKING → DEAD_RECKON
        if self._state in (TrackerState.TRACKING, TrackerState.STRIKING):
            self._dead_reckon_start = now
            self._state = TrackerState.DEAD_RECKON
            logger.debug("TrackerEngine: → DEAD_RECKON (таймер старт)")

        # DEAD_RECKON → REACQUIRE (после DEAD_RECKONING_SEC)
        elif self._state == TrackerState.DEAD_RECKON:
            if (now - self._dead_reckon_start) >= DEAD_RECKONING_SEC:
                vx, vy = self._vision.get_velocity()
                lx, ly = self._vision.get_lead_point()
                self._reacquire_start = now
                self._reacquire_pos   = (lx, ly)
                self._reacquire_vel   = (vx, vy)
                self._state = TrackerState.REACQUIRE
                logger.info("TrackerEngine: DEAD_RECKON → REACQUIRE "
                            f"(vx={vx:.1f} vy={vy:.1f} px/frame)")

        # ACQUIRING timeout → LOST
        elif self._state == TrackerState.ACQUIRING:
            if (now - self._ramp_start) > RAMP_DURATION_SEC + DEAD_RECKONING_SEC:
                self._state = TrackerState.LOST
                logger.warning("TrackerEngine: ACQUIRING timeout → LOST")

        # REACQUIRE: манёвр по Kalman velocity с затуханием (PixEagle паттерн)
        if self._state == TrackerState.REACQUIRE:
            elapsed = now - self._reacquire_start
            if elapsed >= REACQUIRE_TIMEOUT:
                self._state = TrackerState.LOST
                logger.warning("TrackerEngine: REACQUIRE → LOST (таймер истёк)")
            else:
                # Velocity decay (PixEagle ENABLE_VELOCITY_DECAY паттерн):
                # vx_now = vx_at_loss * decay_rate^elapsed_sec
                # 0.85^1 = 85%, 0.85^2 = 72%, 0.85^3 = 61%
                # Предотвращает уход дрона при длительной потере цели
                decay = REACQUIRE_VELOCITY_DECAY ** elapsed
                vx_decayed = self._reacquire_vel[0] * decay
                vy_decayed = self._reacquire_vel[1] * decay

                # Экстраполяция позиции цели по затухающей скорости.
                # frames_elapsed — приближение: предполагается постоянный MAX_FPS.
                # При реальном FPS, отличном от MAX_FPS, погрешность пропорциональна
                # разнице скоростей — допустимо в пределах REACQUIRE_TIMEOUT=1.5с.
                frames_elapsed = elapsed * MAX_FPS
                pred_x = float(np.clip(
                    self._reacquire_pos[0] + vx_decayed * frames_elapsed, 0.0, FRAME_WIDTH
                ))
                pred_y = float(np.clip(
                    self._reacquire_pos[1] + vy_decayed * frames_elapsed, 0.0, FRAME_HEIGHT
                ))
                err_x = pred_x - FRAME_WIDTH  / 2.0
                err_y = pred_y - FRAME_HEIGHT / 2.0
                raw_yaw = RC_MID + self._pid_yaw.update(err_x)
                rc_yaw  = int(np.clip(raw_yaw, RC_SAFE_MIN, RC_SAFE_MAX))
                rc_roll = self._compute_roll_assist(err_x)
                return TrackResult(
                    state         = TrackerState.REACQUIRE,
                    lead_x        = pred_x,
                    lead_y        = pred_y,
                    err_x         = err_x,
                    err_y         = err_y,
                    rc_roll       = rc_roll,
                    rc_pitch      = RC_MID,
                    rc_throttle   = RC_MID,
                    rc_yaw        = rc_yaw,
                    ramp_progress = self._ramp_progress,
                )

        # Безопасные нейтрали для DEAD_RECKON (в окне), ACQUIRING (в окне), LOST
        return TrackResult(
            state         = self._state,
            ramp_progress = self._ramp_progress,
            rc_roll       = RC_RELEASE,   # оператор управляет креном
            rc_pitch      = RC_MID,       # нейтральный pitch — стоп пикирование
            rc_throttle   = RC_MID,       # hover throttle
            rc_yaw        = RC_MID,       # нейтральное рыскание
        )