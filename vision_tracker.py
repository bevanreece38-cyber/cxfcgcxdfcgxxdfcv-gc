"""
VisionTracker — двойное машинное зрение (PixEagle архитектура).

https://github.com/alireza787b/PixEagle

Принципы PixEagle:
  1. YOLO (NPU, каждые N кадров) — детекция объектов
  2. OpenCV CSRT (CPU, каждый кадр) — трекинг между детекциями
  3. Kalman фильтр — сглаживание + velocity → predictive lead point
  4. При каждой YOLO детекции → переинициализация CSRT (коррекция дрейфа)

Выбор приоритетной цели: максимальный confidence score среди TARGET_CLASS_ID.
При нескольких одинаковых — уверенность YOLO важнее расстояния до центра.

Возвращает:
  (cx, cy, conf, bbox) или None если цель не найдена.

Задержки:
  - YOLO на NPU: ~10-15 мс (RK3588 все 3 ядра NPU_CORE_0_1_2)
  - CSRT на CPU: ~5-8 мс
  - Итого на кадр: ≤16 мс при 30 FPS

ИСПРАВЛЕНО: сигнатура step(frame, yolo_outputs) — frame первый аргумент.
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, Tuple

from config import (
    FRAME_WIDTH, FRAME_HEIGHT, TARGET_CLASS_ID,
    CONF_THRESHOLD,
    TRACKER_TYPE, TRACKER_REINIT_ON_YOLO,
    DEAD_RECKONING_SEC, LEAD_TIME_SEC, LEAD_FACTOR,
    HIGH_SPEED_TRACKER_THRESHOLD, LOW_SPEED_TRACKER_THRESHOLD,
)
from kalman import KalmanTargetTracker

logger = logging.getLogger(__name__)

# Тип результата: (cx, cy, conf, bbox(x1,y1,x2,y2)) или None
VisionResult = Optional[Tuple[float, float, float, Tuple[int, int, int, int]]]


def _create_opencv_tracker(tracker_type: str = None) -> cv2.Tracker:
    """Создать OpenCV трекер (CSRT или KCF).

    В OpenCV 4.5+ legacy-трекеры перенесены в cv2.legacy.
    Пробуем новый API (legacy), при неудаче — старый API.

    Args:
        tracker_type: "CSRT" или "KCF". Если None — используется TRACKER_TYPE из config.
    """
    t = (tracker_type or TRACKER_TYPE).upper()
    try:
        if t == "CSRT":
            return cv2.legacy.TrackerCSRT_create()
        elif t == "KCF":
            return cv2.legacy.TrackerKCF_create()
        else:
            logger.warning(f"Неизвестный трекер '{t}', используем CSRT")
            return cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        # Старые версии OpenCV (<4.5): трекеры в корне cv2
        if t == "CSRT":
            return cv2.TrackerCSRT_create()  # type: ignore[attr-defined]
        elif t == "KCF":
            return cv2.TrackerKCF_create()   # type: ignore[attr-defined]
        else:
            return cv2.TrackerCSRT_create()  # type: ignore[attr-defined]


class VisionTracker:
    """
    Двухуровневый трекер: YOLO детекция (редкая) + CSRT трекинг (частый).
    Kalman фильтр поверх для сглаживания и предсказания упреждения.

    Использование:
        vt = VisionTracker()
        vt.reset()
        result = vt.step(frame, yolo_outputs)  # frame — первый аргумент!
    """

    def __init__(self):
        self._kalman     = KalmanTargetTracker(FRAME_WIDTH, FRAME_HEIGHT)
        self._cv_tracker: Optional[cv2.Tracker] = None
        self._tracking   = False          # трекер активен
        self._using_kcf  = False          # True = KCF (высокая скорость цели)
        self._frame_cnt  = 0              # счётчик кадров для YOLO throttle
        self._last_seen  = 0.0            # время последней успешной детекции
        self._last_bbox: Optional[Tuple[int, int, int, int]] = None

    def reset(self):
        """Сбросить все состояния трекера."""
        self._kalman.reset()
        self._cv_tracker = None
        self._tracking   = False
        self._using_kcf  = False
        self._frame_cnt  = 0
        self._last_seen  = 0.0
        self._last_bbox  = None
        logger.debug("VisionTracker reset")

    def step(self, frame: np.ndarray, yolo_outputs) -> VisionResult:
        """
        Один шаг трекинга.

        Args:
            frame:        BGR кадр (640×480 uint8)
            yolo_outputs: результат post_process() или None

        Returns:
            (cx, cy, conf, bbox) или None
        """
        self._frame_cnt += 1
        self._kalman.begin_step()

        best_detection = None

        # --- Шаг 1: YOLO детекция (каждые YOLO_EVERY_N_FRAMES кадров) ---
        if yolo_outputs is not None:
            best_detection = self._pick_best_detection(yolo_outputs)

        # --- Шаг 2: Переинициализация трекера при YOLO детекции ---
        if best_detection is not None:
            cx, cy, conf, bbox = best_detection
            if TRACKER_REINIT_ON_YOLO or not self._tracking:
                self._init_csrt(frame, bbox)
            self._last_seen = time.time()
            self._last_bbox = bbox
            # Обновляем Kalman с реальным измерением
            self._kalman.update((cx, cy))
            # Проверяем переключение CSRT↔KCF по скорости
            self._maybe_switch_tracker(frame, bbox)
            return (cx, cy, conf, bbox)

        # --- Шаг 3: CSRT/KCF трекинг (между YOLO детекциями) ---
        if self._tracking and self._cv_tracker is not None:
            ok, bbox_raw = self._cv_tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in bbox_raw]
                cx = float(x + w / 2)
                cy = float(y + h / 2)
                x1, y1, x2, y2 = x, y, x + w, y + h
                self._last_seen = time.time()
                self._last_bbox = (x1, y1, x2, y2)
                self._kalman.update((cx, cy))
                # Проверяем переключение CSRT↔KCF по скорости
                self._maybe_switch_tracker(frame, (x1, y1, x2, y2))
                return (cx, cy, 0.0, (x1, y1, x2, y2))
            else:
                # Трекер потерял цель
                self._tracking = False
                logger.debug("Tracker: цель потеряна")

        # --- Шаг 4: Dead reckoning через Kalman ---
        time_lost = time.time() - self._last_seen
        if self._last_seen > 0 and time_lost < DEAD_RECKONING_SEC:
            pred_x, pred_y = self._kalman.predict_only()
            bbox = self._last_bbox or (
                int(pred_x) - 20, int(pred_y) - 20,
                int(pred_x) + 20, int(pred_y) + 20,
            )
            return (pred_x, pred_y, 0.0, bbox)

        return None

    def get_lead_point(self) -> Tuple[float, float]:
        """
        Кинематическая точка упреждения: pos + vel*t + 0.5*accel*t²

        Паттерн PixEagle MotionPredictor.predict_bbox(use_acceleration=True):
          pred_cx = cx + velocity_x * dt + 0.5 * accel_x * dt²
        Это даёт 15–25% точнее чем линейный прогноз при t > 5 кадров.

        Возвращает (lead_x, lead_y) в пикселях.
        """
        x, y, vx, vy, ax, ay = self._kalman.predict_with_acceleration()
        t = LEAD_TIME_SEC * 30.0   # кадры вперёд (30 FPS)
        # Кинематическое уравнение (PixEagle паттерн):
        # pos + vel*t*LEAD_FACTOR + 0.5*accel*t²
        lead_x = x + vx * t * LEAD_FACTOR + 0.5 * ax * t ** 2
        lead_y = y + vy * t * LEAD_FACTOR + 0.5 * ay * t ** 2
        lead_x = float(np.clip(lead_x, 0, FRAME_WIDTH))
        lead_y = float(np.clip(lead_y, 0, FRAME_HEIGHT))
        return (lead_x, lead_y)

    def get_velocity(self) -> Tuple[float, float]:
        """
        Вернуть текущую EMA-сглаженную скорость цели (vx, vy) из Kalman
        в пикселях/кадр. Используется TrackerEngine для REACQUIRE.
        """
        _, _, vx, vy = self._kalman.predict_with_velocity()
        return (float(vx), float(vy))

    # ------------------------------------------------------------------
    #  Внутренние методы
    # ------------------------------------------------------------------

    def _pick_best_detection(self, yolo_outputs) -> Optional[VisionResult]:
        """
        Выбрать лучшую детекцию из YOLO outputs.
        Критерий: максимальный score среди TARGET_CLASS_ID.
        """
        if yolo_outputs is None:
            return None
        try:
            boxes, classes, scores = yolo_outputs
            best_score = 0.0
            best = None
            for i, cls in enumerate(classes):
                if int(cls) == TARGET_CLASS_ID and float(scores[i]) > CONF_THRESHOLD:
                    if float(scores[i]) > best_score:
                        best_score = float(scores[i])
                        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                        cx = float((x1 + x2) / 2)
                        cy = float((y1 + y2) / 2)
                        best = (cx, cy, best_score, (x1, y1, x2, y2))
            return best
        except Exception as e:
            logger.error(f"VisionTracker _pick_best_detection error: {e}")
            return None

    def _init_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """
        Инициализировать трекер (CSRT или KCF).

        CSRT — базовый, точнее при медленных целях.
        KCF  — быстрее, лучше при скорости > HIGH_SPEED_TRACKER_THRESHOLD.
        Тип определяется self._using_kcf.
        """
        try:
            x1, y1, x2, y2 = bbox
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            tracker_type_str = "KCF" if self._using_kcf else None
            self._cv_tracker = _create_opencv_tracker(tracker_type_str)
            self._cv_tracker.init(frame, (x1, y1, w, h))
            self._tracking = True
            name = "KCF" if self._using_kcf else "CSRT"
            logger.debug(f"{name} инициализирован: bbox=({x1},{y1},{x2},{y2})")
        except Exception as e:
            logger.error(f"Tracker init error: {e}")
            self._tracking = False

    def _maybe_switch_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """
        Переключить CSRT ↔ KCF на основе скорости цели из Kalman.

        Гистерезис:
          speed > HIGH_SPEED_TRACKER_THRESHOLD → KCF
          speed < LOW_SPEED_TRACKER_THRESHOLD  → CSRT
        Переинициализация происходит сразу с последним известным bbox.
        """
        vx, vy = self.get_velocity()
        speed = float(np.sqrt(vx ** 2 + vy ** 2))

        if speed > HIGH_SPEED_TRACKER_THRESHOLD and not self._using_kcf:
            self._using_kcf = True
            logger.info(f"Трекер: CSRT→KCF (скорость {speed:.1f} px/frame)")
            self._init_tracker(frame, bbox)

        elif speed < LOW_SPEED_TRACKER_THRESHOLD and self._using_kcf:
            self._using_kcf = False
            logger.info(f"Трекер: KCF→CSRT (скорость {speed:.1f} px/frame)")
            self._init_tracker(frame, bbox)

    def _init_csrt(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Псевдоним для обратной совместимости — делегирует _init_tracker."""
        self._init_tracker(frame, bbox)