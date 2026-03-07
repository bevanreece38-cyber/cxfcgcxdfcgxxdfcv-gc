"""
VisionTracker — двойное машинное зрение (PixEagle архитектура).

https://github.com/alireza787b/PixEagle

Принципы PixEagle:
  1. YOLO (NPU, каждые N кадров) — детекция объектов
  2. OpenCV CSRT (CPU, каждый кадр) — трекинг между детекциями
  3. Kalman фильтр — сглаживание + velocity → predictive lead point
  4. При каждой YOLO детекции → переинициализация CSRT (коррекция дрейфа)

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
    CONF_THRESHOLD, YOLO_EVERY_N_FRAMES,
    TRACKER_TYPE, TRACKER_REINIT_ON_YOLO,
    DEAD_RECKONING_SEC, LEAD_TIME_SEC, LEAD_FACTOR,
)
from kalman import KalmanTargetTracker

logger = logging.getLogger(__name__)

# Тип результата: (cx, cy, conf, bbox(x1,y1,x2,y2)) или None
VisionResult = Optional[Tuple[float, float, float, Tuple[int, int, int, int]]]


def _create_opencv_tracker() -> cv2.Tracker:
    """Создать OpenCV трекер с fallback цепочкой: CSRT → KCF → MIL."""
    t = TRACKER_TYPE.upper()

    # Пробуем создать запрошенный тип, затем fallback
    candidates = [t] if t in ("CSRT", "KCF") else ["CSRT"]
    if "KCF" not in candidates:
        candidates.append("KCF")
    candidates.append("MIL")   # последний fallback

    for name in candidates:
        # Сначала пробуем прямой атрибут (opencv-contrib или новые версии)
        factory_direct = getattr(cv2, f"Tracker{name}_create", None)
        if factory_direct is not None:
            try:
                tracker = factory_direct()
                if name != t:
                    logger.warning(f"Трекер '{t}' недоступен, используем {name}")
                return tracker
            except Exception:
                pass

        # Затем пробуем через cv2.legacy (opencv-contrib ≥4.5)
        legacy = getattr(cv2, "legacy", None)
        if legacy is not None:
            factory_legacy = getattr(legacy, f"Tracker{name}_create", None)
            if factory_legacy is not None:
                try:
                    tracker = factory_legacy()
                    if name != t:
                        logger.warning(
                            f"Трекер '{t}' недоступен, используем legacy.{name}"
                        )
                    return tracker
                except Exception:
                    pass

    raise RuntimeError("Не удалось создать ни один OpenCV трекер (CSRT/KCF/MIL)")


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
        self._tracking   = False          # CSRT активен
        self._frame_cnt  = 0              # счётчик кадров для YOLO throttle
        self._last_seen  = 0.0            # время последней успешной детекции
        self._last_bbox: Optional[Tuple[int, int, int, int]] = None

    def reset(self):
        """Сбросить все состояния трекера."""
        self._kalman.reset()
        self._cv_tracker = None
        self._tracking   = False
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

        # --- Шаг 2: Переинициализация CSRT при YOLO детекции ---
        if best_detection is not None:
            cx, cy, conf, bbox = best_detection
            if TRACKER_REINIT_ON_YOLO or not self._tracking:
                self._init_csrt(frame, bbox)
            self._last_seen = time.time()
            self._last_bbox = bbox
            # Обновляем Kalman с реальным измерением
            self._kalman.update((cx, cy))
            return (cx, cy, conf, bbox)

        # --- Шаг 3: CSRT трекинг (между YOLO детекциями) ---
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
                return (cx, cy, 0.0, (x1, y1, x2, y2))
            else:
                # CSRT потерял цель
                self._tracking = False
                logger.debug("CSRT: цель потеряна")

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
        Вычислить точку упреждения на основе velocity из Kalman.
        Используется TrackerEngine для predictive intercept.

        Возвращает (lead_x, lead_y) в пикселях.
        """
        x, y, vx, vy = self._kalman.predict_with_velocity()
        # Прогноз на LEAD_TIME_SEC * MAX_FPS кадров вперёд
        frames_ahead = LEAD_TIME_SEC * 30.0  # 30 FPS
        lead_x = x + vx * frames_ahead * LEAD_FACTOR
        lead_y = y + vy * frames_ahead * LEAD_FACTOR
        # Клампим в пределах кадра
        lead_x = float(np.clip(lead_x, 0, FRAME_WIDTH))
        lead_y = float(np.clip(lead_y, 0, FRAME_HEIGHT))
        return (lead_x, lead_y)

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

    def _init_csrt(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Инициализировать CSRT трекер с заданным bbox."""
        try:
            x1, y1, x2, y2 = bbox
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            self._cv_tracker = _create_opencv_tracker()
            self._cv_tracker.init(frame, (x1, y1, w, h))
            self._tracking = True
            logger.debug(f"CSRT инициализирован: bbox=({x1},{y1},{x2},{y2})")
        except Exception as e:
            logger.error(f"CSRT init error: {e}")
            self._tracking = False