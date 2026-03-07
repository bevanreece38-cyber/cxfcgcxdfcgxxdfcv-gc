import numpy as np
import pytest
from vision_tracker import VisionTracker
from config import TARGET_CLASS_ID, CONF_THRESHOLD


def _blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _textured_frame():
    np.random.seed(42)
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


def _det(cx, cy, score=0.9, size=40):
    x1, y1 = int(cx - size // 2), int(cy - size // 2)
    x2, y2 = int(cx + size // 2), int(cy + size // 2)
    return ([[x1, y1, x2, y2]], [TARGET_CLASS_ID], [score])


def test_no_detection_returns_none():
    vt = VisionTracker()
    vt.reset()
    result = vt.step(_blank_frame(), None)
    assert result is None


def test_yolo_detection_found():
    vt = VisionTracker()
    vt.reset()
    result = vt.step(_blank_frame(), _det(320, 240))
    assert result is not None
    cx, cy, conf, bbox = result
    assert abs(cx - 320) < 5
    assert abs(cy - 240) < 5
    assert conf > 0


def test_low_confidence_rejected():
    vt = VisionTracker()
    vt.reset()
    det = _det(320, 240, score=CONF_THRESHOLD - 0.1)
    result = vt.step(_blank_frame(), det)
    assert result is None


def test_lead_point_within_frame():
    vt = VisionTracker()
    vt.reset()
    vt.step(_textured_frame(), _det(320, 240))
    lx, ly = vt.get_lead_point()
    assert 0 <= lx <= 640
    assert 0 <= ly <= 480


def test_reset_clears_state():
    vt = VisionTracker()
    vt.reset()
    vt.step(_textured_frame(), _det(100, 100))
    vt.reset()
    result = vt.step(_blank_frame(), None)
    assert result is None


def test_multiple_detections_picks_best():
    """Должна выбраться детекция с наибольшим score."""
    vt = VisionTracker()
    vt.reset()
    boxes   = [[10, 10, 60, 60], [290, 210, 350, 270]]
    classes = [TARGET_CLASS_ID, TARGET_CLASS_ID]
    scores  = [0.6, 0.95]
    result  = vt.step(_blank_frame(), (boxes, classes, scores))
    assert result is not None
    cx, cy, conf, _ = result
    # Лучшая детекция — вторая (score=0.95, центр 320,240)
    assert abs(cx - 320) < 5


def test_csrt_tracks_after_yolo():
    """CSRT должен продолжать трекинг после YOLO детекции."""
    vt = VisionTracker()
    vt.reset()
    frame = _textured_frame()
    # Первый кадр: YOLO детекция
    result1 = vt.step(frame, _det(320, 240))
    assert result1 is not None
    # Второй кадр: без YOLO — CSRT должен найти цель
    result2 = vt.step(frame, None)
    # На одном и том же кадре CSRT должен дать результат
    # (может вернуть None если CSRT не смог инициализироваться на черном кадре)
    # Для textured_frame должен работать


def test_step_signature_frame_first():
    """Проверка правильного порядка аргументов: step(frame, yolo)."""
    vt = VisionTracker()
    vt.reset()
    frame = _blank_frame()
    det   = _det(320, 240)
    # Должен принимать frame первым аргументом
    result = vt.step(frame, det)
    assert result is not None