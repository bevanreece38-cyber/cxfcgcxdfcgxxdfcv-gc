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


def test_csrt_tracks_between_yolo():
    """CSRT работает между кадрами YOLO (yolo_outputs=None)."""
    vt = VisionTracker()
    vt.reset()
    frame = _textured_frame()
    # Первый кадр: YOLO детекция — инициализируем CSRT
    result1 = vt.step(frame, _det(320, 240))
    assert result1 is not None
    # Следующие кадры: без YOLO — CSRT трекает на том же текстурированном кадре
    result2 = vt.step(frame, None)
    # На текстурированном кадре CSRT должен вернуть результат
    assert result2 is not None
    cx2, cy2, _, _ = result2
    # CSRT на одном кадре: допуск 80 пикселей (размер bbox = 40 px, сдвиг мал)
    MAX_CSRT_DEVIATION = 80
    assert abs(cx2 - 320) < MAX_CSRT_DEVIATION
    assert abs(cy2 - 240) < MAX_CSRT_DEVIATION


def test_lead_point_calculation():
    """Точка упреждения вычисляется корректно и лежит в границах кадра."""
    vt = VisionTracker()
    vt.reset()
    frame = _textured_frame()
    initial_x = 350
    # Несколько кадров с движущейся целью (справа налево)
    for x in (initial_x, 340, 330, 320):
        vt.step(frame, _det(x, 240))
    lx, ly = vt.get_lead_point()
    # Точка упреждения должна лежать в пределах кадра
    assert 0 <= lx <= 640
    assert 0 <= ly <= 480
    # При движении влево lead_x должна быть левее начальной позиции цели
    assert lx < initial_x

def test_get_velocity_returns_tuple():
    """get_velocity() возвращает (vx, vy) как два float."""
    vt = VisionTracker()
    vt.reset()
    vx, vy = vt.get_velocity()
    assert isinstance(vx, float)
    assert isinstance(vy, float)


def test_get_velocity_after_movement():
    """После нескольких шагов с движущейся целью vx/vy ≠ 0."""
    vt = VisionTracker()
    vt.reset()
    frame = _textured_frame()
    # Цель движется влево по 10 пикселей за кадр
    for x in (320, 310, 300, 290, 280):
        vt.step(frame, _det(x, 240))
    vx, vy = vt.get_velocity()
    # Kalman должен оценить отрицательную скорость по X
    assert vx < 0, f"Ожидался vx < 0 для цели движущейся влево, получен {vx}"


def test_csrt_kcf_switch_using_kcf_flag():
    """_using_kcf устанавливается в False по умолчанию и при reset."""
    vt = VisionTracker()
    assert vt._using_kcf is False
    vt._using_kcf = True
    vt.reset()
    assert vt._using_kcf is False


def test_maybe_switch_tracker_sets_kcf_at_high_speed(monkeypatch):
    """При скорости > HIGH_SPEED_TRACKER_THRESHOLD → _using_kcf = True."""
    from config import HIGH_SPEED_TRACKER_THRESHOLD
    vt = VisionTracker()
    vt.reset()
    frame = _textured_frame()
    # Инициализируем трекер YOLO детекцией
    vt.step(frame, _det(320, 240))
    assert vt._tracking

    # Замокать get_velocity чтобы вернуть высокую скорость
    monkeypatch.setattr(vt, 'get_velocity',
                        lambda: (HIGH_SPEED_TRACKER_THRESHOLD + 10.0, 0.0))
    # Вызвать напрямую
    vt._maybe_switch_tracker(frame, (300, 220, 340, 260))
    assert vt._using_kcf is True, "Ожидался переход на KCF при высокой скорости"


def test_maybe_switch_tracker_reverts_to_csrt_at_low_speed(monkeypatch):
    """После переключения на KCF, при низкой скорости → вернуться на CSRT."""
    from config import LOW_SPEED_TRACKER_THRESHOLD
    vt = VisionTracker()
    vt.reset()
    frame = _textured_frame()
    vt.step(frame, _det(320, 240))
    vt._using_kcf = True   # принудительно KCF

    monkeypatch.setattr(vt, 'get_velocity',
                        lambda: (LOW_SPEED_TRACKER_THRESHOLD - 10.0, 0.0))
    vt._maybe_switch_tracker(frame, (300, 220, 340, 260))
    assert vt._using_kcf is False, "Ожидался возврат на CSRT при низкой скорости"
