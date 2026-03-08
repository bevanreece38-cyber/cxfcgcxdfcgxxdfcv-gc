"""
Тесты для drone_model.post_process().

Синтетические данные имитируют вывод RKNN YOLOv5 в двух форматах:
  - Формат A: один объединённый тензор [1, N, 6]  (post-sigmoid)
  - Формат B: три тензора уровней P3/P4/P5       (post-sigmoid)
"""

import numpy as np
import pytest
from drone_model import post_process

_INPUT_SHAPE = [640, 480]
_CONF = 0.5
_NMS  = 0.45


# ─────────────────────────────────────────────────────────────────────────────
#  Вспомогательные функции для построения синтетических тензоров
# ─────────────────────────────────────────────────────────────────────────────

def _single_tensor(cx, cy, w, h, obj_conf, cls_conf=0.95):
    """Формат A: [1, 1, 6] — один объект (post-sigmoid значения)."""
    return [np.array([[[cx, cy, w, h, obj_conf, cls_conf]]], dtype=np.float32)]


def _three_head_tensors(cx, cy, w, h, obj_conf, cls_conf=0.95,
                        head_idx=1, input_w=640, input_h=480):
    """
    Формат B: 3 тензора [1, na, ny, nx, 5+nc].

    Детекция кодируется в центральной ячейке выбранного уровня (head_idx).
    Значения уже post-sigmoid.

    YOLOv5 обратная формула (для центральной ячейки):
      tx = bx/stride - cx_grid + 0.5   → post-sigmoid: (tx + 0.5) / 2
      ty = by/stride - cy_grid + 0.5   → аналогично
      tw = sqrt(bw / anchor_w) / 2     → post-sigmoid
      th = sqrt(bh / anchor_h) / 2     → аналогично
    """
    from drone_model import _ANCHORS, _STRIDES, _NA
    outputs = []
    strides  = _STRIDES
    anchors  = _ANCHORS

    for i in range(3):
        stride = strides[i]
        ny = input_h // stride
        nx = input_w // stride
        # Пустой тензор: obj_conf = 0 (все ячейки пусты)
        t = np.zeros((1, _NA, ny, nx, 6), dtype=np.float32)

        if i == head_idx:
            # Кодируем детекцию в центральную ячейку, якорь 0
            anc_w, anc_h = anchors[i * _NA + 0]
            cx_grid = nx // 2
            cy_grid = ny // 2

            # Обратная YOLOv5 decode: tx = bx/stride − cx_grid + 0.5
            tx = cx / stride - cx_grid + 0.5
            ty = cy / stride - cy_grid + 0.5
            tw = np.sqrt(w / anc_w) / 2.0
            th = np.sqrt(h / anc_h) / 2.0

            # Клампируем в (0, 1) — post-sigmoid область
            t[0, 0, cy_grid, cx_grid, 0] = float(np.clip(tx, 0.0, 1.0))
            t[0, 0, cy_grid, cx_grid, 1] = float(np.clip(ty, 0.0, 1.0))
            t[0, 0, cy_grid, cx_grid, 2] = float(np.clip(tw, 0.0, 1.0))
            t[0, 0, cy_grid, cx_grid, 3] = float(np.clip(th, 0.0, 1.0))
            t[0, 0, cy_grid, cx_grid, 4] = obj_conf
            t[0, 0, cy_grid, cx_grid, 5] = cls_conf

        outputs.append(t)
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
#  Тесты — базовые случаи
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_input_returns_none():
    """Пустой список выводов → None."""
    assert post_process([], _INPUT_SHAPE, _CONF, _NMS) is None


def test_none_input_returns_none():
    """None в качестве raw_outputs → None."""
    assert post_process(None, _INPUT_SHAPE, _CONF, _NMS) is None


def test_single_tensor_detection_found():
    """Формат A: детекция выше порога → не None."""
    raw = _single_tensor(cx=320, cy=240, w=60, h=50, obj_conf=0.9)
    result = post_process(raw, _INPUT_SHAPE, conf_threshold=0.5)
    assert result is not None
    boxes, classes, scores = result
    assert len(boxes) == 1
    assert len(classes) == 1
    assert len(scores) == 1


def test_single_tensor_class_is_drone():
    """Формат A: класс всегда 0 (дрон)."""
    raw = _single_tensor(cx=320, cy=240, w=60, h=50, obj_conf=0.9)
    _, classes, _ = post_process(raw, _INPUT_SHAPE, conf_threshold=0.5)
    assert all(c == 0 for c in classes)


def test_single_tensor_box_coordinates_sane():
    """Формат A: координаты bbox в пределах кадра и x1<x2, y1<y2."""
    raw = _single_tensor(cx=320, cy=240, w=60, h=50, obj_conf=0.9)
    boxes, _, _ = post_process(raw, _INPUT_SHAPE, conf_threshold=0.5)
    x1, y1, x2, y2 = boxes[0]
    assert 0 <= x1 < x2 <= 640, f"x: {x1} < {x2}"
    assert 0 <= y1 < y2 <= 480, f"y: {y1} < {y2}"


def test_single_tensor_score_within_range():
    """Формат A: score в диапазоне (conf_threshold, 1.0]."""
    raw = _single_tensor(cx=320, cy=240, w=60, h=50, obj_conf=0.9)
    _, _, scores = post_process(raw, _INPUT_SHAPE, conf_threshold=0.5)
    assert all(0.5 <= s <= 1.0 for s in scores), f"scores={scores}"


def test_single_tensor_low_confidence_returns_none():
    """Формат A: obj_conf ниже порога → None."""
    raw = _single_tensor(cx=320, cy=240, w=60, h=50, obj_conf=0.3)
    result = post_process(raw, _INPUT_SHAPE, conf_threshold=0.5)
    assert result is None


def test_single_tensor_center_approximately_correct():
    """Формат A: центр bbox близок к заданным cx, cy."""
    raw = _single_tensor(cx=320, cy=240, w=60, h=50, obj_conf=0.9)
    boxes, _, _ = post_process(raw, _INPUT_SHAPE, conf_threshold=0.5)
    x1, y1, x2, y2 = boxes[0]
    cx_out = (x1 + x2) / 2
    cy_out = (y1 + y2) / 2
    assert abs(cx_out - 320) < 5, f"cx={cx_out}"
    assert abs(cy_out - 240) < 5, f"cy={cy_out}"


# ─────────────────────────────────────────────────────────────────────────────
#  Тесты — формат B (три тензора YOLOv5)
# ─────────────────────────────────────────────────────────────────────────────

def test_three_head_detection_found():
    """Формат B: детекция выше порога → не None."""
    raw = _three_head_tensors(cx=320, cy=240, w=80, h=60, obj_conf=0.9)
    result = post_process(raw, _INPUT_SHAPE, conf_threshold=0.5)
    assert result is not None


def test_three_head_class_is_drone():
    """Формат B: класс всегда 0 (дрон)."""
    raw = _three_head_tensors(cx=320, cy=240, w=80, h=60, obj_conf=0.9)
    result = post_process(raw, _INPUT_SHAPE, conf_threshold=0.5)
    assert result is not None
    _, classes, _ = result
    assert all(c == 0 for c in classes)


def test_three_head_low_confidence_returns_none():
    """Формат B: все ячейки obj_conf=0 → None."""
    raw = _three_head_tensors(cx=320, cy=240, w=80, h=60, obj_conf=0.0)
    result = post_process(raw, _INPUT_SHAPE, conf_threshold=0.5)
    assert result is None


def test_three_head_score_above_threshold():
    """Формат B: score ≥ conf_threshold."""
    raw = _three_head_tensors(cx=320, cy=240, w=80, h=60, obj_conf=0.9)
    result = post_process(raw, _INPUT_SHAPE, conf_threshold=0.5)
    assert result is not None
    _, _, scores = result
    assert all(s >= 0.5 for s in scores), f"scores={scores}"


def test_three_head_box_coordinates_sane():
    """Формат B: координаты bbox в пределах кадра и x1<x2, y1<y2."""
    raw = _three_head_tensors(cx=320, cy=240, w=80, h=60, obj_conf=0.9)
    result = post_process(raw, _INPUT_SHAPE, conf_threshold=0.5)
    assert result is not None
    boxes, _, _ = result
    for box in boxes:
        x1, y1, x2, y2 = box
        assert 0 <= x1 <= x2 <= 640, f"x: {x1} < {x2}"
        assert 0 <= y1 <= y2 <= 480, f"y: {y1} < {y2}"


# ─────────────────────────────────────────────────────────────────────────────
#  Тесты — NMS
# ─────────────────────────────────────────────────────────────────────────────

def test_nms_removes_duplicate():
    """Два практически одинаковых bbox → NMS оставляет один."""
    # Два объекта почти в одном месте (IoU > 0.45 → один останется)
    det1 = [320, 240, 60, 50, 0.95, 0.95]
    det2 = [322, 242, 60, 50, 0.80, 0.90]  # почти совпадает
    raw = [np.array([[det1, det2]], dtype=np.float32)]
    result = post_process(raw, _INPUT_SHAPE, conf_threshold=0.5, nms_threshold=0.45)
    assert result is not None
    boxes, _, _ = result
    assert len(boxes) == 1, f"После NMS должен остаться 1 bbox, получено {len(boxes)}"


def test_nms_keeps_separate_detections():
    """Два разнесённых bbox → NMS оставляет оба."""
    det1 = [100, 100, 50, 50, 0.9, 0.9]   # левый верхний
    det2 = [500, 400, 50, 50, 0.9, 0.9]   # правый нижний
    raw = [np.array([[det1, det2]], dtype=np.float32)]
    result = post_process(raw, _INPUT_SHAPE, conf_threshold=0.5, nms_threshold=0.45)
    assert result is not None
    boxes, _, _ = result
    assert len(boxes) == 2, f"Оба bbox должны остаться, получено {len(boxes)}"


# ─────────────────────────────────────────────────────────────────────────────
#  Тест — совместимость с vision_tracker._pick_best_detection
# ─────────────────────────────────────────────────────────────────────────────

def test_output_compatible_with_vision_tracker():
    """Вывод post_process() совместим с форматом, ожидаемым VisionTracker."""
    from vision_tracker import VisionTracker
    from config import TARGET_CLASS_ID, CONF_THRESHOLD

    raw = _single_tensor(cx=320, cy=240, w=60, h=50, obj_conf=0.9)
    result = post_process(raw, _INPUT_SHAPE, conf_threshold=CONF_THRESHOLD)
    assert result is not None

    vt = VisionTracker()
    vt.reset()
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    track_result = vt.step(frame, result)
    assert track_result is not None, "VisionTracker должен принять вывод post_process()"
    cx, cy, conf, bbox = track_result
    assert abs(cx - 320) < 10, f"cx={cx}"
    assert abs(cy - 240) < 10, f"cy={cy}"
