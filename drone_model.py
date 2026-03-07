"""
drone_model — постобработка вывода NPU YOLOv5 для детекции дронов.

Реализует post_process() для модели drone_model.rknn (RK3588 RKNN-Lite).

Поддерживаемые форматы вывода (RKNN модели с сигмоидом внутри графа):
  1. Три тензора (стандартный YOLOv5 RKNN, уровни P3/P4/P5):
       Форма каждого: [1, na, ny, nx, 5+nc]  ИЛИ  [1, na*(5+nc), ny, nx]
       Значения уже post-sigmoid (активация применена в RKNN графе).
  2. Один объединённый тензор (упрощённый экспорт / decoupled head):
       Форма: [1, num_detections, 5+nc]
       Каждая строка: [cx, cy, w, h, obj_conf, cls_conf...]

Выходной формат post_process():
    (boxes, classes, scores)
        boxes:   list[[x1, y1, x2, y2]]  — координаты в пикселях входного кадра
        classes: list[int]               — ID класса (0 = дрон)
        scores:  list[float]             — confidence score

    Возвращает None если ничего не обнаружено.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
#  Якоря YOLOv5 (стандартные для входа 640×N)
#  P3/stride=8, P4/stride=16, P5/stride=32
# ─────────────────────────────────────────────────────────────────────────────
_ANCHORS = np.array(
    [
        [10, 13], [16, 30], [33, 23],       # P3 / stride 8
        [30, 61], [62, 45], [59, 119],      # P4 / stride 16
        [116, 90], [156, 198], [373, 326],  # P5 / stride 32
    ],
    dtype=np.float32,
)
_STRIDES = [8, 16, 32]
_NA = 3   # якорей на уровень детекции


def _decode_head(
    raw: np.ndarray,
    anchors: np.ndarray,
    stride: int,
    input_w: int,
    input_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Декодировать один уровень детекции YOLOv5.

    Args:
        raw:      [na, ny, nx, 5+nc] — тензор после sigmoid (post-sigmoid RKNN)
        anchors:  [na, 2]            — якоря (pw, ph) для данного stride
        stride:   шаг сетки (8 / 16 / 32)
        input_w:  ширина входного кадра в пикселях
        input_h:  высота входного кадра в пикселях

    Returns:
        boxes:  [N, 4]  float32  (x1, y1, x2, y2) в пикселях
        scores: [N]     float32  (obj_conf * max_class_conf)
    """
    na, ny, nx, _ = raw.shape

    # Сетка ячеек (nx, ny)
    gx, gy = np.meshgrid(
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
    )
    grid = np.stack([gx, gy], axis=-1)                        # (ny, nx, 2)
    grid = np.broadcast_to(grid[np.newaxis], (na, ny, nx, 2)) # (na, ny, nx, 2)

    # Якоря → (na, 1, 1, 2) для broadcast
    anch = anchors[:, np.newaxis, np.newaxis, :]

    # Декодируем центр и размер
    # Формула YOLOv5: bx = (tx * 2 - 0.5 + cx) * stride
    #                  bw = (tw * 2)^2 * pw
    xy = (raw[..., :2] * 2.0 - 0.5 + grid) * stride  # (na, ny, nx, 2)
    wh = (raw[..., 2:4] * 2.0) ** 2 * anch            # (na, ny, nx, 2)

    # (x1, y1, x2, y2)
    x1y1 = xy - wh / 2.0
    x2y2 = xy + wh / 2.0
    boxes = np.concatenate([x1y1, x2y2], axis=-1).reshape(-1, 4)

    # Уверенность = objectness * max(class_scores)
    obj_conf   = raw[..., 4]        # (na, ny, nx)
    class_conf = raw[..., 5:]       # (na, ny, nx, nc)
    scores = (obj_conf * class_conf.max(axis=-1)).reshape(-1)

    return boxes.astype(np.float32), scores.astype(np.float32)


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    input_w: int,
    input_h: int,
    conf_threshold: float,
    nms_threshold: float,
) -> Tuple[list, list]:
    """
    Применить NMS и вернуть (out_boxes, out_scores) после фильтрации.

    Координаты клампируются в пределы [0, input_w/input_h].
    """
    # Клампируем в пределы кадра
    boxes[:, 0] = np.clip(boxes[:, 0], 0, input_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, input_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, input_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, input_h)

    # cv2.dnn.NMSBoxes принимает [x, y, w, h]
    boxes_xywh = [
        [float(b[0]), float(b[1]),
         float(b[2] - b[0]), float(b[3] - b[1])]
        for b in boxes
    ]

    indices = cv2.dnn.NMSBoxes(
        boxes_xywh, scores.tolist(), conf_threshold, nms_threshold
    )

    if len(indices) == 0:
        return [], []

    # cv2 < 4.7 возвращает [[i], [j], ...], cv2 >= 4.7 возвращает [i, j, ...]
    flat_idx = [
        int(i[0]) if isinstance(i, (list, np.ndarray)) else int(i)
        for i in indices
    ]

    out_boxes  = [
        [int(boxes[i][0]), int(boxes[i][1]),
         int(boxes[i][2]), int(boxes[i][3])]
        for i in flat_idx
    ]
    out_scores = [float(scores[i]) for i in flat_idx]
    return out_boxes, out_scores


def post_process(
    raw_outputs: list,
    input_shape: List[int],
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.45,
) -> Optional[Tuple[list, list, list]]:
    """
    Постобработка вывода NPU YOLO: декодирование, фильтрация, NMS.

    Args:
        raw_outputs:    список numpy-массивов из rknn.inference()
        input_shape:    [ширина, высота] входного кадра (напр. [640, 480])
        conf_threshold: порог уверенности (детекции ниже — отбрасываются)
        nms_threshold:  IoU порог для NMS

    Returns:
        (boxes, classes, scores) где:
            boxes:   list[[x1, y1, x2, y2]]  — координаты в пикселях
            classes: list[int]               — ID класса (0 = дрон)
            scores:  list[float]             — confidence score
        Возвращает None если нет обнаружений выше порога.
    """
    if not raw_outputs:
        return None

    input_w, input_h = int(input_shape[0]), int(input_shape[1])

    all_boxes: List[np.ndarray] = []
    all_scores: List[float]     = []

    # ─────────────────────────────────────────────────────────────────────────
    # Формат A: один объединённый тензор [1, N, 5+nc] или [N, 5+nc]
    # (упрощённый экспорт, decoupled head, или уже декодированный вывод)
    # Строки: [cx, cy, w, h, obj_conf, cls_conf...]  (post-sigmoid)
    # ─────────────────────────────────────────────────────────────────────────
    if len(raw_outputs) == 1:
        out = np.asarray(raw_outputs[0], dtype=np.float32)
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]          # (N, 5+nc)
        if out.ndim == 2 and out.shape[-1] >= 5:
            nc = out.shape[-1] - 5
            for det in out:
                obj_conf = float(det[4])
                if obj_conf < conf_threshold:
                    continue
                cls_id = int(np.argmax(det[5:5 + nc])) if nc > 0 else 0
                cls_cf = float(det[5 + cls_id]) if nc > 0 else 1.0
                score  = obj_conf * cls_cf
                if score < conf_threshold:
                    continue
                cx, cy, w, h = det[:4]
                all_boxes.append(
                    np.array([cx - w / 2, cy - h / 2,
                               cx + w / 2, cy + h / 2], dtype=np.float32)
                )
                all_scores.append(score)

    # ─────────────────────────────────────────────────────────────────────────
    # Формат B: три тензора (YOLOv5 P3/P4/P5)
    # Поддерживаемые формы:
    #   [1, na, ny, nx, 5+nc]   — уже разложен по осям
    #   [1, na*(5+nc), ny, nx]  — нужна перестановка осей
    # ─────────────────────────────────────────────────────────────────────────
    elif len(raw_outputs) == 3:
        for idx, raw in enumerate(raw_outputs):
            arr = np.asarray(raw, dtype=np.float32)

            # Убираем batch-измерение если есть
            if arr.ndim == 5 and arr.shape[0] == 1:
                arr = arr[0]                  # (na, ny, nx, 5+nc)

            if arr.ndim == 4:
                if arr.shape[0] == 1:
                    arr = arr[0]              # (na*(5+nc), ny, nx) или (na, ny, nx, 5+nc)
                if arr.ndim == 4:
                    # (na, ny, nx, 5+nc) — уже нужный формат
                    pass
                elif arr.ndim == 3:
                    # (na*(5+nc), ny, nx) — транспонируем → (na, ny, nx, 5+nc)
                    ch, ny, nx = arr.shape
                    nc_total = ch // _NA
                    arr = arr.reshape(_NA, nc_total, ny, nx)
                    arr = arr.transpose(0, 2, 3, 1)  # (na, ny, nx, nc_total)

            # Убедимся, что форма правильная: (na, ny, nx, 5+nc)
            if arr.ndim == 3:
                # (na*(5+nc), ny, nx) — транспонируем
                ch, ny, nx = arr.shape
                nc_total = ch // _NA
                arr = arr.reshape(_NA, nc_total, ny, nx)
                arr = arr.transpose(0, 2, 3, 1)

            if arr.ndim != 4 or arr.shape[0] != _NA:
                continue  # неизвестный формат — пропускаем

            anchors = _ANCHORS[idx * _NA:(idx + 1) * _NA]
            stride  = _STRIDES[idx]

            boxes, scores = _decode_head(arr, anchors, stride, input_w, input_h)
            mask = scores >= conf_threshold
            if mask.any():
                all_boxes.extend(boxes[mask])
                all_scores.extend(scores[mask].tolist())

    if not all_boxes:
        return None

    boxes_arr  = np.array(all_boxes,  dtype=np.float32)
    scores_arr = np.array(all_scores, dtype=np.float32)

    out_boxes, out_scores = _nms(
        boxes_arr, scores_arr, input_w, input_h, conf_threshold, nms_threshold
    )

    if not out_boxes:
        return None

    # Один класс: дрон (TARGET_CLASS_ID = 0)
    out_classes = [0] * len(out_boxes)
    return (out_boxes, out_classes, out_scores)
