"""
drone_model.py — YOLO post-processing for RK3588 NPU (Radxa Rock 5B).

Обрабатывает вывод rknn.inference() и возвращает декодированные детекции
в формате, ожидаемом VisionTracker:

    (boxes, classes, scores)

    boxes:   list of [x1, y1, x2, y2]  — пиксельные координаты в input_shape
    classes: list of int                — идентификаторы классов
    scores:  list of float              — confidence scores

Поддерживаемые форматы вывода RKNN:

    Формат A — стандартный YOLOv5: 3 тензора (по одному на каждый масштаб).
    Допустимые формы каждого тензора:
        (1, na, H, W, 5+nc)  — каналы первые (channels-first)
        (1, H, W, na*(5+nc)) — каналы последние (channels-last)

    Формат B — единый тензор с уже применёнными NMS (модели с встроенным NMS):
        (1, K, 6) — [x1, y1, x2, y2, score, class_id]
        (1, K, 5) — [x1, y1, x2, y2, score], однокласная модель

Якоря и страйды соответствуют стандартному YOLOv5 (обучение 640px).
Для кастомных моделей замените ANCHORS и STRIDES на актуальные значения
из конфигурационного файла экспорта.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
#  Конфигурация якорей — стандарт YOLOv5 (640-px input)
#  При использовании кастомных якорей заменить на значения из model.yaml
# ---------------------------------------------------------------------------
ANCHORS: np.ndarray = np.array(
    [
        [10, 13],  [16, 30],   [33, 23],    # P3 / stride=8  — мелкие объекты
        [30, 61],  [62, 45],   [59, 119],   # P4 / stride=16 — средние
        [116, 90], [156, 198], [373, 326],  # P5 / stride=32 — крупные / дальние цели
    ],
    dtype=np.float32,
)
STRIDES: List[int] = [8, 16, 32]
NA: int = 3  # якорей на масштаб

# Безопасный диапазон аргумента для np.exp: при |x| > 88 результат ±inf
_MAX_EXP_INPUT: float = 88.0


# ---------------------------------------------------------------------------
#  Вспомогательные функции
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -_MAX_EXP_INPUT, _MAX_EXP_INPUT)))


def _decode_head(
    feat: np.ndarray,
    anchors_scale: np.ndarray,
    stride: int,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Декодировать один выходной тензор YOLOv5.

    Принимает feat в форматах:
        (1, na, H, W, 5+nc)   — channels-first
        (1, H, W, na*(5+nc))  — channels-last

    Возвращает:
        boxes   — (N, 4) float32, [x1, y1, x2, y2] в пикселях
        classes — (N,)   int32
        scores  — (N,)   float32
    """
    nc = num_classes

    if feat.ndim == 5:
        # (1, na, H, W, 5+nc) — channels-first
        _, na_in, H, W, _ = feat.shape
        feat = _sigmoid(feat[0])                     # (na, H, W, 5+nc)
    elif feat.ndim == 4:
        # (1, H, W, na*(5+nc)) — channels-last
        _, H, W, C = feat.shape
        na_in = C // (5 + nc)
        feat = _sigmoid(feat[0])                     # (H, W, na*(5+nc))
        feat = feat.reshape(H, W, na_in, 5 + nc).transpose(2, 0, 1, 3)  # → (na, H, W, 5+nc)
    else:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    na_in = feat.shape[0]
    anchors_use = anchors_scale[:na_in]              # обрезаем, если na_in < NA

    grid_y, grid_x = np.mgrid[0:H, 0:W]             # (H, W) each
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)

    # ---- декодирование координат ----
    # feat[a, y, x, 0..1] = tx, ty (уже после sigmoid)
    tx = feat[:, :, :, 0]                            # (na, H, W)
    ty = feat[:, :, :, 1]
    tw = feat[:, :, :, 2]
    th = feat[:, :, :, 3]
    obj = feat[:, :, :, 4]
    cls_probs = feat[:, :, :, 5:]                    # (na, H, W, nc)

    # bx = (tx*2 - 0.5 + gx) * stride  — YOLOv5 v6.0 decode
    bx = (tx * 2.0 - 0.5 + grid_x[np.newaxis]) * stride   # (na, H, W)
    by = (ty * 2.0 - 0.5 + grid_y[np.newaxis]) * stride

    pw = anchors_use[:, 0].reshape(na_in, 1, 1)
    ph = anchors_use[:, 1].reshape(na_in, 1, 1)
    bw = (tw * 2.0) ** 2 * pw
    bh = (th * 2.0) ** 2 * ph

    x1 = (bx - bw / 2.0).flatten()
    y1 = (by - bh / 2.0).flatten()
    x2 = (bx + bw / 2.0).flatten()
    y2 = (by + bh / 2.0).flatten()

    # ---- confidence = objectness × class_prob ----
    if nc == 1:
        scores  = (obj * cls_probs[:, :, :, 0]).flatten()
        classes = np.zeros(len(scores), dtype=np.int32)
    else:
        cls_idx = np.argmax(cls_probs, axis=-1)      # (na, H, W)
        cls_val = np.max(cls_probs, axis=-1)
        scores  = (obj * cls_val).flatten()
        classes = cls_idx.flatten().astype(np.int32)

    boxes = np.stack([x1, y1, x2, y2], axis=1)      # (N, 4)
    return boxes, classes, scores


# ---------------------------------------------------------------------------
#  Публичная функция
# ---------------------------------------------------------------------------

def post_process(
    raw_outputs: list,
    input_shape: List[int],
    conf_threshold: float = 0.5,
    nms_threshold: float  = 0.45,
    num_classes: int      = 1,
) -> Optional[Tuple[list, list, list]]:
    """
    Пост-обработка вывода RKNN-инференса YOLOv5.

    Args:
        raw_outputs:    Список numpy-массивов из rknn.inference().
        input_shape:    [width, height] входного кадра (напр. [640, 480]).
        conf_threshold: Порог уверенности.
        nms_threshold:  Порог IoU для NMS.
        num_classes:    Количество классов в модели.

    Returns:
        (boxes, classes, scores) или None если детекций нет.

        boxes:   list of [x1, y1, x2, y2]
        classes: list of int
        scores:  list of float
    """
    if not raw_outputs:
        return None

    input_w, input_h = int(input_shape[0]), int(input_shape[1])

    # ----------------------------------------------------------------
    # Формат B — готовый или предварительно обработанный 1-тензорный
    # вывод: (1, K, 6) или (1, K, 5).  NMS применяется в любом случае
    # для надёжности (если модель уже включает NMS — повтор безвреден).
    # ----------------------------------------------------------------
    if len(raw_outputs) == 1:
        out = raw_outputs[0]
        if out.ndim == 3:
            out = out[0]                             # (K, 6) или (K, 5)
            ncols = out.shape[-1]
            if ncols in (5, 6):
                mask = out[:, 4] >= conf_threshold
                out  = out[mask]
                if len(out) == 0:
                    return None
                boxes_np   = out[:, :4].astype(np.float32)
                scores_arr = out[:, 4].astype(np.float32)
                classes_arr = (
                    out[:, 5].astype(np.int32) if ncols == 6
                    else np.zeros(len(out), dtype=np.int32)
                )
                # Клиппинг
                boxes_np[:, 0] = np.clip(boxes_np[:, 0], 0.0, input_w)
                boxes_np[:, 1] = np.clip(boxes_np[:, 1], 0.0, input_h)
                boxes_np[:, 2] = np.clip(boxes_np[:, 2], 0.0, input_w)
                boxes_np[:, 3] = np.clip(boxes_np[:, 3], 0.0, input_h)
                # NMS
                final_boxes, final_classes, final_scores = [], [], []
                for cls_id in np.unique(classes_arr):
                    idx       = (classes_arr == cls_id)
                    cb        = boxes_np[idx]
                    cs        = scores_arr[idx]
                    bxywh     = cb.copy()
                    bxywh[:, 2] -= cb[:, 0]
                    bxywh[:, 3] -= cb[:, 1]
                    keep = cv2.dnn.NMSBoxes(
                        bxywh.tolist(), cs.tolist(),
                        conf_threshold, nms_threshold,
                    )
                    if len(keep) == 0:
                        continue
                    for k in keep.flatten():
                        final_boxes.append(cb[k].tolist())
                        final_classes.append(int(cls_id))
                        final_scores.append(float(cs[k]))
                return (final_boxes, final_classes, final_scores) if final_scores else None
        # Тензор не в формате (1, K, 5/6) — пробуем Format A как одноголовный YOLOv5
    if len(raw_outputs) not in (1, 3):
        return None
    outputs_3 = raw_outputs

    # ----------------------------------------------------------------
    # Формат A — стандарт YOLOv5: 3 выходных тензора
    # ----------------------------------------------------------------
    all_boxes:   List[np.ndarray] = []
    all_classes: List[np.ndarray] = []
    all_scores:  List[np.ndarray] = []

    for i, feat in enumerate(outputs_3):
        stride        = STRIDES[i] if i < len(STRIDES) else STRIDES[-1] * (2 ** (i - len(STRIDES) + 1))
        anchors_scale = ANCHORS[i * NA: (i + 1) * NA]
        b, c, s = _decode_head(feat, anchors_scale, stride, num_classes)
        all_boxes.append(b)
        all_classes.append(c)
        all_scores.append(s)

    boxes_np   = np.concatenate(all_boxes,   axis=0)  # (N, 4)
    classes_np = np.concatenate(all_classes, axis=0)  # (N,)
    scores_np  = np.concatenate(all_scores,  axis=0)  # (N,)

    # Фильтр по confidence
    mask       = scores_np >= conf_threshold
    boxes_np   = boxes_np[mask]
    classes_np = classes_np[mask]
    scores_np  = scores_np[mask]

    if len(scores_np) == 0:
        return None

    # Клиппинг в границы кадра
    boxes_np[:, 0] = np.clip(boxes_np[:, 0], 0.0, input_w)
    boxes_np[:, 1] = np.clip(boxes_np[:, 1], 0.0, input_h)
    boxes_np[:, 2] = np.clip(boxes_np[:, 2], 0.0, input_w)
    boxes_np[:, 3] = np.clip(boxes_np[:, 3], 0.0, input_h)

    # NMS per class (OpenCV dnn.NMSBoxes)
    final_boxes:   List[list]  = []
    final_classes: List[int]   = []
    final_scores:  List[float] = []

    for cls_id in np.unique(classes_np):
        idx        = (classes_np == cls_id)
        cls_boxes  = boxes_np[idx]
        cls_scores = scores_np[idx]

        # OpenCV NMSBoxes принимает (x, y, w, h)
        boxes_xywh        = cls_boxes.copy()
        boxes_xywh[:, 2] -= cls_boxes[:, 0]
        boxes_xywh[:, 3] -= cls_boxes[:, 1]

        keep = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            cls_scores.tolist(),
            conf_threshold,
            nms_threshold,
        )
        if len(keep) == 0:
            continue
        for k in keep.flatten():
            final_boxes.append(cls_boxes[k].tolist())
            final_classes.append(int(cls_id))
            final_scores.append(float(cls_scores[k]))

    if not final_scores:
        return None

    return (final_boxes, final_classes, final_scores)
