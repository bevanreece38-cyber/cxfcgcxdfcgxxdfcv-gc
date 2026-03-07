"""
drone_model.py — YOLO post-processing for RK3588 NPU (Radxa Rock 5B+).

Primary model: YOLOv8n RKNN (RK3588 NPU_CORE_0_1_2).

Поддерживаемые форматы вывода rknn.inference() (авто-определение):

  Format B  — встроенный NMS (один тензор):
                (1, K, 6)  [x1,y1,x2,y2, score, class_id]
                (1, K, 5)  [x1,y1,x2,y2, score]  (однокласная)

  Format C8 — YOLOv8n DFL (3 тензора, каждый (1, reg_max*4+nc, H, W) или channels-last):
                Стандартный RKNN-экспорт YOLOv8n без встроенного NMS.
                reg_max обнаруживается автоматически (обычно 16).
                Выполняется DFL decode → ltrb → x1y1x2y2.

  Format D8 — YOLOv8n decoded (3 тензора, каждый (1, 4+nc, H, W)):
                Экспорт YOLOv8n с предварительно декодированными bbox (cx,cy,w,h).

  Format A5 — YOLOv5 (3 тензора, обратная совместимость):
                (1, na, H, W, 5+nc) channels-first или (1, H, W, na*(5+nc))

Возвращает:
  (boxes, classes, scores)
  boxes:   list of [x1, y1, x2, y2]  — пиксельные координаты в input_shape
  classes: list of int
  scores:  list of float

Для установки на Radxa 5B+:
  Экспортируйте YOLOv8n в RKNN формат через rknn-toolkit2:
    model.export_rknn('drone_model.rknn')
  Задайте MODEL_PATH = 'drone_model.rknn' в config.py.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
#  Константы
# ---------------------------------------------------------------------------

# YOLOv5 якоря (обратная совместимость, 640-px input)
_ANCHORS_YV5: np.ndarray = np.array(
    [
        [10, 13],  [16, 30],   [33, 23],    # P3 / stride=8  — мелкие объекты
        [30, 61],  [62, 45],   [59, 119],   # P4 / stride=16 — средние
        [116, 90], [156, 198], [373, 326],  # P5 / stride=32 — крупные / дальние
    ],
    dtype=np.float32,
)
_NA_YV5: int = 3        # якорей на масштаб (YOLOv5)
STRIDES: List[int] = [8, 16, 32]

# Безопасный диапазон для np.exp (|x| > 88 → ±inf)
_MAX_EXP_INPUT: float = 88.0


# ---------------------------------------------------------------------------
#  Вспомогательные функции — YOLOv5
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -_MAX_EXP_INPUT, _MAX_EXP_INPUT)))


def _decode_yv5_head(
    feat: np.ndarray,
    anchors_scale: np.ndarray,
    stride: int,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Декодировать один тензор YOLOv5 (anchor-based)."""
    nc = num_classes

    if feat.ndim == 5:
        # (1, na, H, W, 5+nc) channels-first
        _, na_in, H, W, _ = feat.shape
        feat = _sigmoid(feat[0])              # (na, H, W, 5+nc)
    elif feat.ndim == 4:
        # (1, H, W, na*(5+nc)) channels-last
        _, H, W, C = feat.shape
        na_in = C // (5 + nc)
        feat = _sigmoid(feat[0])              # (H, W, na*(5+nc))
        feat = feat.reshape(H, W, na_in, 5 + nc).transpose(2, 0, 1, 3)
    else:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    na_in = feat.shape[0]
    anchors = anchors_scale[:na_in]

    grid_y, grid_x = np.mgrid[0:H, 0:W]
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)

    tx = feat[:, :, :, 0]
    ty = feat[:, :, :, 1]
    tw = feat[:, :, :, 2]
    th = feat[:, :, :, 3]
    obj = feat[:, :, :, 4]
    cls_probs = feat[:, :, :, 5:]

    bx = (tx * 2.0 - 0.5 + grid_x[np.newaxis]) * stride
    by = (ty * 2.0 - 0.5 + grid_y[np.newaxis]) * stride
    pw = anchors[:, 0].reshape(na_in, 1, 1)
    ph = anchors[:, 1].reshape(na_in, 1, 1)
    bw = (tw * 2.0) ** 2 * pw
    bh = (th * 2.0) ** 2 * ph

    x1 = (bx - bw / 2.0).flatten()
    y1 = (by - bh / 2.0).flatten()
    x2 = (bx + bw / 2.0).flatten()
    y2 = (by + bh / 2.0).flatten()

    if nc == 1:
        scores  = (obj * cls_probs[:, :, :, 0]).flatten()
        classes = np.zeros(len(scores), dtype=np.int32)
    else:
        cls_idx = np.argmax(cls_probs, axis=-1)
        cls_val = np.max(cls_probs, axis=-1)
        scores  = (obj * cls_val).flatten()
        classes = cls_idx.flatten().astype(np.int32)

    return np.stack([x1, y1, x2, y2], axis=1), classes, scores


# ---------------------------------------------------------------------------
#  Вспомогательные функции — YOLOv8n (anchor-free)
# ---------------------------------------------------------------------------

def _dfl_decode(dist: np.ndarray, reg_max: int) -> np.ndarray:
    """
    Distribution Focal Loss decode.

    dist: (1, reg_max*4, H, W)
    Returns: (1, 4, H, W) as [left, top, right, bottom] в grid-единицах.
    """
    b, _, H, W = dist.shape
    # (1, 4, reg_max, H, W)
    d = dist.reshape(b, 4, reg_max, H, W)
    # Softmax по оси reg_max
    d = d - d.max(axis=2, keepdims=True)
    e = np.exp(np.clip(d, -_MAX_EXP_INPUT, _MAX_EXP_INPUT))
    s = e / e.sum(axis=2, keepdims=True)
    # Weighted mean → расстояние в grid-единицах
    w = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1, 1)
    return (s * w).sum(axis=2)     # (1, 4, H, W)


def _decode_yv8_dfl_head(
    feat: np.ndarray,
    stride: int,
    num_classes: int,
    reg_max: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    YOLOv8n DFL-head decode.

    feat: (1, reg_max*4+nc, H, W) или channels-last (1, H, W, reg_max*4+nc).
    """
    # Normalize to channels-first
    if feat.ndim == 4 and feat.shape[1] > feat.shape[-1]:
        pass  # (1, C, H, W) — уже channels-first
    elif feat.ndim == 4:
        feat = feat.transpose(0, 3, 1, 2)  # (1, H, W, C) → (1, C, H, W)

    _, C, H, W = feat.shape
    bbox_ch = reg_max * 4

    dist    = feat[:, :bbox_ch]      # (1, reg_max*4, H, W)
    cls_raw = feat[0, bbox_ch:]      # (nc, H, W)

    # DFL → ltrb в grid-единицах
    ltrb = _dfl_decode(dist, reg_max)[0]   # (4, H, W)

    # Grid centers в пикселях
    gy, gx = np.mgrid[0:H, 0:W]
    cx_px = (gx.astype(np.float32) + 0.5) * stride
    cy_px = (gy.astype(np.float32) + 0.5) * stride

    # ltrb (grid units) → абсолютные пиксели
    x1 = (cx_px - ltrb[0] * stride).flatten()
    y1 = (cy_px - ltrb[1] * stride).flatten()
    x2 = (cx_px + ltrb[2] * stride).flatten()
    y2 = (cy_px + ltrb[3] * stride).flatten()

    boxes = np.stack([x1, y1, x2, y2], axis=1)    # (N, 4)

    # Class scores (sigmoid)
    cls_s = _sigmoid(cls_raw)                       # (nc, H, W)
    if num_classes == 1:
        scores  = cls_s[0].flatten()
        classes = np.zeros(len(scores), dtype=np.int32)
    else:
        cls_idx = np.argmax(cls_s, axis=0)
        scores  = np.max(cls_s, axis=0).flatten()
        classes = cls_idx.flatten().astype(np.int32)

    return boxes, classes, scores


def _decode_yv8_decoded_head(
    feat: np.ndarray,
    stride: int,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    YOLOv8n head с предварительно декодированным bbox (cx, cy, w, h).

    feat: (1, 4+nc, H, W) или channels-last (1, H, W, 4+nc).
    """
    if feat.ndim == 4 and feat.shape[1] > feat.shape[-1]:
        pass
    elif feat.ndim == 4:
        feat = feat.transpose(0, 3, 1, 2)

    _, C, H, W = feat.shape
    f = feat[0]    # (C, H, W)

    gy, gx = np.mgrid[0:H, 0:W]

    # bbox (cx,cy,w,h) в grid-единицах → пиксели
    cx = (f[0] + gx.astype(np.float32)) * stride
    cy = (f[1] + gy.astype(np.float32)) * stride
    bw = np.exp(np.clip(f[2], -_MAX_EXP_INPUT, _MAX_EXP_INPUT)) * stride
    bh = np.exp(np.clip(f[3], -_MAX_EXP_INPUT, _MAX_EXP_INPUT)) * stride

    x1 = (cx - bw / 2.0).flatten()
    y1 = (cy - bh / 2.0).flatten()
    x2 = (cx + bw / 2.0).flatten()
    y2 = (cy + bh / 2.0).flatten()

    boxes   = np.stack([x1, y1, x2, y2], axis=1)
    cls_s   = _sigmoid(f[4:])           # (nc, H, W)
    if num_classes == 1:
        scores  = cls_s[0].flatten()
        classes = np.zeros(len(scores), dtype=np.int32)
    else:
        cls_idx = np.argmax(cls_s, axis=0)
        scores  = np.max(cls_s, axis=0).flatten()
        classes = cls_idx.flatten().astype(np.int32)

    return boxes, classes, scores


# ---------------------------------------------------------------------------
#  Авто-определение формата
# ---------------------------------------------------------------------------

def _channels_of(tensor: np.ndarray) -> int:
    """Вернуть число каналов из тензора (1, C, H, W) или (1, H, W, C)."""
    if tensor.ndim == 4:
        return int(tensor.shape[1]) if tensor.shape[1] <= tensor.shape[-1] else int(tensor.shape[-1])
    return 0


def _detect_format(outputs: list, nc: int) -> str:
    """
    Авто-определение формата вывода RKNN.

    Возвращает одну из строк:
      'nms'        — встроенный NMS (1, K, 5/6)
      'yv8_dfl'    — YOLOv8n DFL 3 головы
      'yv8_decoded'— YOLOv8n decoded 3 головы
      'yv5'        — YOLOv5 anchor-based
      'unknown'
    """
    if len(outputs) == 1:
        t = outputs[0]
        if t.ndim == 3 and t.shape[-1] in (5, 6):
            return 'nms'

    if len(outputs) == 3:
        t0 = outputs[0]
        if t0.ndim == 5:
            return 'yv5'         # (1, na, H, W, 5+nc)
        if t0.ndim == 4:
            C = _channels_of(t0)
            if C == 4 + nc:
                return 'yv8_decoded'
            if C == _NA_YV5 * (5 + nc):
                return 'yv5'
            # YOLOv8 DFL: C = reg_max*4 + nc, reg_max >= 4
            rem = C - nc
            if rem > 0 and rem % 4 == 0 and rem // 4 >= 4:
                return 'yv8_dfl'

    return 'unknown'


# ---------------------------------------------------------------------------
#  NMS helper (shared)
# ---------------------------------------------------------------------------

def _apply_nms(
    boxes_np:   np.ndarray,
    classes_np: np.ndarray,
    scores_np:  np.ndarray,
    input_w:    int,
    input_h:    int,
    conf_thr:   float,
    nms_thr:    float,
) -> Tuple[list, list, list]:
    """Threshold + clip + per-class NMS. Returns (boxes, classes, scores)."""
    mask       = scores_np >= conf_thr
    boxes_np   = boxes_np[mask]
    classes_np = classes_np[mask]
    scores_np  = scores_np[mask]

    if len(scores_np) == 0:
        return [], [], []

    boxes_np[:, 0] = np.clip(boxes_np[:, 0], 0.0, input_w)
    boxes_np[:, 1] = np.clip(boxes_np[:, 1], 0.0, input_h)
    boxes_np[:, 2] = np.clip(boxes_np[:, 2], 0.0, input_w)
    boxes_np[:, 3] = np.clip(boxes_np[:, 3], 0.0, input_h)

    final_boxes:   List[list]  = []
    final_classes: List[int]   = []
    final_scores:  List[float] = []

    for cls_id in np.unique(classes_np):
        idx = (classes_np == cls_id)
        cb  = boxes_np[idx]
        cs  = scores_np[idx]
        # NMSBoxes принимает (x, y, w, h)
        bxywh        = cb.copy()
        bxywh[:, 2] -= cb[:, 0]
        bxywh[:, 3] -= cb[:, 1]
        keep = cv2.dnn.NMSBoxes(bxywh.tolist(), cs.tolist(), conf_thr, nms_thr)
        if len(keep) == 0:
            continue
        for k in keep.flatten():
            final_boxes.append(cb[k].tolist())
            final_classes.append(int(cls_id))
            final_scores.append(float(cs[k]))

    return final_boxes, final_classes, final_scores


# ---------------------------------------------------------------------------
#  Публичная функция
# ---------------------------------------------------------------------------

def post_process(
    raw_outputs,
    input_shape:    List[int],
    conf_threshold: float = 0.5,
    nms_threshold:  float = 0.45,
    num_classes:    int   = 1,
) -> Optional[Tuple[list, list, list]]:
    """
    Пост-обработка вывода RKNN-инференса.

    Поддерживает YOLOv8n (primary) и YOLOv5 (backward compat).

    Args:
        raw_outputs:    список numpy-массивов из rknn.inference().
        input_shape:    [width, height] входного кадра (напр. [640, 480]).
        conf_threshold: порог уверенности.
        nms_threshold:  порог IoU для NMS.
        num_classes:    число классов модели.

    Returns:
        (boxes, classes, scores) или None если детекций нет.
    """
    if not raw_outputs:
        return None

    input_w = int(input_shape[0])
    input_h = int(input_shape[1])
    fmt     = _detect_format(raw_outputs, num_classes)

    # ----------------------------------------------------------------
    # Format B — NMS встроен
    # ----------------------------------------------------------------
    if fmt == 'nms':
        out  = raw_outputs[0][0]                   # (K, 5) или (K, 6)
        ncol = out.shape[-1]
        mask        = out[:, 4] >= conf_threshold
        out         = out[mask]
        if len(out) == 0:
            return None
        boxes_np    = out[:, :4].astype(np.float32)
        scores_arr  = out[:, 4].astype(np.float32)
        classes_arr = (
            out[:, 5].astype(np.int32) if ncol == 6
            else np.zeros(len(out), dtype=np.int32)
        )
        b, c, s = _apply_nms(boxes_np, classes_arr, scores_arr,
                              input_w, input_h, conf_threshold, nms_threshold)
        return (b, c, s) if s else None

    # ----------------------------------------------------------------
    # Format C8 — YOLOv8n DFL (3 heads)
    # ----------------------------------------------------------------
    if fmt == 'yv8_dfl':
        t0      = raw_outputs[0]
        C       = _channels_of(t0)
        reg_max = (C - num_classes) // 4
        all_b, all_c, all_s = [], [], []
        for i, feat in enumerate(raw_outputs):
            stride = STRIDES[i] if i < len(STRIDES) else STRIDES[-1] * (2 ** (i - len(STRIDES) + 1))
            b, c, s = _decode_yv8_dfl_head(feat, stride, num_classes, reg_max)
            all_b.append(b); all_c.append(c); all_s.append(s)
        boxes_np   = np.concatenate(all_b)
        classes_np = np.concatenate(all_c)
        scores_np  = np.concatenate(all_s)
        b, c, s = _apply_nms(boxes_np, classes_np, scores_np,
                              input_w, input_h, conf_threshold, nms_threshold)
        return (b, c, s) if s else None

    # ----------------------------------------------------------------
    # Format D8 — YOLOv8n decoded (3 heads, 4+nc channels)
    # ----------------------------------------------------------------
    if fmt == 'yv8_decoded':
        all_b, all_c, all_s = [], [], []
        for i, feat in enumerate(raw_outputs):
            stride = STRIDES[i] if i < len(STRIDES) else STRIDES[-1] * (2 ** (i - len(STRIDES) + 1))
            b, c, s = _decode_yv8_decoded_head(feat, stride, num_classes)
            all_b.append(b); all_c.append(c); all_s.append(s)
        boxes_np   = np.concatenate(all_b)
        classes_np = np.concatenate(all_c)
        scores_np  = np.concatenate(all_s)
        b, c, s = _apply_nms(boxes_np, classes_np, scores_np,
                              input_w, input_h, conf_threshold, nms_threshold)
        return (b, c, s) if s else None

    # ----------------------------------------------------------------
    # Format A5 — YOLOv5 anchor-based (3 heads, backward compat)
    # ----------------------------------------------------------------
    if fmt == 'yv5':
        all_b, all_c, all_s = [], [], []
        for i, feat in enumerate(raw_outputs):
            stride        = STRIDES[i] if i < len(STRIDES) else STRIDES[-1] * (2 ** (i - len(STRIDES) + 1))
            anchors_scale = _ANCHORS_YV5[i * _NA_YV5: (i + 1) * _NA_YV5]
            b, c, s = _decode_yv5_head(feat, anchors_scale, stride, num_classes)
            all_b.append(b); all_c.append(c); all_s.append(s)
        boxes_np   = np.concatenate(all_b)
        classes_np = np.concatenate(all_c)
        scores_np  = np.concatenate(all_s)
        b, c, s = _apply_nms(boxes_np, classes_np, scores_np,
                              input_w, input_h, conf_threshold, nms_threshold)
        return (b, c, s) if s else None

    return None
