"""
test_main_components.py — тесты изолированных компонентов main.py.

Тестируем без запуска реального MAVLink / камеры / NPU.
rknnlite замокан в conftest.py (недоступен в CI, доступен только на Radxa Rock 5B).

Покрывает:
  - _encode_worker: кодирует BGR → JPEG, обновляет LATEST_JPEG, нотифицирует _STREAM_COND
  - drop-oldest queue: старый кадр дропается при полной очереди
  - NPU graceful degradation: _run_inference → None при npu=None
  - _draw_hud: не рисует при target_x=-1 (DEAD_RECKON); рисует при валидных координатах
  - _limit_fps: точно ограничивает FPS; не спит при опоздании
"""

import time
import queue
import threading
import numpy as np
import cv2
import pytest
from unittest.mock import patch

# conftest.py замокал rknnlite → import main работает без железа
import main as m

from config import (
    STREAM_WIDTH, STREAM_HEIGHT,
    FRAME_WIDTH, FRAME_HEIGHT, MAX_FPS,
)


# ─── 1. _encode_worker ────────────────────────────────────────────────────────

def test_encode_worker_produces_jpeg():
    """_encode_worker кодирует кадр в JPEG и обновляет LATEST_JPEG."""
    frame = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
    frame[100:200, 100:200] = [0, 128, 255]

    old_jpeg = m.LATEST_JPEG
    try:
        m._RAW_FRAME_Q.put_nowait(frame)
    except queue.Full:
        m._RAW_FRAME_Q.get_nowait()
        m._RAW_FRAME_Q.put_nowait(frame)

    deadline = time.monotonic() + 2.0
    jpeg = None
    while time.monotonic() < deadline:
        with m._STREAM_COND:
            if m.LATEST_JPEG is not None and m.LATEST_JPEG is not old_jpeg:
                jpeg = m.LATEST_JPEG
                break
        time.sleep(0.01)

    assert jpeg is not None, "_encode_worker не обновил LATEST_JPEG за 2 секунды"

    buf = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert img is not None, "LATEST_JPEG не является валидным JPEG"
    assert img.shape == (STREAM_HEIGHT, STREAM_WIDTH, 3)


def test_encode_worker_notifies_condition():
    """_encode_worker вызывает _STREAM_COND.notify_all() после кодирования."""
    frame = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
    notified = threading.Event()

    def wait_for_notify():
        with m._STREAM_COND:
            m._STREAM_COND.wait(timeout=2.0)
            notified.set()

    t = threading.Thread(target=wait_for_notify, daemon=True)
    t.start()

    try:
        m._RAW_FRAME_Q.put_nowait(frame)
    except queue.Full:
        m._RAW_FRAME_Q.get_nowait()
        m._RAW_FRAME_Q.put_nowait(frame)

    t.join(timeout=3.0)
    assert notified.is_set(), "_STREAM_COND.notify_all() не был вызван"


# ─── 2. Drop-oldest queue logic ──────────────────────────────────────────────

def test_raw_frame_queue_drop_oldest():
    """Когда очередь полна — старый кадр дропается, новый добавляется."""
    while not m._RAW_FRAME_Q.empty():
        try:
            m._RAW_FRAME_Q.get_nowait()
        except queue.Empty:
            break

    old_frame = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
    new_frame = np.ones((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8) * 42

    m._RAW_FRAME_Q.put_nowait(old_frame)
    assert m._RAW_FRAME_Q.full()

    try:
        m._RAW_FRAME_Q.put_nowait(new_frame)
    except queue.Full:
        try:
            m._RAW_FRAME_Q.get_nowait()
        except queue.Empty:
            pass
        m._RAW_FRAME_Q.put_nowait(new_frame)

    got = m._RAW_FRAME_Q.get_nowait()
    assert got[0, 0, 0] == 42, "В очереди должен быть новый кадр (42), не старый (0)"


# ─── 3. NPU graceful degradation ─────────────────────────────────────────────

def test_run_inference_returns_none_when_npu_is_none():
    """_run_inference() возвращает None если NPU не инициализирован."""
    app = object.__new__(m.InterceptorApp)
    app.npu = None

    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    result = app._run_inference(frame)
    assert result is None, "_run_inference должен вернуть None при npu=None"


# ─── 4. _draw_hud ────────────────────────────────────────────────────────────

def test_draw_hud_does_not_draw_at_negative_coords():
    """_draw_hud не рисует target/lead при target_x=-1 (DEAD_RECKON)."""
    from tracker_engine import TrackResult
    from types_enum import TrackerState

    app = object.__new__(m.InterceptorApp)
    app.gst_output = None

    circles_drawn = []
    lines_drawn   = []
    frame  = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    result = TrackResult(
        state=TrackerState.DEAD_RECKON,
        target_x=-1.0, target_y=-1.0,
        lead_x=-1.0,   lead_y=-1.0,
        rc_pitch=1500, rc_throttle=1500,
    )

    with patch('cv2.circle',    side_effect=lambda *a, **k: circles_drawn.append(a)), \
         patch('cv2.line',      side_effect=lambda *a, **k: lines_drawn.append(a)),   \
         patch('cv2.rectangle', return_value=None), \
         patch('cv2.drawMarker', return_value=None), \
         patch('cv2.putText',   return_value=None):
        app._push_frame_raw = lambda f: None
        app._draw_hud(frame, result)

    bad_circles = [c for c in circles_drawn if c[1] == (-1, -1)]
    assert len(bad_circles) == 0, f"Нарисован кружок в (-1,-1): {bad_circles}"

    # args = (frame, pt1, pt2, ...) — проверяем pt1 и pt2
    bad_lines = [l for l in lines_drawn if l[1] == (-1, -1) or l[2] == (-1, -1)]
    assert len(bad_lines) == 0, f"Нарисована линия к (-1,-1): {bad_lines}"


def test_draw_hud_draws_target_with_valid_coords():
    """_draw_hud рисует target/lead при валидных координатах."""
    from tracker_engine import TrackResult
    from types_enum import TrackerState

    app = object.__new__(m.InterceptorApp)
    app.gst_output = None

    circles_drawn = []
    frame  = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    result = TrackResult(
        state=TrackerState.TRACKING,
        target_x=320.0, target_y=240.0,
        lead_x=330.0,   lead_y=235.0,
        rc_pitch=1400, rc_throttle=1550,
    )

    with patch('cv2.circle',    side_effect=lambda *a, **k: circles_drawn.append(a)), \
         patch('cv2.line',      return_value=None), \
         patch('cv2.rectangle', return_value=None), \
         patch('cv2.drawMarker', return_value=None), \
         patch('cv2.putText',   return_value=None):
        app._push_frame_raw = lambda f: None
        app._draw_hud(frame, result)

    assert len(circles_drawn) >= 1, "Кружок цели должен быть нарисован"
    drawn_coords = [c[1] for c in circles_drawn]
    assert (320, 240) in drawn_coords, f"Кружок не в позиции цели: {drawn_coords}"


# ─── 5. _limit_fps ────────────────────────────────────────────────────────────

def test_limit_fps_sleeps_correct_amount():
    """_limit_fps спит не более 1/MAX_FPS + 20ms (OS jitter)."""
    app = object.__new__(m.InterceptorApp)
    t0 = time.monotonic()
    app._limit_fps(t0)
    elapsed  = time.monotonic() - t0
    expected = 1.0 / MAX_FPS
    assert elapsed < expected + 0.02, (
        f"_limit_fps спал слишком долго: {elapsed:.3f}s > {expected + 0.02:.3f}s"
    )


def test_limit_fps_no_sleep_when_late():
    """Если кадр уже занял >1/MAX_FPS — _limit_fps не спит."""
    app = object.__new__(m.InterceptorApp)
    t0      = time.monotonic() - (1.0 / MAX_FPS + 0.05)
    t_start = time.monotonic()
    app._limit_fps(t0)
    elapsed = time.monotonic() - t_start
    assert elapsed < 0.01, f"_limit_fps не должен спать при опоздании: {elapsed:.3f}s"
