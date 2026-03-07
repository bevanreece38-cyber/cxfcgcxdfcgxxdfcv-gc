"""
test_tracker_engine.py — Version 13

Тесты для TrackerEngine (PixEagle архитектура):
  - Машина состояний (IDLE → ACQUIRING → TRACKING → STRIKING / DEAD_RECKON → LOST)
  - RC значения: roll всегда RC_RELEASE, yaw/throttle/pitch в допустимых диапазонах
  - Рампы: throttle_ramp и ramp_progress нарастают плавно
  - Predictive intercept: err_x/err_y считаются от точки упреждения
  - Диагностика TrackResult

VisionTracker мокируется через unittest.mock.patch чтобы тесты не зависели
от OpenCV CSRT и NPU hardware.
"""

import time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from tracker_engine import TrackerEngine, TrackResult, _idle_result, _lost_result
from types_enum import TrackerState
from config import (
    RC_RELEASE, RC_MID, RC_SAFE_MIN, RC_SAFE_MAX,
    RC_THROTTLE_MIN, RC_THROTTLE_MAX,
    PITCH_NEAR, PITCH_DIVE,
    RAMP_DURATION_SEC, THROTTLE_RAMP_SEC,
    DEAD_RECKONING_SEC,
    FRAME_WIDTH, FRAME_HEIGHT,
)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _blank_frame():
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def _det(cx=320, cy=240, conf=0.9, size=40):
    """Создать YOLO-вывод с одной детекцией."""
    x1, y1 = int(cx - size // 2), int(cy - size // 2)
    x2, y2 = int(cx + size // 2), int(cy + size // 2)
    return ([[x1, y1, x2, y2]], [0], [conf])


def _make_engine_tracking(cx=320, cy=240, lead_x=320.0, lead_y=240.0):
    """
    Создать TrackerEngine в состоянии TRACKING с одной мокированной детекцией.
    Возвращает (engine, result).
    """
    engine = TrackerEngine()
    engine.engage()
    frame = _blank_frame()

    with patch.object(engine._vision, 'step',
                      return_value=(float(cx), float(cy), 0.9,
                                    (cx - 20, cy - 20, cx + 20, cy + 20))):
        with patch.object(engine._vision, 'get_lead_point',
                          return_value=(lead_x, lead_y)):
            result = engine.step(_det(cx, cy), frame)

    return engine, result


# ---------------------------------------------------------------------------
# Тесты машины состояний
# ---------------------------------------------------------------------------

def test_initial_state_is_idle():
    """TrackerEngine стартует в IDLE."""
    engine = TrackerEngine()
    assert engine.state == TrackerState.IDLE


def test_engage_sets_acquiring():
    """engage() переводит в ACQUIRING."""
    engine = TrackerEngine()
    engine.engage()
    assert engine.state == TrackerState.ACQUIRING


def test_disengage_resets_to_idle():
    """disengage() возвращает в IDLE из любого состояния."""
    engine = TrackerEngine()
    engine.engage()
    engine.disengage()
    assert engine.state == TrackerState.IDLE


def test_idle_step_returns_idle_result():
    """step() в IDLE возвращает TrackResult(state=IDLE) без вызова vision."""
    engine = TrackerEngine()
    result = engine.step(None, _blank_frame())
    assert result.state == TrackerState.IDLE


def test_detection_transitions_to_tracking():
    """Первая детекция переводит из ACQUIRING в TRACKING."""
    _, result = _make_engine_tracking()
    assert result.state == TrackerState.TRACKING


def test_disengage_from_tracking_resets_idle():
    """disengage() из TRACKING → IDLE."""
    engine, _ = _make_engine_tracking()
    engine.disengage()
    assert engine.state == TrackerState.IDLE


def test_lost_vision_transitions_to_dead_reckon():
    """Потеря vision при TRACKING → DEAD_RECKON."""
    engine = TrackerEngine()
    engine.engage()
    frame = _blank_frame()

    # Сначала получить детекцию → TRACKING
    with patch.object(engine._vision, 'step',
                      return_value=(320.0, 240.0, 0.9, (300, 220, 340, 260))):
        with patch.object(engine._vision, 'get_lead_point',
                          return_value=(320.0, 240.0)):
            engine.step(_det(), frame)

    assert engine.state == TrackerState.TRACKING

    # Теперь потеря цели → DEAD_RECKON
    with patch.object(engine._vision, 'step', return_value=None):
        result = engine.step(None, frame)

    assert result.state == TrackerState.DEAD_RECKON


def test_prolonged_loss_transitions_to_lost():
    """
    После RAMP_DURATION_SEC + DEAD_RECKONING_SEC без цели → LOST.
    Манипулируем _ramp_start чтобы не ждать реального времени.
    """
    engine = TrackerEngine()
    engine.engage()
    frame = _blank_frame()

    # Сначала трекинг
    with patch.object(engine._vision, 'step',
                      return_value=(320.0, 240.0, 0.9, (300, 220, 340, 260))):
        with patch.object(engine._vision, 'get_lead_point',
                          return_value=(320.0, 240.0)):
            engine.step(_det(), frame)

    # Сдвинуть _ramp_start назад чтобы прошло RAMP_DURATION_SEC + DEAD_RECKONING_SEC
    engine._ramp_start -= (RAMP_DURATION_SEC + DEAD_RECKONING_SEC + 0.1)

    with patch.object(engine._vision, 'step', return_value=None):
        result = engine.step(None, frame)

    assert result.state == TrackerState.LOST


def test_full_ramp_transitions_to_striking():
    """
    Когда рампа пикирования = 100% → STRIKING.
    """
    engine = TrackerEngine()
    engine.engage()
    frame = _blank_frame()

    # Сдвинуть _ramp_start назад на RAMP_DURATION_SEC чтобы рампа = 1.0
    engine._ramp_start -= (RAMP_DURATION_SEC + 0.1)

    with patch.object(engine._vision, 'step',
                      return_value=(320.0, 240.0, 0.9, (300, 220, 340, 260))):
        with patch.object(engine._vision, 'get_lead_point',
                          return_value=(320.0, 240.0)):
            result = engine.step(_det(), frame)

    assert result.state == TrackerState.STRIKING
    assert result.ramp_progress >= 1.0


# ---------------------------------------------------------------------------
# Тесты RC значений
# ---------------------------------------------------------------------------

def test_roll_always_passthrough():
    """rc_roll всегда = RC_RELEASE (65535) — оператор управляет креном."""
    _, result = _make_engine_tracking()
    assert result.rc_roll == RC_RELEASE, (
        f"rc_roll={result.rc_roll}, ожидалось {RC_RELEASE}"
    )


def test_rc_roll_in_idle_is_passthrough():
    """rc_roll в состоянии IDLE тоже RC_RELEASE."""
    engine = TrackerEngine()
    result = engine.step(None, _blank_frame())
    assert result.rc_roll == RC_RELEASE


def test_yaw_in_valid_range():
    """rc_yaw должен быть в [RC_SAFE_MIN, RC_SAFE_MAX] или RC_RELEASE."""
    _, result = _make_engine_tracking()
    if result.rc_yaw != RC_RELEASE:
        assert RC_SAFE_MIN <= result.rc_yaw <= RC_SAFE_MAX, (
            f"rc_yaw={result.rc_yaw} выходит за [{RC_SAFE_MIN}, {RC_SAFE_MAX}]"
        )


def test_throttle_in_valid_range():
    """rc_throttle в [RC_THROTTLE_MIN, RC_THROTTLE_MAX] или RC_RELEASE."""
    _, result = _make_engine_tracking()
    if result.rc_throttle != RC_RELEASE:
        assert RC_THROTTLE_MIN <= result.rc_throttle <= RC_THROTTLE_MAX, (
            f"rc_throttle={result.rc_throttle} выходит за [{RC_THROTTLE_MIN}, {RC_THROTTLE_MAX}]"
        )


def test_pitch_in_valid_range():
    """rc_pitch в [PITCH_DIVE, PITCH_NEAR] или RC_RELEASE."""
    _, result = _make_engine_tracking()
    if result.rc_pitch != RC_RELEASE:
        assert PITCH_DIVE <= result.rc_pitch <= PITCH_NEAR, (
            f"rc_pitch={result.rc_pitch} выходит за [{PITCH_DIVE}, {PITCH_NEAR}]"
        )


def test_release_rc_on_lost():
    """При потере цели (LOST) все RC каналы = RC_RELEASE."""
    engine = TrackerEngine()
    engine.engage()
    frame = _blank_frame()

    engine._ramp_start -= (RAMP_DURATION_SEC + DEAD_RECKONING_SEC + 0.1)

    with patch.object(engine._vision, 'step', return_value=None):
        result = engine.step(None, frame)

    assert result.rc_roll     == RC_RELEASE
    assert result.rc_pitch    == RC_RELEASE
    assert result.rc_throttle == RC_RELEASE
    assert result.rc_yaw      == RC_RELEASE


def test_release_rc_in_idle():
    """В IDLE все RC каналы = RC_RELEASE."""
    result = TrackerEngine().step(None, _blank_frame())
    assert result.rc_roll     == RC_RELEASE
    assert result.rc_pitch    == RC_RELEASE
    assert result.rc_throttle == RC_RELEASE
    assert result.rc_yaw      == RC_RELEASE


def test_yaw_right_when_target_right():
    """Цель правее центра → yaw > RC_MID (поворот вправо)."""
    # Цель на 100 px правее центра
    cx = FRAME_WIDTH // 2 + 100
    _, result = _make_engine_tracking(cx=cx, lead_x=float(cx))
    if result.rc_yaw != RC_RELEASE:
        assert result.rc_yaw > RC_MID, (
            f"Цель справа, rc_yaw={result.rc_yaw} должен быть > {RC_MID}"
        )


def test_yaw_left_when_target_left():
    """Цель левее центра → yaw < RC_MID (поворот влево)."""
    cx = FRAME_WIDTH // 2 - 100
    _, result = _make_engine_tracking(cx=cx, lead_x=float(cx))
    if result.rc_yaw != RC_RELEASE:
        assert result.rc_yaw < RC_MID, (
            f"Цель слева, rc_yaw={result.rc_yaw} должен быть < {RC_MID}"
        )


def test_throttle_up_when_target_above():
    """
    Цель выше центра → err_y < 0 → throttle > RC_MID (подъём).
    throttle_ramp в начале ≈ 0, поэтому нужно подождать немного.
    """
    engine = TrackerEngine()
    engine.engage()
    # Продвинуть throttle рампу: 0.1 сек
    engine._throttle_ramp_start -= 0.1
    cy = FRAME_HEIGHT // 2 - 100   # выше центра
    frame = _blank_frame()

    with patch.object(engine._vision, 'step',
                      return_value=(320.0, float(cy), 0.9, (300, cy - 20, 340, cy + 20))):
        with patch.object(engine._vision, 'get_lead_point',
                          return_value=(320.0, float(cy))):
            result = engine.step(_det(cy=cy), frame)

    if result.rc_throttle != RC_RELEASE:
        assert result.rc_throttle >= RC_MID, (
            f"Цель выше, rc_throttle={result.rc_throttle} должен быть >= {RC_MID}"
        )


def test_throttle_down_when_target_below():
    """Цель ниже центра → err_y > 0 → throttle < RC_MID (снижение)."""
    engine = TrackerEngine()
    engine.engage()
    engine._throttle_ramp_start -= 0.1
    cy = FRAME_HEIGHT // 2 + 100   # ниже центра
    frame = _blank_frame()

    with patch.object(engine._vision, 'step',
                      return_value=(320.0, float(cy), 0.9, (300, cy - 20, 340, cy + 20))):
        with patch.object(engine._vision, 'get_lead_point',
                          return_value=(320.0, float(cy))):
            result = engine.step(_det(cy=cy), frame)

    if result.rc_throttle != RC_RELEASE:
        assert result.rc_throttle <= RC_MID, (
            f"Цель ниже, rc_throttle={result.rc_throttle} должен быть <= {RC_MID}"
        )


# ---------------------------------------------------------------------------
# Тесты рамп (throttle_ramp и ramp_progress)
# ---------------------------------------------------------------------------

def test_throttle_ramp_starts_near_zero():
    """throttle_ramp в начале атаки близко к 0."""
    engine = TrackerEngine()
    engine.engage()
    frame = _blank_frame()

    with patch.object(engine._vision, 'step',
                      return_value=(320.0, 240.0, 0.9, (300, 220, 340, 260))):
        with patch.object(engine._vision, 'get_lead_point',
                          return_value=(320.0, 240.0)):
            result = engine.step(_det(), frame)

    # Рампа только запущена: throttle_ramp должна быть < 0.1
    # (прошло ~0 мс с момента engage())
    assert result.throttle_ramp < 0.2, (
        f"throttle_ramp={result.throttle_ramp} слишком велика в начале"
    )


def test_throttle_ramp_increases_with_time():
    """throttle_ramp нарастает со временем (0 → 1 за THROTTLE_RAMP_SEC)."""
    engine = TrackerEngine()
    engine.engage()
    frame = _blank_frame()

    # Продвинуть дату на половину THROTTLE_RAMP_SEC
    engine._throttle_ramp_start -= THROTTLE_RAMP_SEC / 2

    with patch.object(engine._vision, 'step',
                      return_value=(320.0, 240.0, 0.9, (300, 220, 340, 260))):
        with patch.object(engine._vision, 'get_lead_point',
                          return_value=(320.0, 240.0)):
            result = engine.step(_det(), frame)

    assert result.throttle_ramp > 0.3, (
        f"throttle_ramp={result.throttle_ramp} должна быть > 0.3 на половине рампы"
    )


def test_throttle_ramp_caps_at_one():
    """throttle_ramp не превышает 1.0."""
    engine = TrackerEngine()
    engine.engage()
    # Прошло гораздо больше THROTTLE_RAMP_SEC
    engine._throttle_ramp_start -= THROTTLE_RAMP_SEC * 5
    frame = _blank_frame()

    with patch.object(engine._vision, 'step',
                      return_value=(320.0, 240.0, 0.9, (300, 220, 340, 260))):
        with patch.object(engine._vision, 'get_lead_point',
                          return_value=(320.0, 240.0)):
            result = engine.step(_det(), frame)

    assert result.throttle_ramp == pytest.approx(1.0, abs=1e-6)


def test_pitch_ramp_starts_near_zero():
    """ramp_progress в начале атаки близко к 0."""
    _, result = _make_engine_tracking()
    assert result.ramp_progress < 0.2


def test_pitch_ramp_increases_with_time():
    """ramp_progress нарастает к 1.0 за RAMP_DURATION_SEC."""
    engine = TrackerEngine()
    engine.engage()
    engine._ramp_start -= RAMP_DURATION_SEC / 2
    frame = _blank_frame()

    with patch.object(engine._vision, 'step',
                      return_value=(320.0, 240.0, 0.9, (300, 220, 340, 260))):
        with patch.object(engine._vision, 'get_lead_point',
                          return_value=(320.0, 240.0)):
            result = engine.step(_det(), frame)

    assert result.ramp_progress > 0.3


def test_pitch_ramp_caps_at_one():
    """ramp_progress не превышает 1.0."""
    engine = TrackerEngine()
    engine.engage()
    engine._ramp_start -= RAMP_DURATION_SEC * 5
    frame = _blank_frame()

    with patch.object(engine._vision, 'step',
                      return_value=(320.0, 240.0, 0.9, (300, 220, 340, 260))):
        with patch.object(engine._vision, 'get_lead_point',
                          return_value=(320.0, 240.0)):
            result = engine.step(_det(), frame)

    assert result.ramp_progress == pytest.approx(1.0, abs=1e-6)


def test_pitch_decreases_toward_dive_with_ramp():
    """
    При нарастании рампы rc_pitch уменьшается от RC_MID до PITCH_DIVE.
    """
    engine = TrackerEngine()
    engine.engage()
    frame = _blank_frame()

    # Нет рампы: питч близок к RC_MID
    with patch.object(engine._vision, 'step',
                      return_value=(320.0, 240.0, 0.9, (300, 220, 340, 260))):
        with patch.object(engine._vision, 'get_lead_point',
                          return_value=(320.0, 240.0)):
            result_early = engine.step(_det(), frame)

    # Полная рампа: питч = PITCH_DIVE (для цели в центре кадра)
    engine._ramp_start -= RAMP_DURATION_SEC * 2
    with patch.object(engine._vision, 'step',
                      return_value=(320.0, 240.0, 0.9, (300, 220, 340, 260))):
        with patch.object(engine._vision, 'get_lead_point',
                          return_value=(320.0, 240.0)):
            result_late = engine.step(_det(), frame)

    assert result_late.rc_pitch <= result_early.rc_pitch, (
        "rc_pitch должен уменьшаться при нарастании рампы"
    )


# ---------------------------------------------------------------------------
# Тесты predictive intercept (точка упреждения)
# ---------------------------------------------------------------------------

def test_err_x_from_lead_point():
    """err_x = lead_x - frame_center_x (горизонтальная ошибка)."""
    frame_cx = FRAME_WIDTH / 2.0
    lead_x = 400.0   # правее центра
    _, result = _make_engine_tracking(cx=320, lead_x=lead_x)
    expected_err_x = lead_x - frame_cx
    assert abs(result.err_x - expected_err_x) < 1.0, (
        f"err_x={result.err_x:.2f}, ожидалось {expected_err_x:.2f}"
    )


def test_err_y_from_lead_point():
    """err_y = lead_y - frame_center_y (вертикальная ошибка)."""
    frame_cy = FRAME_HEIGHT / 2.0
    lead_y = 300.0   # ниже центра
    _, result = _make_engine_tracking(cy=240, lead_y=lead_y)
    expected_err_y = lead_y - frame_cy
    assert abs(result.err_y - expected_err_y) < 1.0, (
        f"err_y={result.err_y:.2f}, ожидалось {expected_err_y:.2f}"
    )


def test_zero_error_when_lead_at_center():
    """Если точка упреждения в центре кадра → err_x = err_y = 0."""
    _, result = _make_engine_tracking(
        cx=FRAME_WIDTH // 2,  cy=FRAME_HEIGHT // 2,
        lead_x=float(FRAME_WIDTH // 2),  lead_y=float(FRAME_HEIGHT // 2),
    )
    assert abs(result.err_x) < 1.0
    assert abs(result.err_y) < 1.0


def test_lead_x_lead_y_stored_in_result():
    """lead_x и lead_y из VisionTracker сохраняются в TrackResult."""
    lead_x, lead_y = 380.0, 200.0
    _, result = _make_engine_tracking(lead_x=lead_x, lead_y=lead_y)
    assert abs(result.lead_x - lead_x) < 1.0
    assert abs(result.lead_y - lead_y) < 1.0


# ---------------------------------------------------------------------------
# Тесты TrackResult dataclass
# ---------------------------------------------------------------------------

def test_idle_result_factory():
    r = _idle_result()
    assert r.state == TrackerState.IDLE
    assert r.rc_roll == RC_RELEASE


def test_lost_result_factory():
    r = _lost_result(ramp=0.5)
    assert r.state == TrackerState.LOST
    assert r.ramp_progress == pytest.approx(0.5)


def test_track_result_defaults():
    """TrackResult с дефолтными значениями корректен."""
    r = TrackResult()
    assert r.state         == TrackerState.IDLE
    assert r.rc_roll       == RC_RELEASE
    assert r.rc_pitch      == RC_RELEASE
    assert r.rc_throttle   == RC_RELEASE
    assert r.rc_yaw        == RC_RELEASE
    assert r.ramp_progress == 0.0
    assert r.throttle_ramp == 0.0
    assert r.target_x      == -1.0
    assert r.target_y      == -1.0
    assert r.lead_x        == -1.0
    assert r.lead_y        == -1.0


def test_confidence_passed_through():
    """confidence из детекции сохраняется в TrackResult."""
    engine = TrackerEngine()
    engine.engage()
    frame = _blank_frame()
    expected_conf = 0.87

    with patch.object(engine._vision, 'step',
                      return_value=(320.0, 240.0, expected_conf, (300, 220, 340, 260))):
        with patch.object(engine._vision, 'get_lead_point',
                          return_value=(320.0, 240.0)):
            result = engine.step(_det(), frame)

    assert abs(result.confidence - expected_conf) < 1e-6


# ---------------------------------------------------------------------------
# Тесты повторного engage/disengage
# ---------------------------------------------------------------------------

def test_reengage_resets_ramps():
    """После disengage + повторного engage рампы сбрасываются в 0."""
    engine = TrackerEngine()
    engine.engage()
    # Прокрутить рампу
    engine._ramp_start -= RAMP_DURATION_SEC
    engine._throttle_ramp_start -= THROTTLE_RAMP_SEC

    engine.disengage()
    engine.engage()

    assert engine._ramp_progress == 0.0
    assert engine._throttle_ramp == 0.0


def test_engage_twice_stays_acquiring():
    """Двойной engage() не ломает состояние — остаётся ACQUIRING."""
    engine = TrackerEngine()
    engine.engage()
    engine.engage()
    assert engine.state == TrackerState.ACQUIRING


def test_step_idle_does_not_call_vision():
    """В IDLE шаг не вызывает VisionTracker.step()."""
    engine = TrackerEngine()
    with patch.object(engine._vision, 'step') as mock_step:
        engine.step(None, _blank_frame())
        mock_step.assert_not_called()
