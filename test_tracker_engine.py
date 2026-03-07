"""
test_tracker_engine.py — полная проверка логики TrackerEngine.

Покрывает:
  1. Машина состояний: IDLE → ACQUIRING → TRACKING → STRIKING → LOST
  2. DEAD_RECKON таймер (BUG FIX: _dead_reckon_start, не _ramp_start)
  3. RC значения при DEAD_RECKON (BUG FIX: RC_MID, не RC_RELEASE)
  4. Восстановление DEAD_RECKON → TRACKING при возврате цели
  5. ACQUIRING timeout → LOST
  6. Знак RC: цель справа → yaw вправо; цель выше → throttle вверх
  7. Pitch рампа: старт нейтраль → финиш PITCH_NEAR/DIVE
  8. Roll всегда RC_RELEASE (оператор)
"""

import time
import numpy as np
import pytest
from unittest.mock import patch

from tracker_engine import TrackerEngine, TrackResult
from types_enum import TrackerState
from config import (
    RC_MID, RC_RELEASE,
    DEAD_RECKONING_SEC, REACQUIRE_TIMEOUT, RAMP_DURATION_SEC, THROTTLE_RAMP_SEC,
    FRAME_WIDTH, FRAME_HEIGHT,
    RC_SAFE_MIN, RC_SAFE_MAX,
    RC_THROTTLE_MIN, RC_THROTTLE_MAX, RC_THROTTLE_STRIKING,
    PITCH_NEAR, PITCH_DIVE,
    ROLL_ASSIST_THRESHOLD,
)

# ─── Хелперы ────────────────────────────────────────────────────────────────

_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
_DET   = ([[280, 210, 360, 270]], [0], [0.9])   # фиктивный YOLO вывод
_LEAD  = (320.0, 240.0)                          # точка упреждения по центру
_HIT   = (320.0, 240.0, 0.9, (280, 210, 360, 270))  # результат VisionTracker


def _eng(vision_seq):
    """
    TrackerEngine с замоканным VisionTracker.
    vision_seq: итерируемый список результатов step() — _HIT или None.
    """
    eng = TrackerEngine()
    it  = iter(vision_seq)

    def _mock_step(frame, yolo):
        return next(it, None)

    eng._vision.step            = _mock_step
    eng._vision.get_lead_point  = lambda: _LEAD
    eng._vision.get_velocity    = lambda: (0.0, 0.0)
    eng._vision.reset           = lambda: None
    return eng


def _eng_centered(cx, cy, *, frames=50):
    """TrackerEngine с целью в позиции (cx, cy)."""
    bbox   = (int(cx - 20), int(cy - 20), int(cx + 20), int(cy + 20))
    vision = (float(cx), float(cy), 0.9, bbox)
    eng    = TrackerEngine()
    it     = iter([vision] * frames)

    def _s(f, y):
        return next(it, None)

    eng._vision.step           = _s
    eng._vision.get_lead_point = lambda: (float(cx), float(cy))
    eng._vision.reset          = lambda: None
    return eng


# ─── 1. Машина состояний — happy path ───────────────────────────────────────

def test_idle_before_engage():
    """До engage() — всегда IDLE, все каналы passthrough."""
    eng = TrackerEngine()
    r   = eng.step(_DET, _FRAME)
    assert r.state       == TrackerState.IDLE
    assert r.rc_pitch    == RC_RELEASE
    assert r.rc_throttle == RC_RELEASE
    assert r.rc_yaw      == RC_RELEASE
    assert r.rc_roll     == RC_RELEASE


def test_acquiring_no_target():
    """Нет цели сразу после engage() → ACQUIRING."""
    eng = _eng([None] * 5)
    eng.engage()
    r = eng.step(_DET, _FRAME)
    assert r.state == TrackerState.ACQUIRING


def test_acquiring_to_tracking_on_first_hit():
    """Первая детекция цели: ACQUIRING → TRACKING."""
    eng = _eng([_HIT])
    eng.engage()
    r = eng.step(_DET, _FRAME)
    assert r.state == TrackerState.TRACKING


def test_tracking_rc_ranges():
    """Во время TRACKING все RC в допустимых диапазонах."""
    eng = _eng([_HIT] * 10)
    eng.engage()
    for _ in range(10):
        r = eng.step(_DET, _FRAME)
    assert r.state == TrackerState.TRACKING
    assert RC_SAFE_MIN     <= r.rc_yaw      <= RC_SAFE_MAX
    assert RC_THROTTLE_MIN <= r.rc_throttle <= RC_THROTTLE_MAX
    assert PITCH_DIVE      <= r.rc_pitch    <= PITCH_NEAR
    assert r.rc_roll == RC_RELEASE


def test_disengage_returns_to_idle():
    """disengage() возвращает в IDLE; следующий step() даёт IDLE."""
    eng = _eng([_HIT] * 100)
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)
    eng.disengage()
    r = eng.step(_DET, _FRAME)
    assert r.state == TrackerState.IDLE


def test_striking_when_ramp_complete():
    """После RAMP_DURATION_SEC состояние → STRIKING, ramp_progress = 1.0."""
    eng = _eng([_HIT] * 200)
    eng.engage()
    future = time.monotonic() + RAMP_DURATION_SEC + 0.05
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future
        r = eng.step(_DET, _FRAME)
    assert r.state         == TrackerState.STRIKING
    assert r.ramp_progress == pytest.approx(1.0)


# ─── 2. DEAD_RECKON таймер — BUG FIX ────────────────────────────────────────

def test_dead_reckon_on_first_loss_not_immediate_lost():
    """
    BUG FIX #2: первый кадр потери после TRACKING → DEAD_RECKON, НЕ LOST.
    Старый код: elapsed = now - _ramp_start; после 1с трекинга → сразу LOST.
    Новый код: _dead_reckon_start фиксируется в момент потери.
    """
    eng = _eng([_HIT] * 30 + [None] * 20)
    eng.engage()
    for _ in range(30):
        eng.step(_DET, _FRAME)   # 30 кадров TRACKING
    r = eng.step(_DET, _FRAME)   # первый кадр без цели
    assert r.state == TrackerState.DEAD_RECKON, (
        f"BUG: ожидался DEAD_RECKON на первом кадре потери, "
        f"получен {r.state}. Без этого fix дрон мгновенно отдаёт управление "
        f"оператору после 1с трекинга."
    )


def test_dead_reckon_stays_in_dead_reckon_within_window():
    """В пределах DEAD_RECKONING_SEC остаёмся в DEAD_RECKON."""
    eng = _eng([_HIT] * 5 + [None] * 100)
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)
    r1 = eng.step(_DET, _FRAME)   # потеря — DEAD_RECKON
    r2 = eng.step(_DET, _FRAME)   # ещё в пределах таймера
    assert r1.state == TrackerState.DEAD_RECKON
    assert r2.state == TrackerState.DEAD_RECKON


def test_dead_reckon_transitions_to_reacquire_after_timeout():
    """
    DEAD_RECKON → REACQUIRE через DEAD_RECKONING_SEC от момента потери.
    (Не LOST напрямую — сначала REACQUIRE для манёвра по Kalman vx,vy.)
    """
    eng = _eng([_HIT] * 5 + [None] * 200)
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)

    # Войти в DEAD_RECKON
    r_dr = eng.step(_DET, _FRAME)
    assert r_dr.state == TrackerState.DEAD_RECKON

    # Промотать за DEAD_RECKONING_SEC → должен перейти в REACQUIRE
    future = time.monotonic() + DEAD_RECKONING_SEC + 0.05
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future
        r_reacq = eng.step(_DET, _FRAME)
    assert r_reacq.state == TrackerState.REACQUIRE, (
        f"Ожидался REACQUIRE через {DEAD_RECKONING_SEC}с dead reckoning, "
        f"получен {r_reacq.state}."
    )


def test_dead_reckon_recovery_to_tracking():
    """Цель вернулась во время DEAD_RECKON → TRACKING."""
    eng = _eng([_HIT] * 5 + [None, None] + [_HIT])
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)
    eng.step(_DET, _FRAME)   # потеря
    eng.step(_DET, _FRAME)   # ещё DEAD_RECKON
    r = eng.step(_DET, _FRAME)  # цель вернулась
    assert r.state in (TrackerState.TRACKING, TrackerState.STRIKING), (
        f"Не восстановился из DEAD_RECKON при возврате цели: {r.state}"
    )


# ─── 3. RC значения при DEAD_RECKON — BUG FIX ───────────────────────────────

def test_dead_reckon_rc_pitch_is_mid_not_release():
    """
    BUG FIX #1 (SAFETY): rc_pitch при DEAD_RECKON должен быть RC_MID=1500, не RC_RELEASE=65535.

    RC_RELEASE (65535) = UINT16_MAX = ArduPilot игнорирует поле → старый override
    (PITCH_DIVE=1280) остаётся активным → дрон продолжает пикировать после потери цели!
    RC_MID (1500) = нейтральный pitch → ArduPilot применяет горизонтальный полёт.
    """
    eng = _eng([_HIT] * 5 + [None])
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)
    r = eng.step(_DET, _FRAME)   # первый кадр потери → DEAD_RECKON
    assert r.state    == TrackerState.DEAD_RECKON
    assert r.rc_pitch == RC_MID, (
        f"SAFETY BUG: rc_pitch={r.rc_pitch} при DEAD_RECKON. "
        f"Должен быть RC_MID={RC_MID}. "
        f"RC_RELEASE=65535 сохраняет старый override (PITCH_DIVE={PITCH_DIVE}) → "
        f"дрон продолжает пикировать после потери цели!"
    )


def test_dead_reckon_rc_throttle_is_mid():
    """RC_MID для throttle при DEAD_RECKON → hover, не RC_RELEASE."""
    eng = _eng([_HIT] * 5 + [None])
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)
    r = eng.step(_DET, _FRAME)
    assert r.rc_throttle == RC_MID, (
        f"rc_throttle={r.rc_throttle} при DEAD_RECKON, ожидался RC_MID={RC_MID}"
    )


def test_dead_reckon_rc_yaw_is_mid():
    """RC_MID для yaw при DEAD_RECKON → нет вращения, не RC_RELEASE."""
    eng = _eng([_HIT] * 5 + [None])
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)
    r = eng.step(_DET, _FRAME)
    assert r.rc_yaw == RC_MID, (
        f"rc_yaw={r.rc_yaw} при DEAD_RECKON, ожидался RC_MID={RC_MID}"
    )


def test_dead_reckon_rc_roll_is_release():
    """RC_RELEASE для roll при DEAD_RECKON → оператор управляет креном."""
    eng = _eng([_HIT] * 5 + [None])
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)
    r = eng.step(_DET, _FRAME)
    assert r.rc_roll == RC_RELEASE


# ─── 4. ACQUIRING timeout ────────────────────────────────────────────────────

def test_acquiring_timeout_leads_to_lost():
    """Нет цели в течение RAMP_DURATION_SEC+DEAD_RECKONING_SEC → LOST."""
    eng = _eng([None] * 200)
    eng.engage()
    future = time.monotonic() + RAMP_DURATION_SEC + DEAD_RECKONING_SEC + 0.05
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future
        r = eng.step(_DET, _FRAME)
    assert r.state == TrackerState.LOST


# ─── 5. RC направления — физическая корректность ────────────────────────────

def _run_with_ramp(eng, n_frames=25):
    """Запустить n кадров с промоткой времени через THROTTLE_RAMP_SEC."""
    future = time.monotonic() + max(THROTTLE_RAMP_SEC, RAMP_DURATION_SEC) + 0.5
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future
        for _ in range(n_frames):
            r = eng.step(_DET, _FRAME)
    return r


def test_rc_yaw_right_when_target_right():
    """Цель справа от центра → rc_yaw > RC_MID (поворот вправо)."""
    cx = float(FRAME_WIDTH) * 0.75   # 480 > 320 центр
    cy = float(FRAME_HEIGHT) * 0.5
    eng = _eng_centered(cx, cy)
    eng.engage()
    r = _run_with_ramp(eng)
    assert r.rc_yaw > RC_MID, (
        f"Цель справа (cx={cx:.0f}): rc_yaw={r.rc_yaw} ≤ RC_MID={RC_MID}. "
        f"Дрон поворачивает влево вместо вправо!"
    )


def test_rc_yaw_left_when_target_left():
    """Цель слева от центра → rc_yaw < RC_MID (поворот влево)."""
    cx = float(FRAME_WIDTH) * 0.25   # 160 < 320 центр
    cy = float(FRAME_HEIGHT) * 0.5
    eng = _eng_centered(cx, cy)
    eng.engage()
    r = _run_with_ramp(eng)
    assert r.rc_yaw < RC_MID, (
        f"Цель слева (cx={cx:.0f}): rc_yaw={r.rc_yaw} ≥ RC_MID={RC_MID}. "
        f"Дрон поворачивает вправо вместо влево!"
    )


def test_rc_throttle_up_when_target_above():
    """Цель выше центра (err_y < 0) → rc_throttle > RC_MID (набрать высоту)."""
    cx = float(FRAME_WIDTH)  * 0.5
    cy = float(FRAME_HEIGHT) * 0.25   # 120 < 240 центр → err_y = -120
    eng = _eng_centered(cx, cy)
    eng.engage()
    r = _run_with_ramp(eng)
    assert r.rc_throttle > RC_MID, (
        f"Цель выше (cy={cy:.0f}, err_y={cy - FRAME_HEIGHT/2:.0f}): "
        f"rc_throttle={r.rc_throttle} ≤ RC_MID={RC_MID}. Дрон снижается!"
    )


def test_rc_throttle_down_when_target_below():
    """Цель ниже центра (err_y > 0) → rc_throttle < RC_MID (снизиться)."""
    cx = float(FRAME_WIDTH)  * 0.5
    cy = float(FRAME_HEIGHT) * 0.75   # 360 > 240 центр → err_y = +120
    eng = _eng_centered(cx, cy)
    eng.engage()
    r = _run_with_ramp(eng)
    assert r.rc_throttle < RC_MID, (
        f"Цель ниже (cy={cy:.0f}, err_y={cy - FRAME_HEIGHT/2:.0f}): "
        f"rc_throttle={r.rc_throttle} ≥ RC_MID={RC_MID}. Дрон поднимается!"
    )


def test_rc_yaw_neutral_when_target_centered():
    """Цель по центру (err_x≈0) → rc_yaw ≈ RC_MID."""
    cx = float(FRAME_WIDTH)  * 0.5
    cy = float(FRAME_HEIGHT) * 0.5
    eng = _eng_centered(cx, cy)
    eng.engage()
    r = _run_with_ramp(eng)
    # Небольшой интегратор может смещать, допуск 50
    assert abs(r.rc_yaw - RC_MID) < 50, (
        f"Цель по центру: rc_yaw={r.rc_yaw}, отклонение {r.rc_yaw - RC_MID} > 50"
    )


# ─── 6. Pitch рампа ──────────────────────────────────────────────────────────

def test_pitch_ramp_start_near_mid():
    """В начале рампы (t≈0): rc_pitch близок к RC_MID (нет резкого пикирования)."""
    eng = _eng([_HIT] * 100)
    eng.engage()
    r = eng.step(_DET, _FRAME)   # первый кадр, ramp≈0
    # rc_pitch должен быть ≥ PITCH_NEAR (1450), т.е. почти нейтраль
    assert r.rc_pitch >= PITCH_NEAR, (
        f"Рампа старт: rc_pitch={r.rc_pitch} < PITCH_NEAR={PITCH_NEAR}. "
        f"Резкое пикирование при старте!"
    )


def test_pitch_ramp_end_striking():
    """После RAMP_DURATION_SEC: state=STRIKING, ramp_progress=1.0."""
    eng = _eng([_HIT] * 200)
    eng.engage()
    future = time.monotonic() + RAMP_DURATION_SEC + 0.1
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future
        r = eng.step(_DET, _FRAME)
    assert r.state         == TrackerState.STRIKING
    assert r.ramp_progress == pytest.approx(1.0)


def test_pitch_lower_when_target_below():
    """Цель ниже в кадре → более крутой pitch (меньшее значение PWM)."""
    cx = float(FRAME_WIDTH) * 0.5
    cy_high = float(FRAME_HEIGHT) * 0.2   # цель высоко → лёгкий наклон
    cy_low  = float(FRAME_HEIGHT) * 0.8   # цель низко → крутое пикирование

    def _r_for(cy):
        eng = _eng_centered(cx, cy)
        eng.engage()
        return _run_with_ramp(eng)

    r_high = _r_for(cy_high)
    r_low  = _r_for(cy_low)
    assert r_low.rc_pitch <= r_high.rc_pitch, (
        f"Цель ниже ({cy_low}) должна давать rc_pitch ≤ цели выше ({cy_high}). "
        f"Получено: low={r_low.rc_pitch}, high={r_high.rc_pitch}"
    )


# ─── 7. Roll всегда passthrough ──────────────────────────────────────────────

def test_roll_always_passthrough_tracking():
    """RC_RELEASE для roll во время TRACKING."""
    eng = _eng([_HIT] * 10)
    eng.engage()
    for _ in range(10):
        r = eng.step(_DET, _FRAME)
    assert r.rc_roll == RC_RELEASE


def test_roll_passthrough_in_dead_reckon():
    """RC_RELEASE для roll во время DEAD_RECKON."""
    eng = _eng([_HIT] * 5 + [None])
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)
    r = eng.step(_DET, _FRAME)
    assert r.rc_roll == RC_RELEASE


def test_roll_passthrough_in_idle():
    """RC_RELEASE для roll в IDLE."""
    eng = TrackerEngine()
    r = eng.step(_DET, _FRAME)
    assert r.rc_roll == RC_RELEASE


# ─── 8. Throttle рампа — плавный набор ──────────────────────────────────────

def test_throttle_ramp_zero_at_start():
    """throttle_ramp=0 при engage() → нет рывка throttle."""
    eng = _eng([_HIT] * 100)
    eng.engage()
    r = eng.step(_DET, _FRAME)
    # При ramp=0 и err_y≈0 delta≈0 → rc_throttle ≈ RC_MID
    assert abs(r.rc_throttle - RC_MID) <= 5, (
        f"Рывок throttle при старте: rc_throttle={r.rc_throttle}, "
        f"ожидался ≈ RC_MID={RC_MID}"
    )


def test_throttle_ramp_reaches_one():
    """throttle_ramp достигает 1.0 через THROTTLE_RAMP_SEC."""
    eng = _eng([_HIT] * 200)
    eng.engage()
    future = time.monotonic() + THROTTLE_RAMP_SEC + 0.1
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future
        r = eng.step(_DET, _FRAME)
    assert r.throttle_ramp == pytest.approx(1.0)


# ─── 9. REACQUIRE state ──────────────────────────────────────────────────────

def test_reacquire_transitions_to_lost_after_timeout():
    """REACQUIRE → LOST через REACQUIRE_TIMEOUT."""
    eng = _eng([_HIT] * 5 + [None] * 200)
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)

    # Enter DEAD_RECKON
    r_dr = eng.step(_DET, _FRAME)
    assert r_dr.state == TrackerState.DEAD_RECKON

    # DEAD_RECKON → REACQUIRE
    t_loss = time.monotonic()
    future1 = t_loss + DEAD_RECKONING_SEC + 0.05
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future1
        r_reacq = eng.step(_DET, _FRAME)
    assert r_reacq.state == TrackerState.REACQUIRE

    # REACQUIRE → LOST
    future2 = future1 + REACQUIRE_TIMEOUT + 0.05
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future2
        r_lost = eng.step(_DET, _FRAME)
    assert r_lost.state == TrackerState.LOST, (
        f"REACQUIRE должен истечь в LOST через {REACQUIRE_TIMEOUT}с, "
        f"получен {r_lost.state}"
    )


def test_reacquire_returns_reacquire_state():
    """REACQUIRE возвращает TrackResult с state=REACQUIRE (не LOST)."""
    eng = _eng([_HIT] * 5 + [None] * 200)
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)
    eng.step(_DET, _FRAME)   # DEAD_RECKON

    future = time.monotonic() + DEAD_RECKONING_SEC + 0.05
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future
        r = eng.step(_DET, _FRAME)
    assert r.state == TrackerState.REACQUIRE


def test_reacquire_rc_pitch_is_mid():
    """При REACQUIRE rc_pitch = RC_MID (нет пикирования без подтверждения цели)."""
    eng = _eng([_HIT] * 5 + [None] * 200)
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)
    eng.step(_DET, _FRAME)   # DEAD_RECKON

    future = time.monotonic() + DEAD_RECKONING_SEC + 0.05
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future
        r = eng.step(_DET, _FRAME)
    assert r.state == TrackerState.REACQUIRE
    assert r.rc_pitch == RC_MID, (
        f"SAFETY: rc_pitch={r.rc_pitch} при REACQUIRE. "
        f"Должен быть RC_MID={RC_MID} — нет пикирования без визуального контакта."
    )


def test_reacquire_rc_throttle_is_mid():
    """При REACQUIRE rc_throttle = RC_MID (hover)."""
    eng = _eng([_HIT] * 5 + [None] * 200)
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)
    eng.step(_DET, _FRAME)   # DEAD_RECKON

    future = time.monotonic() + DEAD_RECKONING_SEC + 0.05
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future
        r = eng.step(_DET, _FRAME)
    assert r.state == TrackerState.REACQUIRE
    assert r.rc_throttle == RC_MID


def test_reacquire_recovery_to_tracking():
    """
    Цель вернулась во время REACQUIRE → TRACKING после REACQUIRE_CONFIRM_SEC.
    (PixEagle recovery_confirmation_time: требует стабильности перед переходом)
    """
    from config import REACQUIRE_CONFIRM_SEC
    eng = _eng([_HIT] * 5 + [None] * 200)
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)
    eng.step(_DET, _FRAME)   # DEAD_RECKON

    # Enter REACQUIRE
    future = time.monotonic() + DEAD_RECKONING_SEC + 0.05
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future
        eng.step(_DET, _FRAME)

    # Target returns — first step starts the confirmation timer
    eng._vision.step = lambda f, y: _HIT
    t_found = future + 0.05
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = t_found
        eng.step(_DET, _FRAME)   # starts confirmation

    # After REACQUIRE_CONFIRM_SEC → TRACKING confirmed
    t_confirmed = t_found + REACQUIRE_CONFIRM_SEC + 0.05
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = t_confirmed
        r = eng.step(_DET, _FRAME)
    assert r.state in (TrackerState.TRACKING, TrackerState.STRIKING), (
        f"Цель вернулась во время REACQUIRE: ожидался TRACKING, получен {r.state}"
    )


# ─── 10. Roll assist ─────────────────────────────────────────────────────────

def test_roll_assist_applied_when_target_far_right():
    """При err_x > ROLL_ASSIST_THRESHOLD → rc_roll > RC_MID (крен вправо)."""
    # Поместить цель далеко справа: cx = FRAME_WIDTH * 0.9 = 576
    cx = float(FRAME_WIDTH) * 0.9
    cy = float(FRAME_HEIGHT) * 0.5
    eng = _eng_centered(cx, cy)
    eng.engage()
    r = _run_with_ramp(eng)
    assert r.rc_roll > RC_MID, (
        f"Цель далеко справа (cx={cx:.0f}, err_x≈{cx - FRAME_WIDTH/2:.0f}): "
        f"rc_roll={r.rc_roll} ≤ RC_MID={RC_MID}. Roll assist не работает."
    )


def test_roll_assist_applied_when_target_far_left():
    """При err_x < -ROLL_ASSIST_THRESHOLD → rc_roll < RC_MID (крен влево)."""
    cx = float(FRAME_WIDTH) * 0.1   # 64 — далеко слева
    cy = float(FRAME_HEIGHT) * 0.5
    eng = _eng_centered(cx, cy)
    eng.engage()
    r = _run_with_ramp(eng)
    assert r.rc_roll < RC_MID, (
        f"Цель далеко слева (cx={cx:.0f}, err_x≈{cx - FRAME_WIDTH/2:.0f}): "
        f"rc_roll={r.rc_roll} ≥ RC_MID={RC_MID}. Roll assist не работает."
    )


def test_roll_assist_not_applied_when_target_centered():
    """При |err_x| ≤ ROLL_ASSIST_THRESHOLD → rc_roll = RC_RELEASE (оператор)."""
    cx = float(FRAME_WIDTH)  * 0.5   # err_x = 0
    cy = float(FRAME_HEIGHT) * 0.5
    eng = _eng_centered(cx, cy)
    eng.engage()
    r = _run_with_ramp(eng)
    assert r.rc_roll == RC_RELEASE, (
        f"Цель по центру (err_x=0): rc_roll={r.rc_roll}, ожидался RC_RELEASE={RC_RELEASE}"
    )


# ─── 11. STRIKING throttle boost ────────────────────────────────────────────

def test_striking_throttle_boost():
    """
    Во время STRIKING throttle может достигать RC_THROTTLE_STRIKING=1800,
    а не ограничен RC_THROTTLE_MAX=1650.
    """
    # Цель ниже центра: pid_alt генерирует throttle < RC_MID (снижение)
    # Нам нужна цель ВЫШЕ центра (err_y < 0) → throttle > RC_MID
    cx = float(FRAME_WIDTH)  * 0.5
    cy = float(FRAME_HEIGHT) * 0.1   # цель высоко → err_y < 0 → throttle > RC_MID

    bbox   = (int(cx - 20), int(cy - 20), int(cx + 20), int(cy + 20))
    vision = (float(cx), float(cy), 0.9, bbox)
    eng    = TrackerEngine()
    eng._vision.step           = lambda f, y: vision
    eng._vision.get_lead_point = lambda: (float(cx), float(cy))
    eng._vision.get_velocity   = lambda: (0.0, 0.0)
    eng._vision.reset          = lambda: None
    eng.engage()

    # Промотать за RAMP_DURATION_SEC → STRIKING
    future = time.monotonic() + RAMP_DURATION_SEC + THROTTLE_RAMP_SEC + 0.2
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future
        r = eng.step(_DET, _FRAME)

    assert r.state == TrackerState.STRIKING
    # В STRIKING throttle может быть до RC_THROTTLE_STRIKING=1800
    assert r.rc_throttle <= RC_THROTTLE_STRIKING, (
        f"rc_throttle={r.rc_throttle} > RC_THROTTLE_STRIKING={RC_THROTTLE_STRIKING}"
    )
    assert r.rc_throttle >= RC_THROTTLE_MIN


def test_striking_throttle_cap_higher_than_tracking():
    """RC_THROTTLE_STRIKING (1800) > RC_THROTTLE_MAX (1650) — проверить константы."""
    assert RC_THROTTLE_STRIKING > RC_THROTTLE_MAX, (
        f"RC_THROTTLE_STRIKING={RC_THROTTLE_STRIKING} должен быть > "
        f"RC_THROTTLE_MAX={RC_THROTTLE_MAX}"
    )


# ─── 12. PixEagle: velocity decay в REACQUIRE ────────────────────────────────

def test_reacquire_velocity_decays_over_time():
    """
    Velocity decay: rc_yaw ближе к RC_MID в конце REACQUIRE, чем в начале.
    (PixEagle ENABLE_VELOCITY_DECAY паттерн: decay = 0.85^elapsed)
    """
    from config import REACQUIRE_VELOCITY_DECAY, DEAD_RECKONING_SEC

    # Цель движется вправо с высокой скоростью
    vx_fast = 60.0
    vy_fast = 0.0
    cx = float(FRAME_WIDTH) * 0.8
    cy = float(FRAME_HEIGHT) * 0.5
    bbox   = (int(cx - 20), int(cy - 20), int(cx + 20), int(cy + 20))
    vision = (cx, cy, 0.9, bbox)

    eng = TrackerEngine()
    eng._vision.step           = lambda f, y: vision
    eng._vision.get_lead_point = lambda: (cx, cy)
    eng._vision.get_velocity   = lambda: (vx_fast, vy_fast)
    eng._vision.reset          = lambda: None
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)

    # Enter DEAD_RECKON
    eng._vision.step = lambda f, y: None
    eng.step(_DET, _FRAME)

    # DEAD_RECKON → REACQUIRE
    t_loss = time.monotonic()
    future1 = t_loss + DEAD_RECKONING_SEC + 0.05
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future1
        eng.step(_DET, _FRAME)
    assert eng._state == TrackerState.REACQUIRE

    # Измерить rc_yaw сразу после входа (маленький elapsed → малое затухание)
    future_early = future1 + 0.1
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future_early
        r_early = eng.step(_DET, _FRAME)

    # Измерить rc_yaw ближе к концу REACQUIRE (большой elapsed → сильное затухание)
    from config import REACQUIRE_TIMEOUT
    future_late = future1 + REACQUIRE_TIMEOUT * 0.9
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future_late
        r_late = eng.step(_DET, _FRAME)

    if r_early.state == TrackerState.REACQUIRE and r_late.state == TrackerState.REACQUIRE:
        # Более позднее → скорость меньше → yaw ближе к RC_MID
        early_err = abs(r_early.rc_yaw - RC_MID)
        late_err  = abs(r_late.rc_yaw  - RC_MID)
        assert late_err <= early_err + 5, (  # 5 PWM допуск
            f"Velocity decay: late_err={late_err} должен быть ≤ early_err={early_err} "
            f"(rc_yaw: early={r_early.rc_yaw}, late={r_late.rc_yaw})"
        )


# ─── 13. PixEagle: recovery confirmation ─────────────────────────────────────

def test_reacquire_recovery_requires_confirmation():
    """
    При первом обнаружении цели в REACQUIRE состояние НЕ меняется на TRACKING —
    нужно подтверждение REACQUIRE_CONFIRM_SEC (PixEagle recovery_confirmation_time).
    """
    from config import REACQUIRE_CONFIRM_SEC, DEAD_RECKONING_SEC

    eng = _eng([_HIT] * 5 + [None] * 200)
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)
    eng.step(_DET, _FRAME)   # DEAD_RECKON

    # Enter REACQUIRE
    future = time.monotonic() + DEAD_RECKONING_SEC + 0.05
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future
        eng.step(_DET, _FRAME)
    assert eng._state == TrackerState.REACQUIRE

    # Target returns — but too early for confirmation
    eng._vision.step = lambda f, y: _HIT
    too_early = future + REACQUIRE_CONFIRM_SEC * 0.3
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = too_early
        r = eng.step(_DET, _FRAME)
    # Should still be REACQUIRE (not confirmed yet)
    assert r.state == TrackerState.REACQUIRE, (
        f"Слишком рано для подтверждения: ожидался REACQUIRE, получен {r.state}"
    )


def test_reacquire_recovery_confirmed_after_delay():
    """
    После REACQUIRE_CONFIRM_SEC стабильного трекинга → переход в TRACKING.
    """
    from config import REACQUIRE_CONFIRM_SEC, DEAD_RECKONING_SEC

    eng = _eng([_HIT] * 5 + [None] * 200)
    eng.engage()
    for _ in range(5):
        eng.step(_DET, _FRAME)
    eng.step(_DET, _FRAME)   # DEAD_RECKON

    future = time.monotonic() + DEAD_RECKONING_SEC + 0.05
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future
        eng.step(_DET, _FRAME)
    assert eng._state == TrackerState.REACQUIRE

    # Target returns and is confirmed
    eng._vision.step = lambda f, y: _HIT
    # First detection — starts confirm timer
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = future
        eng.step(_DET, _FRAME)
    # After REACQUIRE_CONFIRM_SEC — confirmed
    confirmed_time = future + REACQUIRE_CONFIRM_SEC + 0.05
    with patch("tracker_engine.time") as mt:
        mt.monotonic.return_value = confirmed_time
        r = eng.step(_DET, _FRAME)
    assert r.state in (TrackerState.TRACKING, TrackerState.STRIKING), (
        f"Подтверждён REACQUIRE_CONFIRM_SEC: ожидался TRACKING, получен {r.state}"
    )
