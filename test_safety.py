"""
test_safety.py — тесты SafetyManager.

Покрывает все триггеры аварийной посадки:
  - Потеря heartbeat
  - Критический заряд батареи / предупреждение
  - Критическая температура / предупреждение
  - Минимальная высота
  - Критическая скорость снижения
  - Ограничение частоты повторных команд (rate-limit)
"""

import time
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from safety import SafetyManager
from state import DroneState
from types_enum import SafetyStatus


def _mav(sent_commands=None):
    """Фиктивный MAVLink хандлер."""
    cmds = sent_commands if sent_commands is not None else []
    mav_obj = SimpleNamespace(
        command_long_send=lambda *a, **k: cmds.append(a),
    )
    return SimpleNamespace(master=SimpleNamespace(
        mav=mav_obj,
        target_system=1,
        target_component=1,
    ))


def _ok_state(**overrides):
    """Базовое состояние дрона: вооружён, нормальная высота, полная батарея."""
    d = dict(
        is_armed=True,
        altitude=20.0,
        altitude_valid=True,
        battery_voltage=16.0,
        temperature=50.0,
        climb_rate=0.0,
    )
    d.update(overrides)
    return DroneState(**d)


# ─── 1. Нормальные условия ───────────────────────────────────────────────────

def test_ok_when_all_normal():
    sm = SafetyManager(_mav())
    assert sm.check(_ok_state(), heartbeat_age=0.5) == SafetyStatus.OK


def test_ok_when_disarmed():
    """Невооружённый дрон — всегда OK (на земле)."""
    sm = SafetyManager(_mav())
    state = _ok_state(is_armed=False, altitude=0.0, battery_voltage=10.0)
    assert sm.check(state, heartbeat_age=0.5) == SafetyStatus.OK


# ─── 2. LAND триггеры ────────────────────────────────────────────────────────

def test_land_on_heartbeat_loss():
    sm = SafetyManager(_mav())
    from config import HEARTBEAT_TIMEOUT
    result = sm.check(_ok_state(), heartbeat_age=HEARTBEAT_TIMEOUT + 0.1)
    assert result == SafetyStatus.LAND


def test_land_on_critical_battery():
    sm = SafetyManager(_mav())
    from config import BATTERY_CRITICAL
    state = _ok_state(battery_voltage=BATTERY_CRITICAL - 0.1)
    assert sm.check(state, heartbeat_age=0.5) == SafetyStatus.LAND


def test_land_on_critical_temperature():
    sm = SafetyManager(_mav())
    from config import TEMP_CRITICAL
    state = _ok_state(temperature=TEMP_CRITICAL + 1.0)
    assert sm.check(state, heartbeat_age=0.5) == SafetyStatus.LAND


def test_land_on_low_altitude():
    sm = SafetyManager(_mav())
    from config import MIN_ALTITUDE
    state = _ok_state(altitude=MIN_ALTITUDE - 0.5, altitude_valid=True)
    assert sm.check(state, heartbeat_age=0.5) == SafetyStatus.LAND


def test_land_on_critical_descent():
    sm = SafetyManager(_mav())
    from config import MAX_DESCENT_RATE
    state = _ok_state(climb_rate=-(MAX_DESCENT_RATE + 0.5))
    assert sm.check(state, heartbeat_age=0.5) == SafetyStatus.LAND


def test_no_land_when_altitude_invalid():
    """altitude_valid=False → низкая высота не триггерит LAND."""
    sm = SafetyManager(_mav())
    state = _ok_state(altitude=0.0, altitude_valid=False)
    result = sm.check(state, heartbeat_age=0.5)
    assert result in (SafetyStatus.OK, SafetyStatus.WARNING)


# ─── 3. WARNING триггеры ─────────────────────────────────────────────────────

def test_warning_on_low_battery():
    sm = SafetyManager(_mav())
    from config import BATTERY_LOW, BATTERY_CRITICAL
    voltage = (BATTERY_LOW + BATTERY_CRITICAL) / 2   # между LOW и CRITICAL
    state = _ok_state(battery_voltage=voltage)
    assert sm.check(state, heartbeat_age=0.5) == SafetyStatus.WARNING


def test_warning_on_high_temperature():
    sm = SafetyManager(_mav())
    from config import TEMP_WARNING, TEMP_CRITICAL
    temp = (TEMP_WARNING + TEMP_CRITICAL) / 2
    state = _ok_state(temperature=temp)
    assert sm.check(state, heartbeat_age=0.5) == SafetyStatus.WARNING


# ─── 4. Priority: LAND > WARNING ─────────────────────────────────────────────

def test_land_priority_over_warning():
    """Если и батарея критическая, и температура предупреждение — LAND."""
    sm = SafetyManager(_mav())
    from config import BATTERY_CRITICAL, TEMP_WARNING, TEMP_CRITICAL
    state = _ok_state(
        battery_voltage=BATTERY_CRITICAL - 0.1,
        temperature=(TEMP_WARNING + TEMP_CRITICAL) / 2,
    )
    assert sm.check(state, heartbeat_age=0.5) == SafetyStatus.LAND


# ─── 5. execute_safety_action — rate-limit + MAVLink команда ─────────────────

def test_execute_sends_nav_land():
    """SafetyStatus.LAND → MAV_CMD_NAV_LAND отправляется."""
    cmds = []
    sm = SafetyManager(_mav(cmds))
    ctrl = MagicMock()
    sm.execute_safety_action(SafetyStatus.LAND, ctrl)
    assert len(cmds) == 1


def test_execute_rate_limited():
    """Одна и та же команда в пределах 1 сек не отправляется повторно."""
    cmds = []
    sm = SafetyManager(_mav(cmds))
    ctrl = MagicMock()
    sm.execute_safety_action(SafetyStatus.LAND, ctrl)
    sm.execute_safety_action(SafetyStatus.LAND, ctrl)
    sm.execute_safety_action(SafetyStatus.LAND, ctrl)
    assert len(cmds) == 1   # только одна команда


def test_execute_ok_and_warning_are_noop():
    """OK и WARNING не отправляют MAVLink команд."""
    cmds = []
    sm = SafetyManager(_mav(cmds))
    ctrl = MagicMock()
    sm.execute_safety_action(SafetyStatus.OK, ctrl)
    sm.execute_safety_action(SafetyStatus.WARNING, ctrl)
    assert len(cmds) == 0


def test_execute_no_mavlink_master():
    """Если нет MAVLink соединения — не падает, возвращает False."""
    sm = SafetyManager(SimpleNamespace(master=None))
    result = sm.execute_safety_action(SafetyStatus.LAND, MagicMock())
    assert result is False
