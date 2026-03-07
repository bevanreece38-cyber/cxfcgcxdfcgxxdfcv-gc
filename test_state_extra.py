"""
test_state.py — дополнительные интеграционные тесты StateEstimator.

Дополняет существующий test_state.py:
  - Корректность разбора всех MAVLink типов
  - VFR_HUD приоритет над GPS (без GPS — только барометр)
  - Атаки: CH10 диапазоны на границах
  - heartbeat_age нарастает
  - Защита от None-сообщений
  - Защита от повреждённых сообщений
"""

import time
import pytest
from types import SimpleNamespace
from state import StateEstimator
from config import ATTACK_CHANNEL_MIN, ATTACK_CHANNEL_MAX


def _msg(msg_type, **fields):
    fields['_type'] = msg_type
    m = SimpleNamespace(**fields)
    m.get_type = lambda: msg_type
    return m


# ─── VFR_HUD / GPS приоритет ────────────────────────────────────────────────

def test_vfr_hud_sets_altitude_and_climb():
    est = StateEstimator()
    est.update_from_message(_msg('VFR_HUD', alt=35.5, climb=1.2))
    s = est.get_state()
    assert s.altitude      == pytest.approx(35.5)
    assert s.climb_rate    == pytest.approx(1.2)
    assert s.altitude_valid is True


def test_gps_ignored_after_vfr_hud():
    """VFR_HUD получен → GPS GLOBAL_POSITION_INT игнорируется."""
    est = StateEstimator()
    est.update_from_message(_msg('VFR_HUD', alt=30.0, climb=0.0))
    est.update_from_message(_msg('GLOBAL_POSITION_INT', relative_alt=5000, vz=0))
    assert est.get_state().altitude == pytest.approx(30.0)


def test_gps_used_before_vfr_hud():
    """До первого VFR_HUD → GPS используется."""
    est = StateEstimator()
    est.update_from_message(_msg('GLOBAL_POSITION_INT', relative_alt=12000, vz=-50))
    s = est.get_state()
    assert s.altitude   == pytest.approx(12.0)   # 12000 mm → 12.0 m
    assert s.climb_rate == pytest.approx(0.5)     # -vz/100 = -(-50)/100 = 0.5


def test_gps_vz_sign_inversion():
    """vz NED: положительный = вниз → climb_rate должен быть отрицательным."""
    est = StateEstimator()
    est.update_from_message(_msg('GLOBAL_POSITION_INT', relative_alt=10000, vz=200))
    s = est.get_state()
    assert s.climb_rate == pytest.approx(-2.0)   # -200/100


# ─── HEARTBEAT ────────────────────────────────────────────────────────────────

def test_heartbeat_armed():
    est = StateEstimator()
    est.update_from_message(_msg('HEARTBEAT', base_mode=0b10000000))
    assert est.get_state().is_armed is True


def test_heartbeat_disarmed():
    est = StateEstimator()
    est.update_from_message(_msg('HEARTBEAT', base_mode=0b00000000))
    assert est.get_state().is_armed is False


def test_heartbeat_age_increases():
    est = StateEstimator()
    est.update_from_message(_msg('HEARTBEAT', base_mode=0))
    time.sleep(0.05)
    assert est.heartbeat_age >= 0.04


# ─── CH10 attack_switch ───────────────────────────────────────────────────────

def test_attack_switch_on_at_min():
    est = StateEstimator()
    est.update_from_message(_msg('RC_CHANNELS', chan10_raw=ATTACK_CHANNEL_MIN))
    assert est.get_state().attack_switch is True


def test_attack_switch_on_at_max():
    est = StateEstimator()
    est.update_from_message(_msg('RC_CHANNELS', chan10_raw=ATTACK_CHANNEL_MAX))
    assert est.get_state().attack_switch is True


def test_attack_switch_off_below_min():
    est = StateEstimator()
    est.update_from_message(_msg('RC_CHANNELS', chan10_raw=ATTACK_CHANNEL_MIN - 1))
    assert est.get_state().attack_switch is False


def test_attack_switch_off_above_max():
    est = StateEstimator()
    est.update_from_message(_msg('RC_CHANNELS', chan10_raw=ATTACK_CHANNEL_MAX + 1))
    assert est.get_state().attack_switch is False


def test_rc_channels_missing_chan10():
    """Если chan10_raw отсутствует — нет AttributeError, attack_switch не меняется."""
    est = StateEstimator()
    msg = _msg('RC_CHANNELS')   # нет chan10_raw
    est.update_from_message(msg)  # не должно бросать исключение
    assert est.get_state().attack_switch is False


# ─── Телеметрия: батарея, температура, attitude ───────────────────────────────

def test_battery_voltage_parsed():
    est = StateEstimator()
    est.update_from_message(_msg('SYS_STATUS', voltage_battery=15200))
    assert est.get_state().battery_voltage == pytest.approx(15.2)


def test_temperature_parsed():
    est = StateEstimator()
    est.update_from_message(_msg('SCALED_PRESSURE', temperature=6500))
    assert est.get_state().temperature == pytest.approx(65.0)


def test_attitude_parsed():
    import math
    est = StateEstimator()
    est.update_from_message(_msg('ATTITUDE',
                                  roll=0.1, pitch=-0.2, yaw=math.pi))
    s = est.get_state()
    assert s.roll  == pytest.approx(0.1)
    assert s.pitch == pytest.approx(-0.2)
    assert s.yaw   == pytest.approx(math.pi)


# ─── Защита от некорректных сообщений ────────────────────────────────────────

def test_none_message_safe():
    est = StateEstimator()
    est.update_from_message(None)   # не должно бросать исключение


def test_corrupted_message_safe():
    """Сообщение без ожидаемых полей — не падает."""
    est = StateEstimator()
    msg = _msg('VFR_HUD')   # нет полей alt и climb
    est.update_from_message(msg)   # не должно бросать AttributeError


def test_get_state_has_timestamp():
    est = StateEstimator()
    s = est.get_state()
    assert s.timestamp > 0
