from types import SimpleNamespace
from state import StateEstimator


def _msg(**kwargs):
    return SimpleNamespace(**kwargs)


def test_armed():
    est = StateEstimator()
    est.update_from_message(_msg(get_type=lambda: 'HEARTBEAT', base_mode=192))
    assert est.state.is_armed is True


def test_disarmed():
    est = StateEstimator()
    est.update_from_message(_msg(get_type=lambda: 'HEARTBEAT', base_mode=64))
    assert est.state.is_armed is False


def test_attack_on():
    est = StateEstimator()
    est.update_from_message(_msg(get_type=lambda: 'RC_CHANNELS', chan10_raw=1900))
    assert est.state.attack_switch is True


def test_attack_off():
    est = StateEstimator()
    est.update_from_message(_msg(get_type=lambda: 'RC_CHANNELS', chan10_raw=1500))
    assert est.state.attack_switch is False


def test_attack_boundary_min():
    est = StateEstimator()
    est.update_from_message(_msg(get_type=lambda: 'RC_CHANNELS', chan10_raw=1800))
    assert est.state.attack_switch is True


def test_attack_boundary_max():
    est = StateEstimator()
    est.update_from_message(_msg(get_type=lambda: 'RC_CHANNELS', chan10_raw=2000))
    assert est.state.attack_switch is True


def test_attack_over_max():
    est = StateEstimator()
    est.update_from_message(_msg(get_type=lambda: 'RC_CHANNELS', chan10_raw=2001))
    assert est.state.attack_switch is False


def test_vfr_hud():
    est = StateEstimator()
    est.update_from_message(_msg(get_type=lambda: 'VFR_HUD', alt=25.0, climb=1.5))
    assert abs(est.state.altitude - 25.0) < 0.01
    assert abs(est.state.climb_rate - 1.5) < 0.01
    assert est.state.altitude_valid is True


def test_vfr_descending():
    est = StateEstimator()
    est.update_from_message(_msg(get_type=lambda: 'VFR_HUD', alt=10.0, climb=-3.0))
    assert abs(est.state.climb_rate - (-3.0)) < 0.01


def test_vfr_priority_over_gps():
    est = StateEstimator()
    est.update_from_message(_msg(get_type=lambda: 'VFR_HUD', alt=30.0, climb=0.0))
    est.update_from_message(_msg(
        get_type=lambda: 'GLOBAL_POSITION_INT', relative_alt=50000, vz=-100
    ))
    assert abs(est.state.altitude - 30.0) < 0.01


def test_global_position_fallback():
    est = StateEstimator()
    est.update_from_message(_msg(
        get_type=lambda: 'GLOBAL_POSITION_INT', relative_alt=15000, vz=250
    ))
    assert abs(est.state.altitude - 15.0) < 0.01
    assert abs(est.state.climb_rate - (-2.5)) < 0.01
    assert est.state.altitude_valid is True


def test_battery():
    est = StateEstimator()
    est.update_from_message(_msg(get_type=lambda: 'SYS_STATUS', voltage_battery=14800))
    assert abs(est.state.battery_voltage - 14.8) < 0.01


def test_temperature():
    est = StateEstimator()
    est.update_from_message(_msg(get_type=lambda: 'SCALED_PRESSURE', temperature=7500))
    assert abs(est.state.temperature - 75.0) < 0.01


def test_none_safe():
    est = StateEstimator()
    est.update_from_message(None)
    assert est.state.is_armed is False


def test_altitude_invalid_default():
    assert StateEstimator().state.altitude_valid is False


def test_heartbeat_age():
    assert StateEstimator().heartbeat_age >= 0.0


def test_get_state_has_timestamp():
    s = StateEstimator().get_state()
    assert s.timestamp > 0