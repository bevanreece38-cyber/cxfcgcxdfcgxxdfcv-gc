import time
from types import SimpleNamespace
from control_manager import ControlManager, _clamp
from config import RC_RELEASE


class DummyMaster:
    def __init__(self):
        self.sent = []
        self.target_system    = 1
        self.target_component = 1
        self.mav = SimpleNamespace(
            rc_channels_override_send=lambda *a, **k: self.sent.append(list(a))
        )


def _make():
    m = DummyMaster()
    c = ControlManager(SimpleNamespace(master=m))
    return c, m


def test_rc_release_is_passthrough():
    """КРИТИЧНО: RC_RELEASE должен быть 65535, не 0."""
    assert RC_RELEASE == 65535, (
        f"RC_RELEASE={RC_RELEASE} — ОПАСНО! "
        f"0 = минимальный PWM, дрон упадёт. "
        f"Должно быть 65535 (passthrough к ELRS пульту)."
    )


def test_default_no_control():
    c, _ = _make()
    assert c.is_controlling is False


def test_take_control():
    c, _ = _make()
    assert c.take_control() is True
    assert c.is_controlling is True
    c.release_control()


def test_release_sends_passthrough():
    """release_control() должен отправить все каналы = 65535 (passthrough)."""
    c, m = _make()
    c.take_control()
    time.sleep(0.1)
    c.release_control()
    assert len(m.sent) > 0
    last = m.sent[-1]
    channels = last[2:]
    assert all(ch == RC_RELEASE for ch in channels), (
        f"release_control() отправил не passthrough значения: {channels[:4]}"
    )


def test_set_without_take_fails():
    c, m = _make()
    assert c.set_channels(yaw=1600) is False
    assert len(m.sent) == 0


def test_set_after_take():
    c, m = _make()
    c.take_control()
    assert c.set_channels(yaw=1600, throttle=1550, pitch=1400) is True
    c.release_control()


def test_roll_passthrough():
    """Roll канал должен быть RC_RELEASE (оператор управляет креном)."""
    c, m = _make()
    c.take_control()
    c.set_channels(roll=RC_RELEASE, yaw=1600, throttle=1550)
    time.sleep(0.05)
    c.release_control()
    # Найти пакет где установлены yaw и throttle
    yaw_set = [p for p in m.sent if len(p) > 5 and p[5] == 1600]  # CH4=yaw
    for pkt in yaw_set:
        assert pkt[2] == RC_RELEASE, f"Roll должен быть passthrough, got {pkt[2]}"


def test_keepalive_frequency():
    c, m = _make()
    c.take_control()
    time.sleep(0.2)
    c.release_control()
    assert len(m.sent) >= 4, f"Получили только {len(m.sent)} пакетов за 0.2 сек"


def test_double_release_safe():
    c, _ = _make()
    c.take_control()
    c.release_control()
    c.release_control()
    assert c.is_controlling is False


def test_clamp_limits():
    """Значения вне диапазона должны клампиться."""
    c, m = _make()
    c.take_control()
    c.set_channels(yaw=9999, pitch=100)
    time.sleep(0.06)
    c.release_control()
    # Проверяем что yaw не превышает 2200
    for pkt in m.sent:
        if len(pkt) > 5:
            yaw_val = pkt[5]  # CH4 = index 5 (target_sys, target_comp, CH1..CH4...)
            if yaw_val != RC_RELEASE:
                assert yaw_val <= 2200, f"yaw={yaw_val} вышел за 2200"