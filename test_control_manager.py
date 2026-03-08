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
    """КРИТИЧНО: RC_RELEASE должен быть 65535 (UINT16_MAX = passthrough).
    65535 → ArduPilot игнорирует поле → канал управляется ELRS пультом.
    0     → set_override(i,0) → has_override()=false → мгновенный аппаратный RC.
    release_control() отправляет 0, keepalive отправляет 65535.
    """
    assert RC_RELEASE == 65535, (
        f"RC_RELEASE={RC_RELEASE} — должно быть 65535 (UINT16_MAX = passthrough)."
    )


def test_default_no_control():
    c, _ = _make()
    assert c.is_controlling is False


def test_take_control():
    c, _ = _make()
    assert c.take_control() is True
    assert c.is_controlling is True
    c.release_control()


def test_release_sends_zero_for_immediate_handover():
    """
    release_control() должен отправить 0 для CH1-8.

    Подтверждено исходником ArduPilot GCS_Common.cpp:
      CH1-8: 0 → set_override(i,0) → override_value=0 → has_override()=false
             → МГНОВЕННЫЙ возврат к аппаратному RC (ELRS пульт)
      65535 → UINT16_MAX → игнорируется ArduPilot → override остаётся активным
              до истечения RC_OVERRIDE_TIME (~1.5 сек)!
    """
    c, m = _make()
    c.take_control()
    time.sleep(0.1)
    c.release_control()
    assert len(m.sent) > 0
    last = m.sent[-1]
    channels = last[2:]
    assert all(ch == 0 for ch in channels), (
        f"release_control() должен отправить 0 для мгновенного сброса override, "
        f"получили: {channels[:4]}"
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
            if yaw_val != RC_RELEASE and yaw_val != 0:
                assert yaw_val <= 2200, f"yaw={yaw_val} вышел за 2200"


def test_clamp_passthrough():
    """_clamp(65535) должен вернуть 65535 — MAVLink «ignore this field»."""
    assert _clamp(RC_RELEASE) == RC_RELEASE, (
        f"_clamp({RC_RELEASE}) вернул не RC_RELEASE: ожидали {RC_RELEASE}"
    )


def test_clamp_zero_passes_through():
    """
    _clamp(0) должен вернуть 0.

    0 — явный сброс override для CH1-8: ArduPilot вызовет set_override(i,0)
    → override_value=0 → has_override()=false → мгновенный аппаратный RC.
    Нельзя клампить до 800 — это сломает мгновенный release_control().
    """
    assert _clamp(0) == 0, "_clamp(0) должен вернуть 0 (явный сброс override)"


def test_release_triple_send():
    """release_control() отправляет 3 пакета подряд для надёжности."""
    c, m = _make()
    c.take_control()
    before = len(m.sent)
    c.release_control()
    release_pkts = m.sent[before:]
    # Минимум 3 пакета от тройной отправки (keepalive мог добавить ещё)
    assert len(release_pkts) >= 3, (
        f"Ожидали ≥3 пакетов release, получили {len(release_pkts)}"
    )
    # Все release пакеты должны содержать 0
    for pkt in release_pkts:
        channels = pkt[2:]
        assert all(ch == 0 for ch in channels), (
            f"release пакет содержит не 0: {channels[:4]}"
        )