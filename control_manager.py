"""
ControlManager — управление RC по DroneEngage принципам.

https://github.com/DroneEngage/droneengage_mavlink

Ключевые принципы:
  - take_control() / release_control() — явная передача управления
  - Keepalive поток 25 Гц пока override активен
    (ArduPilot требует обновление >3 Гц, иначе снимает override)
  - release = все каналы = RC_RELEASE (65535)
    65535 = passthrough: ArduPilot читает этот канал с ELRS пульта
    ВАЖНО: НЕ 0 ! 0 = минимальный PWM = дрон упадёт!
  - Тройная отправка release для надёжности
"""

import time
import logging
import threading
from typing import Optional

from types_enum import ControlState
from config import RC_RELEASE, KEEPALIVE_HZ, RC_SAFE_MIN, RC_SAFE_MAX

logger = logging.getLogger(__name__)

_NUM_CHANNELS       = 18
_KEEPALIVE_INTERVAL = 1.0 / KEEPALIVE_HZ


def _clamp(value: int) -> int:
    """
    RC_RELEASE (65535) = passthrough: отдать канал оператору.
    Иначе клампим в безопасный диапазон 800-2200.
    """
    if value == RC_RELEASE:
        return RC_RELEASE
    return max(800, min(2200, int(value)))


class ControlManager:
    """
    Управляет передачей управления между оператором (ELRS) и компьютером (Radxa).

    Маппинг ArduPilot AltHold Quad-X:
      CH1 = Roll      CH2 = Pitch
      CH3 = Throttle  CH4 = Yaw
      CH5-CH18 = RC_RELEASE (оператор управляет режимами, ARM и т.д.)

    RC_RELEASE = 65535 = passthrough:
      ArduPilot игнорирует override для этого канала и читает ELRS пульт.
    """

    def __init__(self, mav_handler):
        self._mav    = mav_handler
        self._state  = ControlState.MANUAL
        self._lock   = threading.Lock()
        # Все каналы в passthrough по умолчанию
        self._channels = [RC_RELEASE] * _NUM_CHANNELS
        self._stop_event = threading.Event()
        self._keepalive_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    #  Публичный API
    # ------------------------------------------------------------------

    def take_control(self) -> bool:
        """
        Компьютер берёт управление.
        Запускает keepalive поток 25 Гц.
        ArduPilot удерживает override пока приходят пакеты.
        """
        with self._lock:
            if self._state == ControlState.COMPUTER:
                return True
            if self._get_master() is None:
                logger.error("take_control: нет MAVLink соединения")
                return False
            self._state = ControlState.COMPUTER

        self._stop_event.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop,
            name="rc-keepalive",
            daemon=True,
        )
        self._keepalive_thread.start()
        logger.info("✈ take_control() — компьютер управляет дроном")
        return True

    def release_control(self) -> bool:
        """
        Возвращает управление оператору.
        Все каналы = 65535 (passthrough) → ArduPilot читает ELRS пульт.
        Тройная отправка для надёжности (DroneEngage паттерн).
        """
        self._stop_event.set()
        if self._keepalive_thread:
            self._keepalive_thread.join(timeout=1.0)
            self._keepalive_thread = None

        with self._lock:
            self._state    = ControlState.MANUAL
            # Passthrough на всех каналах = оператор получает управление
            self._channels = [RC_RELEASE] * _NUM_CHANNELS

        # Тройная отправка для надёжности
        for _ in range(3):
            self._send_raw([RC_RELEASE] * _NUM_CHANNELS)
        logger.info("✈ release_control() — управление возвращено оператору (ELRS)")
        return True

    def set_channels(
        self,
        roll:     int = RC_RELEASE,
        pitch:    int = RC_RELEASE,
        throttle: int = RC_RELEASE,
        yaw:      int = RC_RELEASE,
    ) -> bool:
        """
        Установить RC значения каналов 1-4.
        RC_RELEASE (65535) = этот канал читается с ELRS пульта.

        Пример: set_channels(yaw=1600, throttle=1550)
          → yaw и throttle управляются компьютером
          → roll и pitch — оператором с пульта
        """
        if self._state != ControlState.COMPUTER:
            return False
        with self._lock:
            self._channels[0] = _clamp(roll)        # CH1 Roll
            self._channels[1] = _clamp(pitch)       # CH2 Pitch
            self._channels[2] = _clamp(throttle)    # CH3 Throttle
            self._channels[3] = _clamp(yaw)         # CH4 Yaw
            # CH5-CH18 = RC_RELEASE (оператор управляет режимами)
            for i in range(4, _NUM_CHANNELS):
                self._channels[i] = RC_RELEASE
        return True

    @property
    def is_controlling(self) -> bool:
        return self._state == ControlState.COMPUTER

    @property
    def state(self) -> ControlState:
        return self._state

    # ------------------------------------------------------------------
    #  Внутренние методы
    # ------------------------------------------------------------------

    def _keepalive_loop(self):
        """
        Отправляет текущие override значения с частотой KEEPALIVE_HZ.
        ArduPilot снимает override если пакеты прекратились >0.5 сек.
        DroneEngage: keepalive обязателен пока override активен.
        """
        logger.debug(f"Keepalive запущен @ {KEEPALIVE_HZ} Гц")
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            with self._lock:
                ch = list(self._channels)
            if self._state == ControlState.COMPUTER:
                self._send_raw(ch)
            elapsed = time.monotonic() - t0
            wait = _KEEPALIVE_INTERVAL - elapsed
            if wait > 0:
                self._stop_event.wait(timeout=wait)
        logger.debug("Keepalive остановлен")

    def _send_raw(self, channels: list) -> bool:
        master = self._get_master()
        if master is None:
            return False
        try:
            master.mav.rc_channels_override_send(
                master.target_system,
                master.target_component,
                *channels[:_NUM_CHANNELS],
            )
            return True
        except Exception as e:
            logger.error(f"RC override send error: {e}")
            return False

    def _get_master(self):
        return getattr(self._mav, 'master', None)