import time
import logging
from dataclasses import dataclass
from config import ATTACK_CHANNEL_MIN, ATTACK_CHANNEL_MAX

logger = logging.getLogger(__name__)


@dataclass
class DroneState:
    is_armed:        bool  = False
    attack_switch:   bool  = False
    roll:            float = 0.0
    pitch:           float = 0.0
    yaw:             float = 0.0
    altitude:        float = 0.0
    climb_rate:      float = 0.0
    altitude_valid:  bool  = False
    temperature:     float = 40.0
    battery_voltage: float = 16.0
    timestamp:       float = 0.0


class StateEstimator:
    """
    Разбирает входящие MAVLink сообщения ArduPilot.

    CH10 RadioMaster → RC_CHANNELS.chan10_raw:
      1800-2000 = attack_switch True  → компьютер берёт управление
      иначе     = attack_switch False → оператор управляет с пульта

    VFR_HUD приоритетнее GLOBAL_POSITION_INT для высоты:
      VFR_HUD использует барометр (всегда доступен без GPS).
    """

    def __init__(self):
        self.state            = DroneState()
        self.last_heartbeat   = time.monotonic()
        self._vfr_hud_received = False

    def update_from_message(self, msg):
        if msg is None:
            return
        msg_type = msg.get_type()
        try:
            if msg_type == 'HEARTBEAT':
                self.state.is_armed   = bool(msg.base_mode & 128)
                self.last_heartbeat   = time.monotonic()

            elif msg_type == 'RC_CHANNELS':
                if hasattr(msg, 'chan10_raw'):
                    raw = msg.chan10_raw
                    self.state.attack_switch = (
                        ATTACK_CHANNEL_MIN <= raw <= ATTACK_CHANNEL_MAX
                    )

            elif msg_type == 'ATTITUDE':
                self.state.roll  = msg.roll
                self.state.pitch = msg.pitch
                self.state.yaw   = msg.yaw

            elif msg_type == 'VFR_HUD':
                # Барометрическая высота — всегда доступна
                self.state.altitude       = msg.alt
                self.state.climb_rate     = msg.climb
                self.state.altitude_valid = True
                self._vfr_hud_received    = True

            elif msg_type == 'GLOBAL_POSITION_INT':
                # GPS высота — только если VFR_HUD ещё не пришёл
                if not self._vfr_hud_received:
                    self.state.altitude   = msg.relative_alt / 1000.0
                    # vz NED: + вниз → инвертируем
                    self.state.climb_rate = -msg.vz / 100.0
                    self.state.altitude_valid = True

            elif msg_type == 'SYS_STATUS':
                self.state.battery_voltage = msg.voltage_battery / 1000.0

            elif msg_type == 'SCALED_PRESSURE':
                self.state.temperature = msg.temperature / 100.0

        except Exception as e:
            logger.warning(f"StateEstimator parse error ({msg_type}): {e}")

    def get_state(self) -> DroneState:
        self.state.timestamp = time.monotonic()
        return self.state

    @property
    def heartbeat_age(self) -> float:
        return time.monotonic() - self.last_heartbeat