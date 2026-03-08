import time
import logging
from pymavlink import mavutil
from config import (
    MIN_ALTITUDE, MAX_DESCENT_RATE,
    BATTERY_CRITICAL, BATTERY_LOW,
    TEMP_CRITICAL, TEMP_WARNING,
    HEARTBEAT_TIMEOUT, NO_FC_TEST_MODE,
)
from types_enum import SafetyStatus

logger = logging.getLogger(__name__)


class SafetyManager:
    def __init__(self, mav_handler):
        self.mav_handler     = mav_handler
        self.last_action     = SafetyStatus.OK
        self.last_action_time = 0.0

    def check(self, state, heartbeat_age: float) -> SafetyStatus:
        if heartbeat_age > HEARTBEAT_TIMEOUT:
            if NO_FC_TEST_MODE or not state.is_armed:
                # Нет FC или не армирован — предупреждение, но не LAND
                logger.warning(
                    f"HEARTBEAT LOST ({heartbeat_age:.1f}s) — no FC / not armed, continuing"
                )
                return SafetyStatus.WARNING
            logger.critical(f"HEARTBEAT LOST ({heartbeat_age:.1f}s) → LAND")
            return SafetyStatus.LAND
        if not state.is_armed:
            return SafetyStatus.OK
        if state.temperature > TEMP_CRITICAL:
            logger.critical(f"TEMP CRITICAL {state.temperature:.1f}°C → LAND")
            return SafetyStatus.LAND
        if state.altitude_valid and state.climb_rate < -MAX_DESCENT_RATE:
            logger.critical(f"DESCENT CRITICAL {state.climb_rate:.2f} m/s → LAND")
            return SafetyStatus.LAND
        if state.battery_voltage > 0 and state.battery_voltage < BATTERY_CRITICAL:
            logger.critical(f"BATTERY CRITICAL {state.battery_voltage:.2f}V → LAND")
            return SafetyStatus.LAND
        if state.altitude_valid and state.altitude < MIN_ALTITUDE:
            logger.warning(f"LOW ALTITUDE {state.altitude:.1f}m → LAND")
            return SafetyStatus.LAND
        if state.battery_voltage > 0 and state.battery_voltage < BATTERY_LOW:
            logger.warning(f"LOW BATTERY {state.battery_voltage:.2f}V")
            return SafetyStatus.WARNING
        if state.temperature > TEMP_WARNING:
            logger.warning(f"HIGH TEMP {state.temperature:.1f}°C")
            return SafetyStatus.WARNING
        return SafetyStatus.OK

    def execute_safety_action(self, status: SafetyStatus, ctrl) -> bool:
        """
        ctrl — ControlManager.
        release_control() вызывается в main ДО этого метода.
        Здесь только MAVLink команда посадки.
        """
        if status in (SafetyStatus.OK, SafetyStatus.WARNING):
            self.last_action = status
            return True
        now = time.monotonic()
        if status == self.last_action and (now - self.last_action_time) < 1.0:
            return True

        logger.critical(f"SAFETY ACTION: {status.name}")
        master = getattr(self.mav_handler, 'master', None)
        if master is None:
            logger.warning("No MAVLink master for safety command")
            self.last_action      = status
            self.last_action_time = now
            return False
        try:
            if status == SafetyStatus.LAND:
                logger.info("→ MAV_CMD_NAV_LAND")
                master.mav.command_long_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_CMD_NAV_LAND,
                    0, 0, 0, 0, 0, 0, 0, 0,
                )
        except Exception as e:
            logger.error(f"Safety MAVLink command failed: {e}")
        self.last_action      = status
        self.last_action_time = now
        return True