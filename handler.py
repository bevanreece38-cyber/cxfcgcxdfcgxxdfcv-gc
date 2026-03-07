import time
import logging
from pymavlink import mavutil
from config import (
    SERIAL_PORT, BAUD_RATE,
    CONNECT_RETRIES, CONNECT_RETRY_DELAY,
    RECONNECT_INTERVAL,
)

logger = logging.getLogger(__name__)


class MAVLinkHandler:
    """
    MAVLink соединение к SpeedyBee F405 через UART /dev/ttyS2.
    Поддерживает автоматическое переподключение.
    """

    def __init__(self, port: str = SERIAL_PORT, baud: int = BAUD_RATE):
        self.port           = port
        self.baud           = baud
        self.master         = None
        self.last_reconnect = 0.0
        self._connect()

    def _connect(self):
        for attempt in range(1, CONNECT_RETRIES + 1):
            try:
                logger.info(f"MAVLink connect {attempt}/{CONNECT_RETRIES} → {self.port}")
                self.master = mavutil.mavlink_connection(
                    self.port, baud=self.baud, timeout=5
                )
                self.master.wait_heartbeat(timeout=5)
                logger.info(
                    f"MAVLink OK  sys={self.master.target_system} "
                    f"comp={self.master.target_component}"
                )
                self.last_reconnect = time.monotonic()
                return
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                self.master = None
                time.sleep(CONNECT_RETRY_DELAY)
        logger.error("All MAVLink connect attempts failed")

    def ensure_connection(self):
        if self.master is None and (
            time.monotonic() - self.last_reconnect > RECONNECT_INTERVAL
        ):
            logger.info("MAVLink reconnect...")
            self._connect()

    def receive_message(self):
        if not self.master:
            return None
        try:
            return self.master.recv_match(blocking=False, timeout=0.01)
        except Exception:
            return None

    def release(self):
        if self.master:
            try:
                self.master.close()
                logger.info("MAVLink connection closed")
            except Exception:
                pass
            self.master = None