import time
import threading
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
    Поддерживает автоматическое переподключение в фоновом потоке
    (не блокирует основной цикл — критично для боевого режима).
    """

    def __init__(self, port: str = SERIAL_PORT, baud: int = BAUD_RATE):
        self.port           = port
        self.baud           = baud
        self.master         = None
        self.last_reconnect = 0.0
        self._reconnecting  = False
        self._connect()

    def _connect(self):
        """Инициализация/переподключение. Вызывается из фонового потока при reconnect."""
        self.last_reconnect = time.monotonic()   # защита от повторного входа
        for attempt in range(1, CONNECT_RETRIES + 1):
            try:
                logger.info(f"MAVLink connect {attempt}/{CONNECT_RETRIES} → {self.port}")
                master = mavutil.mavlink_connection(
                    self.port, baud=self.baud, timeout=5
                )
                master.wait_heartbeat(timeout=5)
                self.master = master           # атомарное присвоение
                logger.info(
                    f"MAVLink OK  sys={self.master.target_system} "
                    f"comp={self.master.target_component}"
                )
                return
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                self.master = None
                time.sleep(CONNECT_RETRY_DELAY)
        logger.error("All MAVLink connect attempts failed")

    def ensure_connection(self):
        """
        Вызывается из основного цикла каждый тик.
        НЕ блокирует цикл — переподключение идёт в фоновом потоке.
        Защита от одновременного запуска двух reconnect-потоков.
        """
        if (
            self.master is None
            and not self._reconnecting
            and (time.monotonic() - self.last_reconnect > RECONNECT_INTERVAL)
        ):
            self._reconnecting = True
            logger.info("MAVLink: запускаем reconnect в фоновом потоке...")
            t = threading.Thread(
                target=self._reconnect_worker,
                name="mavlink-reconnect",
                daemon=True,
            )
            t.start()

    def _reconnect_worker(self):
        try:
            self._connect()
        finally:
            self._reconnecting = False

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