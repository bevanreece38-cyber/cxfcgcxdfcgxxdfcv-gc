import os
import logging
from datetime import datetime
from config import LOG_DIR, MAX_LOG_SIZE

logger = logging.getLogger(__name__)


class FlightLogger:
    """CSV лог с автоматической ротацией файла. Flush каждые 50 строк."""

    HEADER = (
        "timestamp,target_x,target_y,confidence,mode,ramp,"
        "err_x,err_y,rc_roll,rc_pitch,rc_throttle,rc_yaw,"
        "altitude,battery_v,armed,mode_str,safety,climb_rate,"
        "temperature,fps,lead_x,lead_y\n"
    )

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.file_path    = os.path.join(
            LOG_DIR,
            f"flight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        self.handle       = None
        self.flush_cnt    = 0
        self._written     = 0   # байт записано в текущий файл (без os.path.getsize)
        self._create()

    def _create(self):
        try:
            # buffering=4096: OS-буфер 4 КБ → сброс каждые ~20 строк (не 30×/сек)
            self.handle   = open(self.file_path, 'w', buffering=4096)
            self._written = len(self.HEADER.encode())
            self.handle.write(self.HEADER)
            self.handle.flush()
            logger.info(f"FlightLogger: {self.file_path}")
        except Exception as e:
            logger.error(f"FlightLogger create failed: {e}")
            self.handle = None

    def write(self, **d):
        if not self.handle:
            return
        try:
            line = (
                f"{d.get('ts',    0):.3f},"
                f"{d.get('tx',   -1):.1f},"
                f"{d.get('ty',   -1):.1f},"
                f"{d.get('conf',  0):.2f},"
                f"{d.get('mode', '')},"
                f"{d.get('ramp',  0):.2f},"
                f"{d.get('err_x', 0):.1f},"
                f"{d.get('err_y', 0):.1f},"
                f"{d.get('rc_r',  0):.0f},"
                f"{d.get('rc_p',  0):.0f},"
                f"{d.get('rc_t',  0):.0f},"
                f"{d.get('rc_y',  0):.0f},"
                f"{d.get('alt',   0):.2f},"
                f"{d.get('batt',  0):.2f},"
                f"{d.get('armed', False)},"
                f"{d.get('mode_str', '')},"
                f"{d.get('safety','OK')},"
                f"{d.get('climb', 0):.2f},"
                f"{d.get('temp',  0):.1f},"
                f"{d.get('fps',   0):.1f},"
                f"{d.get('lead_x',-1):.1f},"
                f"{d.get('lead_y',-1):.1f}\n"
            )
            # Ротация файла по размеру — через in-memory счётчик, без os.path.getsize()
            # os.path.getsize() вызывался при каждой записи (30 Гц = 30 syscall/s)
            if self._written + len(line) > MAX_LOG_SIZE:
                self.handle.close()
                self.file_path = os.path.join(
                    LOG_DIR,
                    f"flight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                )
                self._create()
                if not self.handle:
                    return
            self.handle.write(line)
            self._written  += len(line)
            self.flush_cnt += 1
            if self.flush_cnt >= 50:
                self.handle.flush()
                self.flush_cnt = 0
        except Exception as e:
            logger.error(f"FlightLogger write error: {e}")

    def close(self):
        if self.handle:
            try:
                self.handle.flush()
                self.handle.close()
            except Exception:
                pass
            self.handle = None