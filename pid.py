import time
import math
from config import PID_INTEGRAL_LIMIT


class PIDController:
    """
    PID регулятор с:
      - Защитой от integral windup (клампинг интеграла)
      - Без D-spike на первом update после reset
      - Защитой от NaN/Inf на входе и выходе

    Пересчитан для высокоскоростных целей 150-180 км/ч:
      KP_YAW=0.8, KI_YAW=0.008, KD_YAW=0.15
      KP_ALT=1.0, KI_ALT=0.015, KD_ALT=0.2
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_limit: float = PID_INTEGRAL_LIMIT,
    ):
        self.kp             = kp
        self.ki             = ki
        self.kd             = kd
        self.integral_limit = integral_limit
        self.integral       = 0.0
        self.prev_error     = 0.0
        self.prev_time      = time.monotonic()
        self._first_update  = True

    def reset(self):
        self.integral      = 0.0
        self.prev_error    = 0.0
        self.prev_time     = time.monotonic()
        self._first_update = True

    def update(self, error: float) -> float:
        try:
            error = float(error)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(error) or math.isinf(error):
            return 0.0

        now = time.monotonic()
        dt  = max(now - self.prev_time, 0.001)
        self.prev_time = now

        p = self.kp * error

        self.integral += error * dt
        self.integral  = max(-self.integral_limit,
                             min(self.integral_limit, self.integral))
        i = self.ki * self.integral

        if self._first_update:
            d = 0.0
            self._first_update = False
        else:
            d = self.kd * (error - self.prev_error) / dt

        self.prev_error = error
        output = p + i + d
        if math.isnan(output) or math.isinf(output):
            return 0.0
        return output