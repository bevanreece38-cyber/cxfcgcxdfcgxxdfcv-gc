import time
import math
from pid import PIDController


def test_proportional():
    pid = PIDController(1.0, 0.0, 0.0)
    out = pid.update(10.0)
    assert abs(out - 10.0) < 0.01


def test_nan_input():
    pid = PIDController(1.0, 0.1, 0.1)
    assert pid.update(float('nan')) == 0.0


def test_inf_input():
    pid = PIDController(1.0, 0.1, 0.1)
    assert pid.update(float('inf')) == 0.0


def test_anti_windup():
    pid = PIDController(0.0, 1.0, 0.0, integral_limit=10.0)
    for _ in range(300):
        pid.update(100.0)
        time.sleep(0.001)
    assert abs(pid.integral) <= 10.0 + 1e-6


def test_reset_clears_state():
    pid = PIDController(1.0, 1.0, 1.0)
    pid.update(50.0)
    pid.reset()
    assert pid.integral == 0.0
    assert pid.prev_error == 0.0
    assert pid._first_update is True


def test_no_d_spike_after_reset():
    pid = PIDController(kp=1.0, ki=0.0, kd=10.0)
    pid.reset()
    time.sleep(0.001)
    out = pid.update(100.0)
    # Только P компонента: 1.0 * 100 = 100.0 (без D)
    assert abs(out - 100.0) < 1.0


def test_d_works_on_second_update():
    pid = PIDController(kp=0.0, ki=0.0, kd=1.0)
    pid.reset()
    pid.update(0.0)
    time.sleep(0.01)
    out = pid.update(10.0)
    assert out != 0.0


def test_zero_error():
    assert PIDController(1.0, 0.0, 0.0).update(0.0) == 0.0


def test_negative_error():
    assert PIDController(1.0, 0.0, 0.0).update(-5.0) < 0.0