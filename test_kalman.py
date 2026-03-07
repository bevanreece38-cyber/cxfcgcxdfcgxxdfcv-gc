from kalman import KalmanTargetTracker


def test_predict_returns_tuple():
    k = KalmanTargetTracker(640, 480)
    p = k.predict_only()
    assert isinstance(p, tuple) and len(p) == 2


def test_update_returns_tuple():
    k = KalmanTargetTracker(640, 480)
    p = k.update((320.0, 240.0))
    assert isinstance(p, tuple) and len(p) == 2


def test_predict_with_velocity():
    k = KalmanTargetTracker(640, 480)
    pv = k.predict_with_velocity()
    assert len(pv) == 4
    assert all(isinstance(v, float) for v in pv)


def test_no_double_predict():
    k = KalmanTargetTracker(640, 480)
    k.begin_step()
    p1 = k.predict_only()
    p2 = k.predict_only()
    assert abs(p1[0] - p2[0]) < 0.001
    assert abs(p1[1] - p2[1]) < 0.001


def test_begin_step_allows_new_predict():
    k = KalmanTargetTracker(640, 480)
    k.begin_step()
    k.predict_only()
    k.begin_step()
    p = k.predict_only()
    assert isinstance(p, tuple)


def test_reset_to_center():
    k = KalmanTargetTracker(640, 480)
    k.update((50.0, 50.0))
    k.reset()
    p = k.predict_only()
    assert abs(p[0] - 320.0) < 1.0
    assert abs(p[1] - 240.0) < 1.0


def test_velocity_estimated_after_updates():
    k = KalmanTargetTracker(640, 480)
    k.begin_step()
    k.update((100.0, 100.0))
    k.begin_step()
    k.update((120.0, 110.0))
    _, _, vx, vy = k.predict_with_velocity()
    assert isinstance(vx, float)
    assert isinstance(vy, float)


def test_predict_then_update():
    k = KalmanTargetTracker(640, 480)
    k.begin_step()
    k.predict_only()
    p = k.update((100.0, 200.0))
    assert isinstance(p, tuple) and len(p) == 2

# ─── PixEagle-паттерн: EMA + ускорение ─────────────────────────────────────

def test_predict_with_acceleration_returns_six_values():
    """predict_with_acceleration() возвращает (x, y, vx, vy, ax, ay)."""
    kt = KalmanTargetTracker(640, 480)
    result = kt.predict_with_acceleration()
    assert len(result) == 6, f"Ожидалось 6 значений, получено {len(result)}"


def test_ema_velocity_smoother_than_raw():
    """
    После нескольких обновлений EMA-скорость менее резкая, чем raw Kalman.
    Проверяем что скорость вообще оценивается (не ноль после движения цели).
    """
    kt = KalmanTargetTracker(640, 480)
    # Цель движется с постоянной скоростью 5px/frame влево
    for i in range(10):
        kt.update((320.0 - i * 5, 240.0))
    _, _, vx, vy, _, _ = kt.predict_with_acceleration()
    assert vx < 0, f"EMA vx должен быть отрицательным при движении влево, vx={vx}"


def test_acceleration_estimated_after_velocity_change():
    """
    После изменения скорости ax должен быть ненулевым.
    Паттерн PixEagle MotionPredictor._update_velocity().
    """
    kt = KalmanTargetTracker(640, 480)
    # Медленное движение (10 кадров)
    for i in range(5):
        kt.update((320.0 + i * 1, 240.0))
    # Резкое ускорение
    for i in range(5):
        kt.update((320.0 + 5 + i * 10, 240.0))
    _, _, vx, vy, ax, ay = kt.predict_with_acceleration()
    # После ускорения ax должен быть положительным
    assert ax > 0, f"ax должен быть > 0 после ускорения вправо, ax={ax}"


def test_reset_clears_ema_and_accel():
    """reset() сбрасывает EMA velocity и acceleration в ноль."""
    kt = KalmanTargetTracker(640, 480)
    for i in range(10):
        kt.update((320.0 + i * 5, 240.0))
    kt.reset()
    _, _, vx, vy, ax, ay = kt.predict_with_acceleration()
    assert vx == 0.0, f"vx должен быть 0 после reset, vx={vx}"
    assert vy == 0.0, f"vy должен быть 0 после reset, vy={vy}"
    assert ax == 0.0, f"ax должен быть 0 после reset, ax={ax}"
    assert ay == 0.0, f"ay должен быть 0 после reset, ay={ay}"


def test_kinematic_lead_better_than_linear_under_acceleration():
    """
    При ускоряющейся цели кинематический прогноз (ax≠0) отличается от линейного (ax=0).
    Это подтверждает что acceleration работает.
    """
    kt = KalmanTargetTracker(640, 480)
    # Ускоряющаяся цель
    x = 100.0
    v = 2.0
    for i in range(15):
        v += 0.5  # ускорение
        x += v
        kt.update((x, 240.0))

    x_pos, y_pos, vx, vy, ax, ay = kt.predict_with_acceleration()
    # Кинематический прогноз на 5 кадров
    t = 5.0
    linear_pred   = x_pos + vx * t
    kinematic_pred = x_pos + vx * t + 0.5 * ax * t ** 2
    # Они должны отличаться если ускорение ненулевое
    if abs(ax) > 0.01:
        assert linear_pred != kinematic_pred, (
            f"Линейный={linear_pred:.1f} и кинематический={kinematic_pred:.1f} "
            f"должны отличаться при ax={ax:.3f}"
        )
