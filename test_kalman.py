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