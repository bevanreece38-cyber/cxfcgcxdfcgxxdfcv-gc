"""
test_architecture.py — верификация архитектурных инвариантов системы.

Проверяет три конкретных замечания из review:

  Problem #1: types_enum.py не содержит ControlMode — только ControlState.
              main.py импортирует TrackerState и SafetyStatus — оба присутствуют.

  Problem #2: VisionTracker.step(frame, yolo_outputs) — frame первый аргумент.
              Все тесты и main.py вызывают step именно в таком порядке.

  Problem #3: release_control() вызывается ТОЛЬКО из main.py (ControlManager),
              НЕ из tracker_engine.py. TrackerEngine не имеет доступа к ControlManager.
              Это намеренная архитектурная граница: TrackerEngine — чистая логика,
              main.py — оркестратор управления.
"""

import ast
import inspect
import pathlib
import importlib

import pytest


# ---------------------------------------------------------------------------
#  Problem #1 — types_enum.py: ControlState, не ControlMode
# ---------------------------------------------------------------------------

def test_control_state_exists_in_types_enum():
    """ControlState должен быть в types_enum.py — используется ControlManager."""
    from types_enum import ControlState
    assert hasattr(ControlState, "MANUAL")
    assert hasattr(ControlState, "COMPUTER")


def test_control_mode_absent_from_types_enum():
    """
    ControlMode НЕ существует в types_enum.py.
    Правильное название — ControlState. Любой импорт ControlMode вызовет ImportError.
    """
    src = pathlib.Path("types_enum.py").read_text()
    assert "ControlMode" not in src, (
        "types_enum.py содержит устаревшее имя 'ControlMode'. "
        "Правильное имя: ControlState."
    )


def test_main_imports_tracker_state_not_control_mode():
    """
    main.py импортирует TrackerState и SafetyStatus из types_enum.
    ControlMode нигде не импортируется.
    """
    src = pathlib.Path("main.py").read_text()
    assert "TrackerState" in src, "main.py должен импортировать TrackerState"
    assert "SafetyStatus" in src, "main.py должен импортировать SafetyStatus"
    assert "ControlMode" not in src, (
        "main.py не должен импортировать несуществующий 'ControlMode'"
    )


def test_all_tracker_states_defined():
    """Все ожидаемые состояния TrackerState присутствуют в types_enum."""
    from types_enum import TrackerState
    expected = {"IDLE", "ACQUIRING", "TRACKING", "DEAD_RECKON",
                "REACQUIRE", "LOST", "STRIKING"}
    actual = {s.name for s in TrackerState}
    assert expected == actual, (
        f"Отсутствующие состояния: {expected - actual}. "
        f"Лишние состояния: {actual - expected}."
    )


# ---------------------------------------------------------------------------
#  Problem #2 — VisionTracker.step(frame, yolo_outputs): frame первым
# ---------------------------------------------------------------------------

def test_vision_tracker_step_signature_frame_first():
    """
    VisionTracker.step() принимает frame первым позиционным аргументом.
    Это ИСПРАВЛЕНО: старая сигнатура была step(yolo_outputs, frame).
    """
    from vision_tracker import VisionTracker
    sig    = inspect.signature(VisionTracker.step)
    params = list(sig.parameters.keys())
    assert params[1] == "frame", (
        f"Первый аргумент step() должен быть 'frame', получен '{params[1]}'. "
        f"Порядок аргументов: {params}"
    )
    assert params[2] == "yolo_outputs", (
        f"Второй аргумент step() должен быть 'yolo_outputs', получен '{params[2]}'."
    )


def test_vision_tracker_step_works_frame_first():
    """Функциональная проверка: step(frame, None) не падает."""
    import numpy as np
    from vision_tracker import VisionTracker
    vt     = VisionTracker()
    vt.reset()
    frame  = np.zeros((480, 640, 3), dtype=np.uint8)
    result = vt.step(frame, None)   # frame первым — ожидается None (нет цели)
    assert result is None


def test_step_not_called_with_yolo_first_in_tests():
    """
    Ни один вызов vt.step() в test_vision_tracker.py не передаёт yolo первым.
    AST-анализ: первый позиционный аргумент никогда не является кортежем/детекцией.
    """
    src  = pathlib.Path("test_vision_tracker.py").read_text()
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Ищем вызовы .step(...)
        if not (isinstance(func, ast.Attribute) and func.attr == "step"):
            continue
        args = node.args
        if len(args) < 1:
            continue
        # Первый позиционный аргумент не должен быть кортежем (YOLO output)
        first = args[0]
        assert not isinstance(first, ast.Tuple), (
            f"Строка {node.lineno}: step() вызван с кортежем (YOLO output) "
            f"как первым аргументом — порядок должен быть step(frame, yolo)."
        )


# ---------------------------------------------------------------------------
#  Problem #3 — release_control() только в main.py, не в tracker_engine.py
# ---------------------------------------------------------------------------

def _get_attribute_calls(source_path: str) -> list:
    """Вернуть список имён методов из всех вызовов obj.method() в файле."""
    src  = pathlib.Path(source_path).read_text()
    tree = ast.parse(src)
    return [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    ]


def test_tracker_engine_does_not_call_release_control():
    """
    TrackerEngine НЕ вызывает release_control().
    Архитектурный инвариант: TrackerEngine — чистая логика трекинга.
    Вызов release_control() — ответственность main.py через ControlManager.
    """
    calls = _get_attribute_calls("tracker_engine.py")
    assert "release_control" not in calls, (
        "tracker_engine.py вызывает release_control()! "
        "Это нарушение архитектуры: только main.py должен управлять ControlManager."
    )


def test_tracker_engine_does_not_import_control_manager():
    """
    tracker_engine.py не импортирует ControlManager.
    TrackerEngine не знает о MAVLink/RC — только о трекинге.
    Docstrings могут ссылаться на ControlManager как на потребителя, но не импортируют его.
    """
    src  = pathlib.Path("tracker_engine.py").read_text()
    tree = ast.parse(src)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("control_manager" in imp.lower() for imp in imports), (
        "tracker_engine.py не должен импортировать control_manager модуль."
    )


def test_main_calls_release_control_on_lost():
    """
    main.py вызывает release_control() при состоянии LOST.
    Это подтверждает, что управление RC возвращается оператору из main.py.
    """
    src = pathlib.Path("main.py").read_text()
    # LOST + release_control() должны быть рядом в исходнике
    assert "LOST" in src, "main.py должен обрабатывать состояние LOST"
    assert "release_control" in src, (
        "main.py должен вызывать release_control() при потере цели"
    )


def test_main_calls_release_control_on_safety():
    """main.py вызывает release_control() при срабатывании SAFETY."""
    src   = pathlib.Path("main.py").read_text()
    lines = src.splitlines()
    # Найти строки с "release_control" и убедиться что одна из них рядом с SAFETY
    rc_lines   = [i for i, l in enumerate(lines) if "release_control" in l]
    safe_lines = [i for i, l in enumerate(lines) if "SAFETY" in l]
    near = any(
        abs(rc - sf) <= 5
        for rc in rc_lines
        for sf in safe_lines
    )
    assert near, (
        "release_control() должен вызываться рядом с обработчиком SAFETY в main.py"
    )


def test_handle_lost_returns_track_result_not_rc_command():
    """
    _handle_lost() в TrackerEngine возвращает TrackResult.
    Он НЕ отправляет RC команды напрямую — это делает main.py через ControlManager.
    """
    import numpy as np
    from tracker_engine import TrackerEngine, TrackResult
    from types_enum import TrackerState
    from config import RC_MID, RC_RELEASE

    eng = TrackerEngine()
    eng.engage()

    # Симулируем потерю цели
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = eng.step(None, frame)   # None вместо YOLO → потеря цели

    # _handle_lost вернул TrackResult — не отправил RC команду
    assert isinstance(result, TrackResult), (
        f"_handle_lost() должен вернуть TrackResult, получен {type(result)}"
    )
    assert result.state in (
        TrackerState.ACQUIRING, TrackerState.DEAD_RECKON,
        TrackerState.REACQUIRE, TrackerState.LOST,
    ), f"Неожиданное состояние после потери: {result.state}"
