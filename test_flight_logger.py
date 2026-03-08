"""
test_flight_logger.py — тесты FlightLogger.

Покрывает:
  - Создание файла и запись заголовка
  - Запись строк и их структура (22 поля)
  - Byte-counter вместо os.path.getsize (нет syscall при каждой записи)
  - Ротация при достижении MAX_LOG_SIZE
  - Graceful close и повторный close
  - Защита от write() при закрытом handle
"""

import os
import csv
import pytest
import tempfile
from unittest.mock import patch

from flight_logger import FlightLogger
from config import MAX_LOG_SIZE


@pytest.fixture
def tmp_logger(tmp_path):
    """FlightLogger с временной директорией логов."""
    with patch('flight_logger.LOG_DIR', str(tmp_path)):
        fl = FlightLogger()
        yield fl, tmp_path
        fl.close()


# ─── 1. Создание и заголовок ─────────────────────────────────────────────────

def test_file_created(tmp_logger):
    fl, tmp_path = tmp_logger
    files = list(tmp_path.glob('*.csv'))
    assert len(files) == 1


def test_header_written(tmp_logger):
    fl, tmp_path = tmp_logger
    fl.close()
    csv_file = next(tmp_path.glob('*.csv'))
    with open(csv_file) as f:
        header = f.readline()
    assert 'timestamp' in header
    assert 'target_x' in header
    assert 'lead_y' in header


def test_header_has_22_columns(tmp_logger):
    fl, tmp_path = tmp_logger
    fl.close()
    csv_file = next(tmp_path.glob('*.csv'))
    with open(csv_file) as f:
        cols = f.readline().strip().split(',')
    assert len(cols) == 22, f"Ожидалось 22 колонки, получено {len(cols)}: {cols}"


# ─── 2. Запись строк ─────────────────────────────────────────────────────────

def _sample_row():
    return dict(
        ts=1700000000.0, tx=320.0, ty=240.0, conf=0.95,
        mode='TRACKING', ramp=0.5,
        err_x=10.0, err_y=-5.0,
        rc_r=65535, rc_p=1400, rc_t=1550, rc_y=1600,
        alt=25.0, batt=15.8, armed=True, mode_str='TRACKING',
        safety='OK', climb=0.3, temp=52.0, fps=30.0,
        lead_x=330.0, lead_y=235.0,
    )


def test_write_produces_csv_row(tmp_logger):
    fl, tmp_path = tmp_logger
    fl.write(**_sample_row())
    fl.close()
    csv_file = next(tmp_path.glob('*.csv'))
    with open(csv_file) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2   # header + 1 data row
    assert len(rows[1]) == 22


def test_write_values_correct(tmp_logger):
    fl, tmp_path = tmp_logger
    fl.write(**_sample_row())
    fl.close()
    csv_file = next(tmp_path.glob('*.csv'))
    with open(csv_file) as f:
        rows = list(csv.reader(f))
    data = rows[1]
    assert float(data[0]) == pytest.approx(1700000000.0, abs=0.001)
    assert float(data[1]) == pytest.approx(320.0)
    assert float(data[3]) == pytest.approx(0.95)
    assert data[4] == 'TRACKING'


def test_write_multiple_rows(tmp_logger):
    fl, tmp_path = tmp_logger
    for i in range(10):
        fl.write(**_sample_row())
    fl.close()
    csv_file = next(tmp_path.glob('*.csv'))
    with open(csv_file) as f:
        rows = list(f.readlines())
    assert len(rows) == 11   # header + 10 data


# ─── 3. Byte-counter (нет os.path.getsize) ───────────────────────────────────

def test_no_getsize_calls(tmp_logger):
    """os.path.getsize НЕ должен вызываться при обычной записи."""
    fl, tmp_path = tmp_logger
    with patch('os.path.getsize') as mock_gs:
        for _ in range(10):
            fl.write(**_sample_row())
    assert mock_gs.call_count == 0, (
        f"os.path.getsize вызван {mock_gs.call_count} раз — должно быть 0"
    )


def test_written_counter_tracks_bytes(tmp_logger):
    """_written счётчик нарастает после каждой строки."""
    fl, _ = tmp_logger
    before = fl._written
    fl.write(**_sample_row())
    after = fl._written
    assert after > before


def test_written_initialized_with_header_size(tmp_logger):
    fl, _ = tmp_logger
    header_bytes = len(FlightLogger.HEADER.encode())
    assert fl._written == header_bytes


# ─── 4. Ротация файла ────────────────────────────────────────────────────────

def test_rotation_on_size_limit(tmp_logger):
    """Ротация: при превышении MAX_LOG_SIZE _written сбрасывается (новый файл)."""
    fl, tmp_path = tmp_logger
    import flight_logger as fl_mod
    small_limit   = 2048
    orig_limit    = fl_mod.MAX_LOG_SIZE
    row           = _sample_row()
    try:
        fl_mod.MAX_LOG_SIZE = small_limit
        fl._written = 0   # начинаем с нуля
        for _ in range(20):   # ~20*150 = 3000 байт > 2048 → должна произойти ротация
            fl.write(**row)
    finally:
        fl_mod.MAX_LOG_SIZE = orig_limit

    # После ротации _written сбрасывается до len(HEADER).
    # Без ротации _written ≈ 3000 >> small_limit.
    header_size = len(FlightLogger.HEADER.encode())
    assert fl._written <= small_limit, (
        f"Ротация не произошла: _written={fl._written} > small_limit={small_limit}. "
        f"Ожидалось ≤ {small_limit} (header={header_size} + часть строк после ротации)."
    )


# ─── 5. Graceful handling ────────────────────────────────────────────────────

def test_write_after_close_is_safe(tmp_logger):
    """write() после close() не бросает исключений."""
    fl, _ = tmp_logger
    fl.close()
    fl.write(**_sample_row())   # не должно падать


def test_double_close_safe(tmp_logger):
    """Двойной close() безопасен."""
    fl, _ = tmp_logger
    fl.close()
    fl.close()   # не должно бросать


def test_flush_on_50_rows(tmp_logger):
    """Каждые 50 строк вызывается flush (сброс буфера)."""
    fl, _ = tmp_logger
    with patch.object(fl.handle, 'flush', wraps=fl.handle.flush) as mock_flush:
        for _ in range(55):
            fl.write(**_sample_row())
    # flush() должен быть вызван хотя бы 1 раз за 55 строк
    assert mock_flush.call_count >= 1
