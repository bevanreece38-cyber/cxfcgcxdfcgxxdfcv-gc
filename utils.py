import logging
import os
from datetime import datetime
from config import LOG_DIR


def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Создаёт логгер с выводом в консоль (INFO) и в файл (DEBUG).
    Повторный вызов с тем же именем возвращает уже настроенный логгер.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    # Консоль
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Файл
    log_file = os.path.join(
        LOG_DIR,
        f"flight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger