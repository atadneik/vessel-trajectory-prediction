"""Thiết lập logging tập trung cho dự án dự đoán quỹ đạo tàu."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "vessel",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Tạo và trả về một logger đã được cấu hình.

    Parameters
    ----------
    name : str
        Tên logger (thường là ``__name__`` của module).
    log_dir : str, optional
        Thư mục chứa file log. Được tạo nếu chưa tồn tại.
    level : int
        Mức logging (mặc định ``logging.INFO``).
    log_file : str, optional
        Tên file log tường minh. Khi là *None* và *log_dir* được cung cấp,
        file mặc định là ``<log_dir>/<name>.log``.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler cho console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Handler cho file (mode='w' → ghi đè mỗi lần chạy)
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fname = log_file if log_file else f"{name}.log"
        fh = logging.FileHandler(log_path / fname, mode="w")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
