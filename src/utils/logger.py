# src/utils/logger.py

import sys
from loguru import logger as _logger


def get_logger(name: str):
    _logger.remove()
    _logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
        level="INFO",
    )
    _logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        enqueue=True,
    )
    return _logger.bind(name=name)
