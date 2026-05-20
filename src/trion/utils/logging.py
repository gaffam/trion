from __future__ import annotations

from loguru import logger
import sys


def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, format="<green>{time}</green> | <level>{level}</level> | {message}")
