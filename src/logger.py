"""
src/logger.py
-------------
Centralised logging setup.  Import `get_logger(__name__)` in every module.
"""

import logging
import sys
from pathlib import Path

import config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(config.LOG_LEVEL)
    fmt = logging.Formatter(config.LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
