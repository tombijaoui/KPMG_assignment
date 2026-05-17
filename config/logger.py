from __future__ import annotations

import logging
import sys


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure a logger with format: module:function - LEVEL - message.

    Args:
        name: Logger name (__name__ of the calling module)
        level: Logging level

    Returns:
        Configured logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s:%(funcName)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_logger(module_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger for the given module. Always pass __name__ from the caller module.

    Args:
        module_name: Module name (required, typically __name__).
        level: Logging level used when the logger is first created.

    Returns:
        Logger instance.

    Raises:
        ValueError: If module_name is empty.
    """
    
    if not module_name:
        raise ValueError("module_name is required (pass __name__)")

    logger = logging.getLogger(module_name)

    if not logger.handlers:
        setup_logger(module_name, level=level)

    return logger
