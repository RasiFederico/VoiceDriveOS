"""Structured, high-precision logging for VoiceDriveOS.

Every log line includes a millisecond-precision timestamp, the logger name,
level, and message, matching the format specified in the project brief::

    [12:32:01.201] STATE=LISTENING
    [12:32:02.114] speech_end

Call :func:`configure_logging` exactly once at application startup (this is
idempotent: calling it again just reconfigures the root VoiceDriveOS logger).
Modules should then call :func:`get_logger(__name__)` to obtain a child
logger.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT_LOGGER_NAME = "voicedriveos"
_CONFIGURED = False


class _MillisecondFormatter(logging.Formatter):
    """Formatter that renders timestamps as ``HH:MM:SS.mmm``."""

    default_msec_format = "%s.%03d"

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        ct = self.converter(record.created)
        base = super().formatTime(record, datefmt="%H:%M:%S")
        return f"{base}.{int(record.msecs):03d}"


def configure_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    to_console: bool = True,
) -> logging.Logger:
    """Configure the VoiceDriveOS root logger.

    Args:
        level: Logging level (e.g. ``logging.DEBUG`` or ``"DEBUG"``).
        log_file: Optional path to also write logs to a file.
        to_console: Whether to attach a stream handler on stdout.

    Returns:
        The configured root VoiceDriveOS logger.
    """
    global _CONFIGURED

    root_logger = logging.getLogger(_ROOT_LOGGER_NAME)
    root_logger.setLevel(level)
    root_logger.propagate = False

    # Avoid duplicate handlers if configure_logging() is called more than once.
    root_logger.handlers.clear()

    formatter = _MillisecondFormatter(fmt="[%(asctime)s] %(name)s: %(message)s")

    if to_console:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if not to_console and log_file is None:
        root_logger.addHandler(logging.NullHandler())

    _CONFIGURED = True
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the VoiceDriveOS root logger.

    If :func:`configure_logging` has not been called yet, a sane default
    (INFO level, console output) is configured automatically so that modules
    can be imported and log independently (e.g. in unit tests).
    """
    if not _CONFIGURED:
        configure_logging()

    if name.startswith(_ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
