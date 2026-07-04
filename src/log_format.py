"""Colored console logging: timestamp | LEVEL | message."""

from __future__ import annotations

import logging
import os
import re
import sys

RESET = "\033[0m"
DIM = "\033[2m"

LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}

TIMESTAMP_COLOR = "\033[90m"
HTTP_STATUS_RE = re.compile(r'(" HTTP/1\.[01]" )(\d{3})( -)$')


def _use_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


def _status_color(code: int) -> str:
    if 200 <= code < 300:
        return "\033[32m"
    if 400 <= code < 500:
        return "\033[33m"
    if code >= 500:
        return "\033[31m"
    return ""


class ColoredFormatter(logging.Formatter):
    def __init__(self, *, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        timestamp = f"{timestamp},{int(record.msecs):03d}"
        level = record.levelname
        message = record.getMessage()

        if self.use_color:
            timestamp = f"{TIMESTAMP_COLOR}{timestamp}{RESET}"
            level_color = LEVEL_COLORS.get(level, "")
            level = f"{level_color}{level}{RESET}"
            message = self._colorize_message(message)

        line = f"{timestamp} | {level} | {message}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line

    def _colorize_message(self, message: str) -> str:
        match = HTTP_STATUS_RE.search(message)
        if not match:
            return message
        prefix, code_text, suffix = match.group(1), match.group(2), match.group(3)
        color = _status_color(int(code_text))
        if not color:
            return message
        start, end = match.span(2)
        return f"{message[:start]}{color}{code_text}{RESET}{message[end:]}"


def configure_logging(level: int = logging.INFO, stream=None) -> None:
    """Attach a colored formatter to the root logger."""
    stream = stream or sys.stdout
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(ColoredFormatter(use_color=_use_color(stream)))
    root.addHandler(handler)
    root.setLevel(level)
