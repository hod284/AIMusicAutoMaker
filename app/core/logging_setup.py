"""로깅 설정. 비밀값(API 키, Bearer 토큰)은 로그에서 마스킹한다."""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_SECRET_PATTERNS = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"(\"?ai_token\"?\s*[:=]\s*\"?)[^\",\s]+", re.IGNORECASE),
    re.compile(r"(api[_-]?key\"?\s*[:=]\s*\"?)[^\",\s]+", re.IGNORECASE),
]


def redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(r"\1***", text)
    return text


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = RedactingFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, "_studio", False):
            root.removeHandler(handler)
    file_handler = RotatingFileHandler(log_dir / "studio.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler._studio = True  # type: ignore[attr-defined]
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console._studio = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)
    root.addHandler(console)
