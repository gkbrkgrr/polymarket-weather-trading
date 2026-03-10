from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(
    level: str,
    *,
    log_file: str | Path | None = None,
    log_to_stdout: bool = True,
    log_rotate_max_mb: int = 128,
    log_rotate_backups: int = 20,
    log_http_requests: bool = False,
) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    configured_handlers: list[logging.Handler] = []

    if bool(log_to_stdout):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        configured_handlers.append(stream_handler)

    if log_file:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        max_bytes = max(1, int(log_rotate_max_mb)) * 1024 * 1024
        backup_count = max(1, int(log_rotate_backups))
        file_handler = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        configured_handlers.append(file_handler)

    if not configured_handlers:
        fallback_handler = logging.StreamHandler()
        fallback_handler.setFormatter(formatter)
        configured_handlers.append(fallback_handler)

    for handler in configured_handlers:
        root.addHandler(handler)

    if not bool(log_http_requests):
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
