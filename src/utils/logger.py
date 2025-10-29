"""
Provides a simple, project-wide logger. Tries to use loguru; falls back to
the standard library `logging` if loguru cannot be imported (e.g., on
restricted or unusual Windows environments).

This project prefers loguru for its concise, higher-level API for sinks (add/remove), 
built-in rotation/retention, pretty/structured formatting, and easy runtime reconfiguration.
"""

from pathlib import Path
import logging

# --- Paths ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = LOG_DIR / "project.log"


def _configure_stdlib_logger():
       """Return a stdlib logger configured to approximate the previous behavior."""
       logger = logging.getLogger("project")
       logger.setLevel(logging.DEBUG)
       # File handler
       fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
       fh.setLevel(logging.DEBUG)
       fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s")
       fh.setFormatter(fmt)
       # Console handler
       ch = logging.StreamHandler()
       ch.setLevel(logging.INFO)
       ch.setFormatter(fmt)
       # Avoid duplicate handlers
       if not logger.handlers:
              logger.addHandler(fh)
              logger.addHandler(ch)
       return logger


# Try to use loguru, but fall back to stdlib logging if import fails
try:
       from loguru import logger

       # Configure loguru
       logger.remove()
       logger.add(LOG_FILE, rotation="10 MB", retention="7 days", level="INFO",
                        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                                     "<level>{level: <8}</level> | "
                                     "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                                     "<level>{message}</level>")
       logger.add(lambda msg: print(msg, end=""), level="INFO")
except Exception:
       # Fall back to standard logging
       logger = _configure_stdlib_logger()


def close_logger():
       """Close/flush any active logger handlers so log files can be safely
       deleted (Windows locks files otherwise). This is safe to call regardless
       of whether loguru or stdlib logger is active."""
       try:
              # Attempt to close loguru handlers if available
              from loguru import logger as _l
              try:
                     _l.remove()
              except Exception:
                     pass
       except Exception:
              # stdlib logger
              try:
                     root = logging.getLogger("project")
                     for h in list(root.handlers):
                            try:
                                   h.flush()
                            except Exception:
                                   pass
                            try:
                                   h.close()
                            except Exception:
                                   pass
                     root.handlers = []
              except Exception:
                     pass


def reopen_logger():
       """Re-initialize the module-level `logger` after it has been closed.
       This mirrors the module's initial behaviour (prefer loguru, fall back to
       stdlib)."""
       global logger
       try:
              from loguru import logger as _l
              # configure loguru similar to the module init
              try:
                     _l.remove()
              except Exception:
                     pass
              _l.add(LOG_FILE, rotation="10 MB", retention="7 days", level="INFO",
                             format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                                          "<level>{level: <8}</level> | "
                                          "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                                          "<level>{message}</level>")
              _l.add(lambda msg: print(msg, end=""), level="INFO")
              logger = _l
       except Exception:
              logger = _configure_stdlib_logger()
