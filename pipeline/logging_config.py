"""
Shared logging setup for the pipeline package.

Centralizing this so every phase module logs through the same format/handler
instead of sprinkling print() everywhere -- makes it trivial to redirect to
a file or bump verbosity (e.g. for the Phase 5 observability/drift story,
or just to see what's happening behind the scenes during a run).
"""

from __future__ import annotations

import logging

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once. Safe to call multiple times.

    `level` controls verbosity of our own `pipeline.*` loggers only. Root
    stays at INFO so chatty third-party libraries (httpx, urllib3, huggingface
    hub transfer, etc.) don't flood DEBUG output when --verbose is used.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True

    logging.getLogger("pipeline").setLevel(level)
    logging.getLogger(__package__ or "pipeline").setLevel(level)
    # Keep noisy HTTP/transfer libraries quiet regardless of our verbosity.
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
