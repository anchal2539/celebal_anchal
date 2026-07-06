"""
helper.py
---------
Small utility functions shared across the app (timing, formatting, stats).
"""

import time
from contextlib import contextmanager


@contextmanager
def timer():
    """
    Usage:
        with timer() as t:
            do_something()
        print(t.elapsed)  # seconds, available after the block exits
    """
    class _Timer:
        elapsed = None

    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start


def document_stats(text: str, chunks: list[str]) -> dict:
    """Basic stats about the processed document, for display in the UI."""
    return {
        "characters": len(text),
        "words": len(text.split()),
        "chunks": len(chunks),
    }


def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"
