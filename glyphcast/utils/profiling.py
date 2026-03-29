"""Timing utilities."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def timed_step(metrics: dict[str, float], name: str) -> Iterator[None]:
    start = perf_counter()
    yield
    metrics[name] = perf_counter() - start
