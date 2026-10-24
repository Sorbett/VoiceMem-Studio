"""Inference priority captured by tasks and explicitly contextualized workers."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from enum import IntEnum


class Service(IntEnum):
    """Scheduling classes; aging eventually admits lower-priority waiters."""

    AUDIO = 0
    INTERACTIVE = 1
    BACKGROUND = 2


_service = ContextVar("voicemem_inference_service", default=Service.INTERACTIVE)


@contextmanager
def inference_scope(service):
    """Propagate resource priority through explicitly captured worker contexts."""
    token = _service.set(Service(service))
    try:
        yield
    finally:
        _service.reset(token)
