"""Opt-in cancellation at native admission boundaries, never inside a model."""
from contextlib import contextmanager
from contextvars import ContextVar


_cancelled = ContextVar("voicemem_native_cancelled", default=None)


class InferenceCancelled(Exception):
    """A disposable operation became obsolete before native execution."""


@contextmanager
def cancellation_scope(cancelled):
    """Capture a thread-safe predicate; nested scopes inherit it when omitted."""
    token = _cancelled.set(cancelled if cancelled is not None else _cancelled.get())
    try:
        yield
    finally:
        _cancelled.reset(token)


def check_cancelled():
    predicate = _cancelled.get()
    if predicate is not None and predicate():
        raise InferenceCancelled()


def cancellable_wait():
    """Poll only opt-in callers so an event can interrupt native queue waits."""
    return .05 if _cancelled.get() is not None else None
