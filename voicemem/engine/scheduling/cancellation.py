"""Opt-in cancellation at native admission boundaries, never inside a model."""
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
import asyncio
import threading


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


async def run_cancellable(operation, *, executor=None):
    """Offload disposable work and stop its pending native admission on cancel.

    Active native calls retain ownership until they return. Parent cancellation
    predicates remain effective inside the new worker scope.
    """
    abandoned = threading.Event()
    parent = _cancelled.get()

    def cancelled():
        return abandoned.is_set() or bool(parent is not None and parent())

    def work():
        with cancellation_scope(cancelled):
            check_cancelled()
            value = operation()
            check_cancelled()
            return value

    try:
        if executor is None:
            return await asyncio.to_thread(work)
        context = copy_context()
        return await asyncio.get_running_loop().run_in_executor(executor, context.run, work)
    except BaseException:
        abandoned.set()
        raise
