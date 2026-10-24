"""Captured background admission and cancellation-safe native-work ownership."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar

from .context import Service, inference_scope
from .admission import admission_scope


_wait = ContextVar("voicemem_background_wait", default=None)


@contextmanager
def background_scope(wait):
    """Capture a synchronous idle waiter for this operation and its worker context."""
    token = _wait.set(wait)
    try:
        with inference_scope(Service.BACKGROUND), admission_scope():
            yield
    finally:
        _wait.reset(token)


def background_checkpoint():
    """Yield to host foreground work before a stage acquires any resource locks."""
    wait = _wait.get()
    if wait is not None:
        wait()


class BackgroundLane:
    """Own one active job, including native work whose caller was cancelled."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.tasks = set()

    async def run(self, wait_idle, operation, *, is_current=lambda: True):
        """Wait for idle admission; cancel queued jobs but drain started native work."""
        started = False
        detached = False

        async def execute():
            nonlocal started
            async with self.lock:
                await wait_idle()
                if not is_current():
                    return None
                started = True
                return await operation()

        task = asyncio.create_task(execute())
        self.tasks.add(task)

        def done(finished):
            self.tasks.discard(finished)
            if not finished.cancelled():
                error = finished.exception()
                if error is not None and detached:
                    print(f"[background] {type(error).__name__}: {error}", flush=True)

        task.add_done_callback(done)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            detached = True
            if not started:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            raise
