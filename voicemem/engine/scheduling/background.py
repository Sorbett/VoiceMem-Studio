"""Captured background admission and cancellation-safe native-work ownership."""
from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from contextvars import ContextVar

from .context import Service, inference_scope
from .admission import admission_scope
from .cancellation import cancellation_scope
from .reuse import reuse_scope


_wait = ContextVar("voicemem_background_wait", default=None)


@contextmanager
def background_scope(wait, *, cancelled=None):
    """Capture an idle waiter and optional thread-safe cancellation predicate.

    Cancellation skips waiting native calls only. Durable operations omit it so
    already accepted persistence retains its completion guarantee.
    """
    token = _wait.set(wait)
    try:
        with (inference_scope(Service.BACKGROUND), admission_scope(),
              cancellation_scope(cancelled), reuse_scope()):
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

        async def execute():
            nonlocal started
            async with self.lock:
                await wait_idle()
                if not is_current():
                    return None
                started = True
                return await operation()

        return await self._own(execute, lambda: started)

    async def _own(self, execute, has_started):
        """Retain started native work after cancellation of its async caller."""
        detached = False
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
            if not has_started():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            raise


class BackgroundPipeline(BackgroundLane):
    """Bound deferred jobs while preparing and settling them in separate stages.

    Preparation is FIFO and receives a thread event for all preceding completions.
    Deferred workers must wait on that event before ordered side effects. settle
    runs even if preparation fails and must await any native work it started.
    """

    def __init__(self, max_active=2):
        if not isinstance(max_active, int) or max_active < 1:
            raise ValueError("max_active must be a positive integer")
        super().__init__()
        self._capacity = asyncio.Semaphore(max_active)
        self._tail = None

    async def run(self, wait_idle, prepare, settle):
        """Overlap bounded preparation with ordered completion; retain native ownership."""
        started = False

        async def execute():
            nonlocal started
            async with self._capacity:
                completion = previous = None
                try:
                    async with self.lock:
                        await wait_idle()
                        started = True
                        previous = self._tail
                        event = threading.Event()
                        future = asyncio.get_running_loop().create_future()
                        completion = (event, future)
                        self._tail = completion
                        predecessor = previous[0] if previous else threading.Event()
                        if previous is None:
                            predecessor.set()
                        result = await prepare(predecessor)
                finally:
                    if completion is not None:
                        try:
                            await settle()
                        finally:
                            # A failed preparation cannot let later writes overtake
                            # an earlier native job that is still running.
                            if previous is not None:
                                await asyncio.shield(previous[1])
                            completion[0].set()
                            completion[1].set_result(None)
                            if self._tail is completion:
                                self._tail = None
                return result

        return await self._own(execute, lambda: started)
