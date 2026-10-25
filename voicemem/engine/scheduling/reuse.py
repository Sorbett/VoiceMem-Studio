"""Bounded reuse of immutable inference results within one captured operation."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, TimeoutError
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import threading
import time

from .cancellation import check_cancelled, cancellable_wait


_current = ContextVar("voicemem_inference_reuse", default=None)
_disabled = object()


@dataclass(frozen=True)
class ReusableResult:
    """Adapter-owned immutable CPU payload and its retained size in bytes."""

    payload: object
    size_bytes: int


class InferenceReuse:
    """Keep exact-request results and coalesce concurrent misses within one job.

    Providers opt in only for deterministic inference with immutable model state.
    Keys must include model ownership and every result-affecting input. freeze
    returns an immutable CPU payload within the supplied byte limit, or None to
    decline reuse. thaw must return an independent result for each consumer.
    No provider operation, copy or wait runs under the metadata lock.
    """

    def __init__(self, *, max_entries=64, max_bytes=2 * 1024 * 1024):
        if type(max_entries) is not int or type(max_bytes) is not int or max_entries < 1 or max_bytes < 1:
            raise ValueError("Inference reuse limits must be positive")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._lock = threading.Lock()
        self._ready = OrderedDict()
        self._pending = {}
        self._bytes = 0

    def run(self, key, operation, *, freeze, thaw):
        """Return (result, reused); propagate failures without retaining them.

        Call outside provider and device locks. Cancellation of a waiter never
        releases another thread's native ownership. Recursive or excess misses
        bypass coalescing rather than waiting on themselves or growing the pool.
        """
        check_cancelled()
        ident = threading.get_ident()
        with self._lock:
            cached = self._ready.get(key)
            if cached is not None:
                self._ready.move_to_end(key)
            pending = self._pending.get(key)
            owner = cached is None and pending is None and len(self._pending) < self.max_entries
            if owner:
                pending = (ident, Future())
                self._pending[key] = pending
        if cached is not None:
            return thaw(cached.payload), True
        if not owner:
            if pending is None or pending[0] == ident:
                return operation(), False
            while True:
                check_cancelled()
                try:
                    cached = pending[1].result(timeout=cancellable_wait())
                    break
                except TimeoutError:
                    if pending[1].done():
                        cached = pending[1].result()
                        break
            check_cancelled()
            return (thaw(cached.payload), True) if cached is not None else (operation(), False)
        try:
            value = operation()
            cached = freeze(value, self.max_bytes)
            if cached is not None:
                if not isinstance(cached, ReusableResult) or not 0 <= cached.size_bytes <= self.max_bytes:
                    raise ValueError("Invalid reusable inference payload")
                with self._lock:
                    while self._ready and (len(self._ready) >= self.max_entries
                                           or self._bytes + cached.size_bytes > self.max_bytes):
                        _, expired = self._ready.popitem(last=False)
                        self._bytes -= expired.size_bytes
                    self._ready[key] = cached
                    self._bytes += cached.size_bytes
            pending[1].set_result(cached)
            return value, False
        except BaseException as exc:
            pending[1].set_exception(exc)
            raise
        finally:
            with self._lock:
                del self._pending[key]

    def snapshot(self):
        """Return bounded metadata without keys or retained inference payloads."""
        with self._lock:
            return {"entries": len(self._ready), "pending": len(self._pending), "bytes": self._bytes}


@contextmanager
def reuse_scope(*, enabled=True):
    """Capture a bounded operation-local pool; an explicit opt-out also covers children."""
    current = _current.get()
    token = _current.set((current or InferenceReuse()) if enabled else _disabled)
    try:
        yield
    finally:
        _current.reset(token)


def reuse_call(key, operation, *, freeze, thaw, label=None):
    """Reuse a pure call outside device locks, using a lazy exact-request key.

    None declines participation. An optional static label records reused calls
    as cache work, separately from native execution and admission estimates.
    """
    pool = _current.get()
    if key is None or pool is None or pool is _disabled:
        return operation(), False
    from ..backends.torch import owns_device_lock
    if owns_device_lock():
        return operation(), False
    started = time.perf_counter()
    key = key()
    if key is None:
        return operation(), False
    result, reused = pool.run(key, operation, freeze=freeze, thaw=thaw)
    if reused and label is not None:
        from ..telemetry.metrics import record
        record(label, "cache", time.perf_counter()-started)
    return result, reused
