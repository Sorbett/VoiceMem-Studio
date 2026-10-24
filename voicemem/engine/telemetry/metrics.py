"""Metadata-only model execution, admission and lock-wait measurements."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import threading
import time

from ..scheduling.admission import resources
from ..scheduling.context import Service, _service, inference_scope
from ..backends.devices import device_key


_model = ContextVar("voicemem_inference_model", default="unlabelled")
_mutex = threading.Lock()
_metrics = {}


def record(model, backend, run_s, *, lock_s=0.0, admission_s=0.0, queue_s=0.0, forced=False, error=False,
           device=None):
    """Aggregate static adapter labels only; never pass prompts or identifiers."""
    key = (model, backend, _service.get().name.lower(), device_key(device))
    with _mutex:
        if key not in _metrics and len(_metrics) >= 128:
            return
        row = _metrics.setdefault(key, dict(calls=0, run_ms=0., max_run_ms=0.,
                                           lock_ms=0., admission_ms=0., queue_ms=0., forced=0, errors=0))
        row["calls"] += 1
        row["run_ms"] += run_s * 1000
        row["max_run_ms"] = max(row["max_run_ms"], run_s * 1000)
        row["lock_ms"] += lock_s * 1000
        row["admission_ms"] += admission_s * 1000
        row["queue_ms"] += queue_s * 1000
        row["forced"] += bool(forced)
        row["errors"] += bool(error)
    if _service.get() == Service.BACKGROUND:
        resources.observe(model, run_s, device=device)


def model_metrics(*, reset=False):
    """Return bounded process aggregates, optionally draining the current window."""
    with _mutex:
        rows = [dict(model=m, backend=b, service=s, device=d, **v)
                for (m, b, s, d), v in _metrics.items()]
        if reset:
            _metrics.clear()
        return rows


@contextmanager
def inference_call(model, backend, *, lock=None, service=None, device=None, queue_s=0.):
    """Time a native call, waiting for admission before acquiring its model lock.

    Call from existing native workers only. Device-asynchronous adapters must
    include their existing materialization boundary inside this scope. A tuple
    of locks acquires model ownership before device admission; later entries
    may be zero-argument factories resolving mutable placement under ownership.
    """
    with inference_scope(_service.get() if service is None else service):
        locks = lock if isinstance(lock, tuple) else (() if lock is None else (lock,))
        device = device_key(device if device is not None else getattr(lock, "device", None))
        owned = any(getattr(item, "_is_owned", lambda: False)() for item in locks)
        admission, forced = ((resources.wait(model, device=device) if device else resources.wait(model))
                             if _service.get() == Service.BACKGROUND and not owned
                             else (0., False))
        token = _model.set(model)
        acquired = []
        waiting = time.perf_counter()
        try:
            for item in locks:
                native_lock = item() if not hasattr(item, "acquire") else item
                native_lock.acquire()
                acquired.append(native_lock)
            if device is None:
                device = next((getattr(item, "device", None) for item in reversed(acquired)
                               if getattr(item, "device", None) is not None), None)
            started = time.perf_counter()
            failed = False
            try:
                yield
            except BaseException:
                failed = True
                raise
            finally:
                elapsed = time.perf_counter() - started
                record(model, backend, elapsed, lock_s=started-waiting,
                       admission_s=admission, forced=forced, error=failed, device=device, queue_s=queue_s)
        finally:
            for native_lock in reversed(acquired):
                native_lock.release()
            _model.reset(token)
