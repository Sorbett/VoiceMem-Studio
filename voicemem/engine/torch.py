"""Process-wide priority admission for shared Torch accelerator operations."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import threading
import time
import sys

from .context import Service, _service
from .devices import device_key


class _CaptureGate:
    """Allow independent CUDA devices except during process-wide graph capture."""

    def __init__(self):
        self.condition = threading.Condition()
        self.readers = {}
        self.owner = None
        self.waiting = 0

    def acquire(self, blocking=True, timeout=-1):
        ident = threading.get_ident()
        deadline = None if timeout == -1 else time.monotonic() + timeout
        with self.condition:
            while (self.owner not in (None, ident) or
                   (self.waiting and self.owner != ident and ident not in self.readers)):
                remaining = None if deadline is None else deadline-time.monotonic()
                if not blocking or (remaining is not None and remaining <= 0):
                    return False
                self.condition.wait(remaining)
            self.readers[ident] = self.readers.get(ident, 0) + 1
            return True

    def release(self):
        ident = threading.get_ident()
        with self.condition:
            self.readers[ident] -= 1
            if not self.readers[ident]:
                del self.readers[ident]
            self.condition.notify_all()

    @contextmanager
    def exclusive(self):
        ident = threading.get_ident()
        with self.condition:
            if ident in self.readers or self.owner == ident:
                raise RuntimeError("CUDA initialization must precede device admission")
            self.waiting += 1
            try:
                while self.owner is not None or self.readers:
                    self.condition.wait()
                self.owner = ident
            finally:
                self.waiting -= 1
                self.condition.notify_all()
        try:
            yield
        finally:
            with self.condition:
                self.owner = None
                self.condition.notify_all()


_capture_gate = _CaptureGate()


@dataclass(eq=False)
class _Waiter:
    service: Service
    arrived: float


class InferenceLock:
    """Reentrant, priority-aware native admission with FIFO starvation protection.

    An admitted native call is never preempted. Waiting callers may time out;
    expired tickets are removed without consuming a permit. Do not wait on a
    different worker while owning this lock.
    """

    def __init__(self, *, max_defer_s=0.5, device=None, _gate=None):
        self._condition = threading.Condition()
        self._owner = None
        self._depth = 0
        self._waiting = []
        self._model = None
        self._service = None
        self._started = 0.0
        self._memory = {}
        self._memory_sampled = 0.0
        self.max_defer_s = max_defer_s
        self.device = device
        self._gate = _gate

    def _next(self, now):
        overdue = [w for w in self._waiting if now - w.arrived >= self.max_defer_s]
        return (overdue[0] if overdue else
                min(self._waiting, key=lambda w: (w.service, w.arrived)))

    def acquire(self, blocking=True, timeout=-1):
        """Acquire with RLock-compatible blocking/timeout and reentrant ownership."""
        if not blocking and timeout != -1:
            raise ValueError("can't specify a timeout for a non-blocking call")
        if timeout < 0 and timeout != -1:
            raise ValueError("timeout value must be positive")
        ident = threading.get_ident()
        with self._condition:
            if self._owner == ident:
                self._depth += 1
                return True
        entered = time.monotonic()
        if self._gate is not None and not self._gate.acquire(blocking, timeout):
            return False
        try:
            return self._acquire(ident, blocking, timeout, entered)
        finally:
            with self._condition:
                admitted = self._owner == ident
            if self._gate is not None and not admitted:
                self._gate.release()

    def _acquire(self, ident, blocking, timeout, entered):
        """Own the condition and a CUDA capture read permit, when applicable."""
        with self._condition:
            waiter = _Waiter(_service.get(), time.monotonic())
            deadline = None if timeout == -1 else entered + timeout
            self._waiting.append(waiter)
            try:
                while True:
                    now = time.monotonic()
                    if self._owner is None and self._next(now) is waiter:
                        self._owner, self._depth = ident, 1
                        from .metrics import _model
                        self._model, self._service = _model.get(), waiter.service.name.lower()
                        self._started = now
                        return True
                    if not blocking or (deadline is not None and now >= deadline):
                        return False
                    self._condition.wait(None if deadline is None else max(0, deadline - now))
            finally:
                self._waiting.remove(waiter)
                self._condition.notify_all()

    def release(self):
        """Release one recursion level; only its owning thread may release it."""
        with self._condition:
            if self._owner != threading.get_ident():
                raise RuntimeError("cannot release un-acquired lock")
            self._depth -= 1
            if self._depth == 0:
                now = time.monotonic()
                if now-self._memory_sampled >= 1:
                    self._memory_sampled = now
                    torch = sys.modules.get("torch")
                    try:
                        if torch is not None and self.device and self.device.startswith("cuda:"):
                            cuda = torch.cuda
                            free, total = cuda.mem_get_info(self.device)
                            self._memory = {
                                "allocated_bytes": cuda.memory_allocated(self.device),
                                "reserved_bytes": cuda.memory_reserved(self.device),
                                "peak_allocated_bytes": cuda.max_memory_allocated(self.device),
                                "free_bytes": free, "total_bytes": total,
                            }
                            from .admission import resources
                            resources.device_pressure(self.device, free < max(256*1024**2, .05*total))
                        elif torch is not None and torch.backends.mps.is_available():
                            self._memory = {
                                "allocated_bytes": torch.mps.current_allocated_memory(),
                                "process_gpu_driver_bytes": torch.mps.driver_allocated_memory(),
                            }
                    except Exception:
                        self._memory = {"sampling_unavailable": True}
                self._owner = None
                self._model = self._service = None
                self._condition.notify_all()
                if self._gate is not None:
                    self._gate.release()

    def _is_owned(self):
        with self._condition:
            return self._owner == threading.get_ident()

    def snapshot(self):
        """Return metadata-only native admission counts without touching models."""
        with self._condition:
            return {**self._memory, "active": self._owner is not None,
                    "model": self._model, "service": self._service,
                    "active_ms": (time.monotonic()-self._started)*1000 if self._owner else 0,
                    "waiting": len(self._waiting),
                    "background_waiting": sum(w.service == Service.BACKGROUND for w in self._waiting)}

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()


TORCH_LOCK = InferenceLock()

_devices_lock = threading.Lock()
_devices = {}


def device_lock(device=None):
    """Share CUDA admission per visible device; retain the legacy MPS/default lock."""
    key = device_key(device)
    if key is None or not key.startswith("cuda:"):
        return TORCH_LOCK
    with _devices_lock:
        if key not in _devices:
            _devices[key] = InferenceLock(device=key, _gate=_capture_gate)
        return _devices[key]


def torch_snapshot():
    """Read cached device metadata without calling CUDA or initializing a runtime."""
    with _devices_lock:
        devices = dict(_devices)
    return {**TORCH_LOCK.snapshot(),
            "devices": {key: lock.snapshot() for key, lock in devices.items()}}


def cuda_initialization():
    """Exclude registered CUDA work while loading and capturing streaming graphs.

    Enter on the model's native owner before acquiring any CUDA device lock.
    CUDA graph capture cannot overlap unrelated CUDA execution on other threads.
    """
    return _capture_gate.exclusive()
