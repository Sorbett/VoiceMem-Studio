"""Process-wide capacity for background native work across provider locks."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import threading
import time

from ..backends.devices import device_key


@dataclass(frozen=True)
class CapacityLease:
    """Queue time and nesting state for one thread-owned native reservation."""

    wait_s: float = 0.0
    reentrant: bool = False


def _domain(device):
    key = device_key(device)
    if key is None:
        return None
    return key if key.startswith("cuda:") else "host"


def _conflicts(left, right):
    return left is None or right is None or left == right


class BackgroundCapacity:
    """Serialize competing background stages before speech and model admission.

    CPU and MPS share host capacity. Explicit CUDA devices have independent
    capacity; unknown placement conflicts with every domain. Foreground calls
    bypass this budget. Configure it only while no reservations are outstanding.
    """

    def __init__(self, *, enabled=False):
        self._condition = threading.Condition()
        self._enabled = bool(enabled)
        self._active = {}
        self._waiting = []

    def configure(self, enabled):
        """Change the policy at a quiescent boundary, or raise RuntimeError."""
        with self._condition:
            if self._active or self._waiting:
                raise RuntimeError("Background capacity is still in use")
            self._enabled = bool(enabled)

    @contextmanager
    def reserve(self, device=None):
        """Wait FIFO outside model locks; release capacity on every exit path.

        A nested call inherits its thread's reservation and must stay within
        that resource domain. Waiting for another worker under a reservation
        is unsupported, as with native model/device ownership.
        """
        ident, domain = threading.get_ident(), _domain(device)
        acquired = False
        lease = CapacityLease()
        with self._condition:
            if self._enabled:
                if ident in self._active:
                    outer = self._active[ident]
                    if outer is not None and domain != outer:
                        raise RuntimeError("Nested background inference changes resource domain")
                    lease = CapacityLease(reentrant=True)
                else:
                    started = time.perf_counter()
                    ticket = (object(), domain)
                    self._waiting.append(ticket)
                    try:
                        while (any(_conflicts(domain, other) for other in self._active.values())
                               or any(_conflicts(domain, other) for _, other in
                                      self._waiting[:self._waiting.index(ticket)])):
                            self._condition.wait()
                        self._active[ident] = domain
                        acquired = True
                        lease = CapacityLease(time.perf_counter()-started)
                    finally:
                        self._waiting.remove(ticket)
                        self._condition.notify_all()
        try:
            yield lease
        finally:
            if acquired:
                with self._condition:
                    del self._active[ident]
                    self._condition.notify_all()

    def snapshot(self):
        """Return bounded scheduling metadata without querying model runtimes."""
        with self._condition:
            return {"enabled": self._enabled, "active": len(self._active),
                    "waiting": len(self._waiting)}


def _enabled_from_environment(default):
    value = os.environ.get("VOICEMEM_ENGINE_BACKGROUND_CAPACITY", "1" if default else "0")
    if value not in {"0", "1"}:
        raise ValueError("VOICEMEM_ENGINE_BACKGROUND_CAPACITY must be 0 or 1")
    return value == "1"


def _configured_capacity():
    return BackgroundCapacity(enabled=_enabled_from_environment(False))


background_capacity = _configured_capacity()


def configure_background_capacity(*, default=False):
    """Apply an application's default before native work starts.

    An explicit environment setting takes precedence. Invalid configuration
    raises ValueError; outstanding native reservations raise RuntimeError.
    """
    enabled = _enabled_from_environment(default)
    background_capacity.configure(enabled)
    return enabled
