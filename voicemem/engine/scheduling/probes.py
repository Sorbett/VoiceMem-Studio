"""Nonblocking admission for replaceable CPU inference snapshots."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
import time


eot_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voicemem-eot")


@dataclass(frozen=True)
class ProbeResult:
    """A completed snapshot and its captured ownership and wall timings."""

    context: object
    value: object
    error: Exception | None
    run_s: float
    queue_s: float
    finished_at: float


class LatestProbe:
    """Own at most one queued or running snapshot without blocking its producer.

    Call poll before offering the latest immutable snapshot. Busy offers are
    coalesced by the caller retaining current input, not by queuing old copies.
    Invalidation cancels unstarted work and rejects started work's result while
    retaining its slot until native completion. All methods belong to one loop.
    """

    def __init__(self, executor, *, budget_s=.04, clock=time.monotonic):
        self._executor, self._clock = executor, clock
        self._budget_s = budget_s
        self._future = None
        self._generation = 0
        self._submitted_generation = 0
        self._ready_at = 0.0
        self._closed = False
        self._run_s = self._queue_s = 0.0
        self.submitted = self.completed = self.discarded = 0

    @property
    def ready(self):
        return not self._closed and self._future is None and self._clock() >= self._ready_at

    def submit(self, operation, *, context=None):
        """Offer an immutable snapshot operation; return False while occupied or cooling."""
        if not self.ready:
            return False
        submitted = self._clock()
        captured = copy_context()

        def run():
            started = self._clock()
            value = error = None
            try:
                value = captured.run(operation)
            except Exception as exc:
                error = exc
            finished = self._clock()
            return ProbeResult(context, value, error, finished-started,
                               started-submitted, finished)

        self._future = self._executor.submit(run)
        self._submitted_generation = self._generation
        self.submitted += 1
        return True

    def poll(self):
        """Consume a completed result once, or return None without waiting."""
        future = self._future
        if future is None or not future.done():
            return None
        self._future = None
        result = future.result()
        self._run_s, self._queue_s = result.run_s, result.queue_s
        self.completed += 1
        # Over-budget probes get an equal recovery gap rather than running
        # continuously while the producer advances to a newer snapshot.
        self._ready_at = result.finished_at + (result.run_s if result.run_s > self._budget_s else 0)
        if self._closed or self._submitted_generation != self._generation:
            self.discarded += 1
            return None
        return result

    def invalidate(self):
        """Reject old results; native work already running keeps its admission slot."""
        self._generation += 1
        self._ready_at = 0.0
        if self._future is not None and self._future.cancel():
            self._future = None
            self.discarded += 1

    def close(self):
        """Cancel queued work without interrupting a native call owned by the executor."""
        self.invalidate()
        self._closed = True

    def snapshot(self):
        """Return scheduling metadata without audio, text or ownership identifiers."""
        return {"pending": self._future is not None, "submitted": self.submitted,
                "completed": self.completed, "discarded": self.discarded,
                "run_ms": self._run_s * 1000,
                "queue_ms": self._queue_s * 1000,
                "cooling_ms": max(0.0, self._ready_at-self._clock()) * 1000}
