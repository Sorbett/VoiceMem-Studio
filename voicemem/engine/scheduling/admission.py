"""Cooperative background admission from live speech inventory and model cost."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import math
import threading
import time

from ..backends.devices import device_key


_speech = ContextVar("voicemem_speech_feedback", default=None)
_budget = ContextVar("voicemem_admission_budget", default=None)


class _WaitBudget:
    def __init__(self):
        self.waited = 0.0
        self.lock = threading.Lock()

    def spent(self):
        with self.lock:
            return self.waited

    def charge(self, elapsed):
        with self.lock:
            self.waited += elapsed


@contextmanager
def admission_scope():
    """Share accumulated resource waits across one job and captured child workers."""
    token = _budget.set(_budget.get() or _WaitBudget())
    try:
        yield
    finally:
        _budget.reset(token)


class SpeechFeedback:
    """Output-owned scheduling estimates, independent of heard-text accounting.

    Browser inventory replaces the producer estimate when available. Neither
    estimate is evidence that audio was heard. Native calls cannot be preempted.
    """

    def __init__(self, *, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._buffer_s = 0.0
        self._updated = clock()
        self._rtf = None
        self._step_s = None
        self._chunks = 0
        self._state = "starting"
        self._reported = False

    def _remaining(self, now):
        drain = max(0.0, now - self._updated) if self._state != "paused" else 0.0
        return max(0.0, self._buffer_s - drain)

    def observe(self, audio_s, compute_s, *, first=False):
        """Record a completed native synthesis step, excluding scheduler waits."""
        if audio_s <= 0 or compute_s < 0:
            return
        with self._lock:
            now = self._clock()
            if not self._reported:
                self._buffer_s = self._remaining(now) + audio_s
                self._updated = now
            self._chunks += 1
            # Prefill has a different cost from steady acoustic decoding.
            if self._chunks > 1 and not first:
                ratio = compute_s / audio_s
                self._rtf = ratio if self._rtf is None else .25 * ratio + .75 * self._rtf
                self._step_s = (compute_s if self._step_s is None else
                                max(compute_s, .8 * self._step_s + .2 * compute_s))
        resources.wake()

    def checkpoint(self, buffered_samples, sample_rate, state):
        """Accept inventory only after the transport validates output ownership."""
        try:
            samples, rate = float(buffered_samples), float(sample_rate)
        except (TypeError, ValueError):
            return
        if not math.isfinite(samples) or not math.isfinite(rate) or samples < 0 or rate <= 0:
            return
        with self._lock:
            self._buffer_s = min(samples / rate, 120.0)
            self._reported = True
            self._updated = self._clock()
            self._state = state
        resources.wake()

    def snapshot(self):
        with self._lock:
            return {"buffer_s": self._remaining(self._clock()),
                    "rtf": self._rtf, "step_s": self._step_s,
                    "state": self._state, "chunks": self._chunks}


class ResourceAdmission:
    """Delay background native stages while foreground speech lacks headroom.

    Deferral is bounded per stage so confirmed persistence can progress during
    continuous speech. This is admission between calls, never device-wide
    serialization or preemption of an already running model.
    """

    def __init__(self, *, max_defer_s=6.0, clock=time.monotonic):
        self._condition = threading.Condition()
        self._active = {}
        self._bound = {}
        self._retained = set()
        self._devices = {}
        self._cost = {}
        self._paging = (0.0, 0.0)
        self._pressure = {}
        self._aged_at = {}
        self.max_defer_s = max_defer_s
        self.clock = clock
        self.enabled = True
        self.deferred = self.forced = 0
        self.aged = 0

    @contextmanager
    def bind(self, feedback):
        """Retain admission across segments only after local synthesis starts."""
        with self._condition:
            self._bound[feedback] = self._bound.get(feedback, 0) + 1
        try:
            yield feedback
        finally:
            with self._condition:
                self._bound[feedback] -= 1
                if not self._bound[feedback]:
                    del self._bound[feedback]
                    if feedback in self._retained:
                        self._retained.remove(feedback)
                        self._release(feedback)
                self._condition.notify_all()

    def _release(self, feedback):
        count = self._active[feedback] - 1
        if count:
            self._active[feedback] = count
        else:
            del self._active[feedback]
            self._devices.pop(feedback, None)
            self._aged_at.pop(feedback, None)

    @contextmanager
    def producing(self, feedback, *, device=None):
        with self._condition:
            self._devices.setdefault(feedback, set()).add(device_key(device))
            if feedback in self._bound and feedback not in self._retained:
                self._retained.add(feedback)
                self._active[feedback] = self._active.get(feedback, 0) + 1
            self._active[feedback] = self._active.get(feedback, 0) + 1
        try:
            yield feedback
        finally:
            with self._condition:
                self._release(feedback)
                self._condition.notify_all()

    def wake(self):
        with self._condition:
            self._condition.notify_all()

    def paging(self, mib_per_s):
        """Publish fresh swap traffic; allocated swap alone is not pressure."""
        with self._condition:
            self._paging = (max(0.0, mib_per_s), self.clock())
            self._condition.notify_all()

    def device_pressure(self, device, constrained):
        """Publish device-local free-memory pressure from its native owner."""
        with self._condition:
            self._pressure[device_key(device)] = (bool(constrained), self.clock())
            self._condition.notify_all()

    def observe(self, model, elapsed, *, device=None):
        with self._condition:
            key = (model, device_key(device))
            old = self._cost.get(key, elapsed)
            self._cost[key] = max(elapsed, .8 * old + .2 * elapsed)

    def _blocked(self, model, now, device=None, *, aged=False):
        device = device_key(device)
        cost = self._cost.get((model, device), .35)
        paging, sampled = self._paging
        constrained, measured = self._pressure.get(device, (False, 0.))
        pressured = ((now - sampled < 3 and paging > 8)
                     or (now - measured < 3 and constrained))
        aged = aged and not pressured
        reserve = max(.2 if aged else .45, 1.5 * cost)
        if pressured:
            reserve *= 2
        for feedback in self._active:
            placements = self._devices.get(feedback, {None})
            if device and device.startswith("cuda:") and all(
                    d and d.startswith("cuda:") and d != device for d in placements):
                continue
            state = feedback.snapshot()
            if state["state"] == "paused":
                continue
            rtf = state["rtf"]
            if aged and (rtf is None or state["state"] == "stalled" or state["chunks"] < 2
                         or self._aged_at.get(feedback) == state["chunks"]):
                return True
            headroom = (max(reserve, cost + state["step_s"] + .08) if aged else reserve)
            # Aged work may spend safe inventory once per new synthesis step;
            # saturated production retains the bounded forced-progress path.
            if state["buffer_s"] < headroom or (rtf is not None and rtf >= (1.0 if aged else .95)):
                return True
        return False

    def wait(self, model, *, device=None):
        """Wait outside model/storage locks; return wait seconds and aging flag."""
        started = self.clock()
        waited = False
        budget = _budget.get()
        try:
            with self._condition:
                while self.enabled and self._blocked(model, self.clock(), device):
                    elapsed = self.clock() - started
                    if (budget is not None and budget.spent() + elapsed >= self.max_defer_s
                            and not self._blocked(model, self.clock(), device, aged=True)):
                        self.aged += 1
                        self._aged_at.update((feedback, feedback.snapshot()["chunks"])
                                             for feedback in self._active)
                        return elapsed, False
                    remaining = self.max_defer_s - elapsed
                    if remaining <= 0:
                        self.forced += 1
                        self._aged_at.update((feedback, feedback.snapshot()["chunks"])
                                             for feedback in self._active)
                        return elapsed, True
                    if not waited:
                        self.deferred += 1
                        waited = True
                    self._condition.wait(min(.05, remaining))
            return self.clock() - started, False
        finally:
            if budget is not None and waited:
                budget.charge(self.clock() - started)

    def snapshot(self):
        with self._condition:
            return {"speech_jobs": len(self._active), "deferred": self.deferred,
                    "forced": self.forced, "aged": self.aged}


resources = ResourceAdmission()


@contextmanager
def speech_scope(feedback=None, *, producing=True, device=None):
    """Capture one output's feedback; only native producers reserve resources."""
    feedback = feedback or _speech.get() or SpeechFeedback()
    token = _speech.set(feedback)
    try:
        if producing:
            with resources.producing(feedback, device=device):
                yield feedback
        else:
            with resources.bind(feedback):
                yield feedback
    finally:
        _speech.reset(token)


def observe_speech(audio_s, compute_s, *, first=False):
    """Update the synthesis job's captured output without retaining PCM."""
    feedback = _speech.get()
    if feedback is not None:
        feedback.observe(audio_s, compute_s, first=first)
