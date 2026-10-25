"""Output-owned transport pacing from validated browser buffer reports."""
import asyncio
import math
import time


class PlaybackBudget:
    """Pause ahead-of-playback delivery only while browser feedback is fresh.

    Missing feedback preserves legacy delivery. Estimates are scheduling data,
    never evidence of heard audio. The owning conversation validates output IDs.
    """

    def __init__(self, *, high_s=2.0, low_s=1.0, stale_s=2.0, clock=time.monotonic):
        self.high_s, self.low_s, self.stale_s = high_s, low_s, stale_s
        self.clock = clock
        self._updated = None
        self._buffer_s = 0.0
        self._state = "starting"
        self._changed = asyncio.Event()

    def checkpoint(self, samples, rate, state):
        try:
            seconds = float(samples) / float(rate)
            valid = float(rate) > 0 and math.isfinite(float(rate))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return
        if not valid or not math.isfinite(seconds) or seconds < 0:
            return
        self._buffer_s = min(seconds, 120.0)
        self._updated = self.clock()
        self._state = state
        self._changed.set()

    def sent(self, seconds):
        self._buffer_s += seconds

    async def wait(self):
        """Yield above the high watermark until low, stale, resumed or cancelled."""
        threshold = self.high_s
        while self._updated is not None:
            age = max(0.0, self.clock() - self._updated)
            if age >= self.stale_s:
                return
            remaining = self._buffer_s - (age if self._state == "playing" else 0.0)
            if remaining <= threshold:
                return
            threshold = self.low_s
            self._changed.clear()
            delay = min(.1, self.stale_s - age)
            try:
                await asyncio.wait_for(self._changed.wait(), delay)
            except asyncio.TimeoutError:
                pass
