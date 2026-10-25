"""Bound PCM held between synthesis and ordered delivery on one event loop."""
import asyncio
from collections import deque


class PCMBuffer:
    """Preserve chunks and terminal errors with a byte and item budget.

    One oversized provider chunk is allowed when empty. Closing never waits for
    capacity, including when the consumer has failed or has been cancelled.
    """

    def __init__(self, sample_rate, *, seconds=1.0, max_items=8):
        self.limit = max(1, int(sample_rate * 2 * seconds))
        self.max_items = max_items
        self._items = deque()
        self._bytes = 0
        self._closed = False
        self._error = None
        self._changed = asyncio.Event()

    async def put(self, chunk):
        size = len(getattr(chunk, "pcm", chunk))
        while self._items and (self._bytes + size > self.limit
                               or len(self._items) >= self.max_items):
            if self._closed:
                raise RuntimeError("PCM buffer is closed")
            self._changed.clear()
            await self._changed.wait()
        if self._closed:
            raise RuntimeError("PCM buffer is closed")
        self._items.append((chunk, size))
        self._bytes += size
        self._changed.set()

    async def get(self):
        while not self._items and not self._closed:
            self._changed.clear()
            await self._changed.wait()
        if self._items:
            chunk, size = self._items.popleft()
            self._bytes -= size
            self._changed.set()
            return chunk
        error, self._error = self._error, None
        return error

    def close(self, error=None):
        """Deliver any queued PCM, then the error once, followed by EOF."""
        self._closed = True
        self._error = error
        self._changed.set()
