"""Process-local change revisions for one persistent Memory Space."""
from collections import OrderedDict
import logging
from pathlib import Path
import threading
import uuid


class MemoryChanges:
    """Publish committed mutations without retaining memory text or doing I/O.

    Listeners run on the writer thread, outside the revision lock, and must only
    schedule work. Subscriber failures never turn a committed write into a failure.
    Revisions and recent IDs are advisory process state, not a persistent journal.
    """

    def __init__(self):
        self.epoch = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._revision = 0
        self._recent = OrderedDict()
        self._listeners = set()

    def state(self):
        """Return the current revision and recently changed IDs, newest first."""
        with self._lock:
            return self._revision, list(reversed(self._recent))

    def subscribe(self, listener):
        """Register a nonblocking listener and return an idempotent unsubscribe."""
        with self._lock:
            self._listeners.add(listener)

        def unsubscribe():
            with self._lock:
                self._listeners.discard(listener)
        return unsubscribe

    def publish(self, memory_ids=()):
        """Notify listeners after a mutation commits; return its revision."""
        with self._lock:
            self._revision += 1
            revision = self._revision
            for mid in memory_ids:
                mid = str(mid)
                self._recent.pop(mid, None)
                self._recent[mid] = None
            while len(self._recent) > 128:
                self._recent.popitem(last=False)
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(revision)
            except Exception:
                logging.getLogger(__name__).exception("Memory change listener failed")
        return revision


_feeds = {}
_feeds_lock = threading.Lock()


def for_space(root):
    """Share one revision source between stores and observers of the same root."""
    key = str(Path(root).expanduser().resolve())
    with _feeds_lock:
        feed = _feeds.get(key)
        if feed is None:
            feed = _feeds[key] = MemoryChanges()
        return feed
