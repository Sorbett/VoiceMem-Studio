"""Bounded reference-code reuse on a speech provider's owning worker."""
from copy import deepcopy
import hashlib

import numpy as np


class ReferenceAudioCache:
    """Cache one exact waveform encoding without exposing mutable cached output.

    The caller retains codec thread ownership. Sample rate, shape, dtype and
    waveform bytes must all match; replacement audio evicts the previous entry.
    """

    def __init__(self, codec):
        self.codec = codec
        self._key = self._encoded = None
        self.hits = self.misses = 0

    def __getattr__(self, name):
        return getattr(self.codec, name)

    def encode(self, audio, *, sr):
        samples = np.ascontiguousarray(audio)
        key = (sr, samples.shape, samples.dtype.str,
               hashlib.sha256(samples.tobytes()).digest())
        if key == self._key:
            self.hits += 1
            return deepcopy(self._encoded)
        encoded = self.codec.encode(audio, sr=sr)
        self._encoded = deepcopy(encoded)
        self._key = key
        self.misses += 1
        return encoded

    def clear(self):
        """Release the reference entry before its provider is closed."""
        self._key = self._encoded = None
