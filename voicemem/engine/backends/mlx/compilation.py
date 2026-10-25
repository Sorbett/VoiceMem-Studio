"""Validated compilation for array-valued decode steps on the MLX owner."""
from __future__ import annotations

import logging


class VerifiedDecode:
    """Compile a pure array step after checking its output and RNG transition.

    Construct, verify and call on the existing GpuLoop owner. The step may update
    ``mx.random.state`` but must not mutate model weights or external caches.
    Explicit startup verification replays both paths from a supplied seed and
    resets that seed before returning. Do not verify during a live generation.
    Compilation failure or mismatch disables compilation for this instance.
    Eager failures and subsequent compiled failures propagate. Unverified calls
    and input shape or dtype changes use eager execution without new traces.
    Verification covers the supplied sample, not every possible numerical input.
    """

    def __init__(self, step, *, label):
        self.step = step
        self.label = label
        self.verified = False
        self.enabled = False
        self._compiled = None
        self._signature = None
        self._compiled_calls = self._eager_calls = 0

    def __call__(self, *args):
        matching = self._signature == self._input_signature(args)
        if self.enabled and matching:
            self._compiled_calls += 1
            return self._compiled(*args)
        self._eager_calls += 1
        return self.step(*args)

    def snapshot(self):
        """Return owner-thread execution counts without model inputs or RNG keys."""
        return dict(verified=self.verified, enabled=self.enabled,
                    compiled_calls=self._compiled_calls, eager_calls=self._eager_calls)

    def verify(self, *args, seed):
        """Check a startup sample, resetting its seed even on a new check's failure.

        Returns whether compilation is enabled for the supplied input signature.
        Eager execution errors propagate; compile errors log a static reason.
        Repeated checks return the existing result without touching RNG state.
        """
        if self.verified:
            return self.enabled and self._signature == self._input_signature(args)

        import mlx.core as mx

        reason = None
        try:
            mx.random.seed(seed)
            expected = self.step(*args)
            after = list(mx.random.state)
            mx.eval(expected, *after)
            try:
                mx.random.seed(seed)
                compiled = mx.compile(self.step, inputs=mx.random.state,
                                      outputs=mx.random.state)
                actual = compiled(*args)
                mx.eval(actual, *mx.random.state)
                if not self._equal(mx, expected, actual):
                    reason = 'output_mismatch'
                elif len(after) != len(mx.random.state) or not all(
                        self._equal(mx, left, right)
                        for left, right in zip(after, mx.random.state)):
                    reason = 'random_state_mismatch'
                else:
                    self._compiled = compiled
                    self.enabled = True
            except Exception as exc:
                reason = f'compile_error:{type(exc).__name__}'
        finally:
            mx.random.seed(seed)

        self.verified = True
        self._signature = self._input_signature(args)
        if reason:
            logging.getLogger(__name__).warning(
                '[mlx-compile] model=%s enabled=false reason=%s', self.label, reason)
        else:
            logging.getLogger(__name__).info(
                '[mlx-compile] model=%s enabled=true verified=true', self.label)
        return self.enabled

    @staticmethod
    def _input_signature(args):
        return tuple((value.shape, value.dtype) for value in args)

    @staticmethod
    def _equal(mx, left, right):
        return (left.shape == right.shape and left.dtype == right.dtype
                and bool(mx.array_equal(left, right)))
