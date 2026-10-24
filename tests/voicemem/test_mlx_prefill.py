"""Cooperative prefill boundaries without optional MLX runtime imports."""
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tests.helpers.gpu import cpu_gpu_loop
from voicemem.engine.backends.mlx.prefill import prefill_steps


class PrefillTests(unittest.TestCase):
    def test_generation_keeps_exactly_the_last_token_for_sampling(self):
        tokens, seen = list(range(8)), []
        cache = [SimpleNamespace(state=None)]
        def model(values, *, cache):
            seen.extend(values[0].tolist())
        with cpu_gpu_loop() as gpu:
            result = gpu.run_steps(lambda: prefill_steps(
                model, tokens, cache, step_size=3, keep_last=True))
        self.assertEqual(result, 7)
        self.assertEqual(seen + tokens[result:], tokens)

    def test_short_prompt_needs_no_extra_forward(self):
        model = Mock()
        with cpu_gpu_loop() as gpu:
            for tokens in ([], [7]):
                self.assertIsNone(gpu.run_steps(lambda: prefill_steps(
                    model, tokens, [], keep_last=True)))
            with self.assertRaises(ValueError):
                gpu.run_steps(lambda: prefill_steps(model, [1], [], step_size=0))
        model.assert_not_called()

    def test_queued_cancellation_does_not_create_or_load_model(self):
        entered, release, cancelled = (threading.Event() for _ in range(3))
        factory = Mock(return_value=iter([1]))
        def occupied():
            entered.set()
            release.wait(2)
            yield None
        with cpu_gpu_loop() as gpu:
            blocker = gpu.iter(occupied)
            self.assertTrue(entered.wait(1))
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as pool:
                work = pool.submit(gpu.run_steps, factory, cancelled=cancelled)
                cancelled.set()
                release.set()
                self.assertIsNone(work.result(timeout=2))
            blocker.cancel()
        factory.assert_not_called()

    def test_owner_thread_nested_call_finishes_and_closes_generator(self):
        closed = Mock()
        def steps():
            try:
                yield 1
                yield 2
            finally:
                closed()
        with cpu_gpu_loop() as gpu:
            self.assertEqual(gpu.call(lambda: gpu.run_steps(steps)), 2)
        closed.assert_called_once()

    def test_cancelled_cleanup_failure_reaches_caller_after_owner_cleanup(self):
        cancelled, closed = threading.Event(), threading.Event()
        def steps():
            try:
                cancelled.set()
                yield 1
            finally:
                closed.set()
                raise ValueError("synthetic cleanup failure")
        with cpu_gpu_loop() as gpu:
            with self.assertRaisesRegex(ValueError, "synthetic cleanup failure"):
                gpu.run_steps(steps, cancelled=cancelled)
            self.assertTrue(closed.is_set())
            self.assertEqual(gpu.run_steps(lambda: iter(["next"])), "next")
