"""Operation-local reuse, bounded ownership and cancellation regressions."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import threading
import unittest
from unittest.mock import Mock, patch

from voicemem.engine import (InferenceCancelled, ReusableResult, background_scope,
                             reuse_call, reuse_scope)
from voicemem.engine.scheduling.cancellation import cancellation_scope
from voicemem.engine.scheduling.reuse import InferenceReuse


def freeze(value, limit):
    data = bytes(value)
    return ReusableResult(data, len(data)) if len(data) <= limit else None


def thaw(payload):
    return bytearray(payload)


class InferenceReuseTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("voicemem.engine.telemetry.metrics._metrics", {}))

    def run_call(self, pool, key, operation):
        return pool.run(key, operation, freeze=freeze, thaw=thaw)

    def test_invalid_bounds_are_rejected(self):
        for options in ({"max_entries": 0}, {"max_bytes": 0},
                        {"max_entries": 1.5}, {"max_bytes": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                InferenceReuse(**options)

    def test_failure_does_not_evict_an_independent_successful_result(self):
        pool = InferenceReuse()
        self.run_call(pool, "ok", lambda: b"ok")
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            self.run_call(pool, "error", Mock(side_effect=RuntimeError("fixture failure")))
        self.assertEqual(self.run_call(pool, "ok", lambda: self.fail("lost result")), (b"ok", True))
        self.assertEqual(pool.snapshot()["pending"], 0)

    def test_results_are_independent_and_limits_evict_least_recently_used(self):
        pool = InferenceReuse(max_entries=2, max_bytes=4)
        one = bytearray(b"ab")
        self.assertEqual(self.run_call(pool, "a", lambda: one), (one, False))
        one[:] = b"xx"
        hit, reused = self.run_call(pool, "a", lambda: self.fail("cache miss"))
        self.assertTrue(reused)
        self.assertEqual(hit, b"ab")
        hit[:] = b"yy"
        self.run_call(pool, "b", lambda: b"cd")
        self.run_call(pool, "a", lambda: self.fail("cache miss"))
        self.run_call(pool, "c", lambda: b"ef")
        value, reused = self.run_call(pool, "b", lambda: b"gh")
        self.assertFalse(reused)
        self.assertEqual(value, b"gh")
        self.assertEqual(pool.snapshot(), {"entries": 2, "pending": 0, "bytes": 4})

    def test_oversized_and_declined_results_are_not_retained(self):
        pool = InferenceReuse(max_bytes=2)
        operation = Mock(return_value=b"large")
        self.run_call(pool, "a", operation)
        self.run_call(pool, "a", operation)
        self.assertEqual(operation.call_count, 2)
        self.assertEqual(pool.snapshot(), {"entries": 0, "pending": 0, "bytes": 0})

    def test_concurrent_identical_misses_execute_once(self):
        pool = InferenceReuse()
        entered, release = threading.Event(), threading.Event()
        def native():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("fixture not released")
            return bytearray(b"fixture")
        operation = Mock(side_effect=native)
        with ThreadPoolExecutor(max_workers=3) as workers:
            first = workers.submit(self.run_call, pool, "a", operation)
            self.assertTrue(entered.wait(1))
            others = [workers.submit(self.run_call, pool, "a", operation) for _ in range(2)]
            release.set()
            results = [job.result(timeout=1) for job in (first, *others)]
        operation.assert_called_once()
        self.assertEqual([reused for _, reused in results], [False, True, True])
        self.assertEqual([bytes(value) for value, _ in results], [b"fixture"] * 3)
        self.assertEqual(len({id(value) for value, _ in results}), 3)

    def test_failure_is_visible_to_waiters_and_retry_can_compute(self):
        pool = InferenceReuse()
        entered, release, joined = (threading.Event() for _ in range(3))
        def fail():
            entered.set()
            if not release.wait(2):
                raise RuntimeError("fixture not released")
            raise TimeoutError("native fixture error")
        original = __import__("voicemem.engine.scheduling.reuse", fromlist=["Future"]).Future
        class ObservedFuture(original):
            def result(self, timeout=None):
                joined.set()
                return super().result(timeout)
        with patch("voicemem.engine.scheduling.reuse.Future", ObservedFuture), ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(self.run_call, pool, "a", fail)
            self.assertTrue(entered.wait(1))
            second = workers.submit(self.run_call, pool, "a", fail)
            try:
                self.assertTrue(joined.wait(1))
            finally:
                release.set()
            for job in (first, second):
                with self.assertRaisesRegex(TimeoutError, "native fixture error"):
                    job.result(timeout=1)
        self.assertEqual(self.run_call(pool, "a", lambda: b"ok"), (b"ok", False))

    def test_waiter_cancellation_does_not_release_active_native_ownership(self):
        pool = InferenceReuse()
        entered, release, cancelled = (threading.Event() for _ in range(3))
        def native():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("fixture not released")
            return b"ok"
        def wait():
            with cancellation_scope(cancelled.is_set):
                return self.run_call(pool, "a", lambda: self.fail("duplicate inference"))
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(self.run_call, pool, "a", native)
            self.assertTrue(entered.wait(1))
            second = workers.submit(wait)
            cancelled.set()
            try:
                with self.assertRaises(InferenceCancelled):
                    second.result(timeout=1)
                self.assertEqual(pool.snapshot()["pending"], 1)
            finally:
                release.set()
            self.assertEqual(first.result(timeout=1), (b"ok", False))
        self.assertEqual(self.run_call(pool, "a", lambda: self.fail("cache miss")), (b"ok", True))

    def test_recursion_and_excess_pending_misses_bypass_without_deadlock(self):
        pool = InferenceReuse(max_entries=1)
        def recursive():
            return self.run_call(pool, "a", lambda: b"nested")[0]
        self.assertEqual(self.run_call(pool, "a", recursive), (b"nested", False))
        pool = InferenceReuse(max_entries=1)
        entered, release = threading.Event(), threading.Event()
        def native():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("fixture not released")
            return b"one"
        with ThreadPoolExecutor(max_workers=1) as workers:
            first = workers.submit(self.run_call, pool, "a", native)
            try:
                self.assertTrue(entered.wait(1))
                self.assertEqual(self.run_call(pool, "b", lambda: b"two"), (b"two", False))
                self.assertEqual(pool.snapshot()["pending"], 1)
            finally:
                release.set()
            first.result(timeout=1)

    def test_scopes_capture_children_and_do_not_share_between_jobs(self):
        operation = Mock(return_value=b"ok")
        key = Mock(return_value="a")
        def call():
            return reuse_call(key, operation, freeze=freeze, thaw=thaw)
        self.assertFalse(call()[1])
        key.assert_not_called()
        with background_scope(None), ThreadPoolExecutor(max_workers=1) as workers:
            self.assertFalse(call()[1])
            captured = copy_context()
            self.assertTrue(workers.submit(captured.run, call).result(timeout=1)[1])
            with reuse_scope():
                self.assertTrue(call()[1])
        with background_scope(None):
            self.assertFalse(call()[1])
        self.assertFalse(call()[1])
        self.assertEqual(operation.call_count, 4)

    def test_cache_metrics_never_reset_native_cost_or_thermal_recovery(self):
        from voicemem.engine.telemetry.metrics import record
        with patch("voicemem.engine.telemetry.metrics.resources") as resources, background_scope(None):
            record("embedding", "cache", .001)
            resources.observe.assert_not_called()
            record("embedding", "torch", .1)
            resources.observe.assert_called_once_with("embedding", .1, device=None)

    def test_explicit_opt_out_is_inherited_by_nested_background_scopes(self):
        operation = Mock(return_value=b"ok")
        def call():
            return reuse_call(lambda: "a", operation, freeze=freeze, thaw=thaw)
        with reuse_scope():
            self.assertFalse(call()[1])
            with reuse_scope(enabled=False), background_scope(None):
                self.assertFalse(call()[1])
                self.assertFalse(call()[1])
            self.assertTrue(call()[1])
        self.assertEqual(operation.call_count, 3)


if __name__ == "__main__":
    unittest.main()
