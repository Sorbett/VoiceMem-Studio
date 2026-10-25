"""Bounded snapshot ownership, runtime pacing and cancellation without native models."""
from contextvars import ContextVar
import unittest

from tests.helpers.probes import ManualExecutor
from voicemem.engine import LatestProbe


class ProbeSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.now = 1.0
        self.executor = ManualExecutor()
        self.probe = LatestProbe(self.executor, clock=lambda: self.now)

    def test_one_slot_and_results_are_consumed_once(self):
        self.assertTrue(self.probe.submit(lambda: .9, context="first"))
        self.assertFalse(self.probe.submit(lambda: self.fail("obsolete offer ran")))
        self.assertIsNone(self.probe.poll())
        self.assertFalse(self.probe.ready)
        self.assertEqual(len(self.executor.jobs), 1)
        self.executor.run_next()
        self.assertFalse(self.probe.ready)
        result = self.probe.poll()
        self.assertEqual((result.value, result.context, result.error), (.9, "first", None))
        self.assertIsNone(self.probe.poll())
        self.assertTrue(self.probe.ready)

    def test_slow_native_duration_spaces_probes_and_excludes_queue_wait(self):
        def score():
            self.now += .08
            return .1
        self.probe.submit(score)
        self.now += 2
        self.executor.run_next()
        result = self.probe.poll()
        self.assertAlmostEqual(result.run_s, .08)
        self.assertAlmostEqual(result.queue_s, 2)
        self.assertFalse(self.probe.ready)
        self.assertAlmostEqual(self.probe.snapshot()["cooling_ms"], 80)
        self.now += .079
        self.assertFalse(self.probe.submit(lambda: None))
        self.now += .002
        self.assertTrue(self.probe.submit(lambda: None))

    def test_fast_probe_has_no_additional_recovery_gap(self):
        def score():
            self.now += .01
            return .1
        self.probe.submit(score)
        self.executor.run_next()
        self.probe.poll()
        self.assertTrue(self.probe.ready)
        self.assertEqual(self.probe.snapshot()["cooling_ms"], 0)

    def test_reset_cancels_queued_work_before_provider_entry(self):
        self.probe.submit(lambda: self.fail("cancelled provider ran"))
        self.probe.invalidate()
        self.executor.run_next()
        self.assertIsNone(self.probe.poll())
        self.assertTrue(self.probe.ready)
        self.assertEqual(self.probe.snapshot()["discarded"], 1)

    def test_running_old_work_retains_ownership_and_discards_results_or_errors(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                def score():
                    if failure:
                        raise ValueError("synthetic stale failure")
                    return .99
                self.probe.submit(score)
                finish = self.executor.start_next()
                self.probe.invalidate()
                self.assertFalse(self.probe.ready)
                self.assertFalse(self.probe.submit(lambda: None))
                finish()
                self.assertIsNone(self.probe.poll())
                self.assertTrue(self.probe.ready)
        self.assertEqual(self.probe.snapshot()["discarded"], 2)

    def test_close_cancels_queued_work_and_never_reopens(self):
        self.probe.submit(lambda: self.fail("closed provider ran"))
        self.probe.close()
        self.executor.run_next()
        self.assertFalse(self.probe.ready)
        self.assertFalse(self.probe.submit(lambda: None))
        self.assertIsNone(self.probe.poll())

    def test_caller_context_and_current_errors_cross_executor_boundary(self):
        role = ContextVar("synthetic_probe_role", default="unset")
        token = role.set("audio")
        try:
            self.probe.submit(role.get)
        finally:
            role.reset(token)
        self.executor.run_next()
        self.assertEqual(self.probe.poll().value, "audio")

        def fail():
            raise ValueError("synthetic current failure")
        self.probe.submit(fail)
        self.executor.run_next()
        self.assertIsInstance(self.probe.poll().error, ValueError)
        self.assertEqual(set(self.probe.snapshot()), {
            "pending", "submitted", "completed", "discarded", "run_ms", "queue_ms", "cooling_ms"})


if __name__ == "__main__":
    unittest.main()
