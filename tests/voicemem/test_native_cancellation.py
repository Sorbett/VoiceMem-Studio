"""Cancellation before native execution must release every admission owner."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
from unittest.mock import patch

from voicemem.engine import BackgroundCapacity, InferenceCancelled, background_scope, inference_call
from voicemem.engine.scheduling.admission import ResourceAdmission, SpeechFeedback


class NativeCancellationTests(unittest.TestCase):
    def wait_for(self, predicate):
        deadline = time.monotonic() + 1
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.001)
        self.assertTrue(predicate())

    def test_capacity_wait_cancels_before_owner_finishes_and_leaves_no_ticket(self):
        capacity = BackgroundCapacity(enabled=True)
        cancelled = threading.Event()
        calls = []

        def work():
            with background_scope(None, cancelled=cancelled.is_set):
                with inference_call('synthetic', 'torch'):
                    calls.append(True)

        with patch('voicemem.engine.telemetry.metrics.background_capacity', capacity), \
                ThreadPoolExecutor(max_workers=1) as pool:
            with capacity.reserve():
                future = pool.submit(work)
                self.wait_for(lambda: capacity.snapshot()['waiting'] == 1)
                cancelled.set()
                with self.assertRaises(InferenceCancelled):
                    future.result(timeout=1)
                self.assertEqual(capacity.snapshot()['waiting'], 0)
                self.assertEqual(capacity.snapshot()['active'], 1)
            self.assertEqual(capacity.snapshot()['active'], 0)
        self.assertEqual(calls, [])

    def test_speech_wait_cancels_without_running_or_forcing_progress(self):
        admission, feedback = ResourceAdmission(), SpeechFeedback()
        cancelled = threading.Event()

        def work():
            with background_scope(None, cancelled=cancelled.is_set):
                with inference_call('synthetic', 'torch'):
                    self.fail('obsolete inference ran')

        with patch('voicemem.engine.telemetry.metrics.resources', admission), \
                admission.producing(feedback), ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(work)
            self.wait_for(lambda: admission.deferred == 1)
            cancelled.set()
            with self.assertRaises(InferenceCancelled):
                future.result(timeout=1)
        self.assertEqual(admission.forced, 0)

    def test_model_lock_wait_releases_capacity_on_cancellation(self):
        capacity = BackgroundCapacity(enabled=True)
        cancelled = threading.Event()
        lock = threading.Lock()

        def work():
            with background_scope(None, cancelled=cancelled.is_set):
                with inference_call('synthetic', 'torch', lock=lock):
                    self.fail('obsolete inference ran')

        with patch('voicemem.engine.telemetry.metrics.background_capacity', capacity), \
                ThreadPoolExecutor(max_workers=1) as pool:
            with lock:
                future = pool.submit(work)
                self.wait_for(lambda: capacity.snapshot()['active'] == 1)
                cancelled.set()
                with self.assertRaises(InferenceCancelled):
                    future.result(timeout=1)
                self.assertTrue(lock.locked())
                self.assertEqual(capacity.snapshot()['active'], 0)

    def test_cancel_after_native_start_retains_capacity_and_result_until_completion(self):
        capacity = BackgroundCapacity(enabled=True)
        cancelled, entered, release = (threading.Event() for _ in range(3))

        def work():
            with background_scope(None, cancelled=cancelled.is_set):
                with inference_call('synthetic', 'torch'):
                    entered.set()
                    release.wait(2)
                    return 'unchanged result'

        with patch('voicemem.engine.telemetry.metrics.background_capacity', capacity), \
                ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(work)
            try:
                self.assertTrue(entered.wait(1))
                cancelled.set()
                self.assertEqual(capacity.snapshot()['active'], 1)
                self.assertFalse(future.done())
            finally:
                release.set()
            self.assertEqual(future.result(timeout=1), 'unchanged result')
            self.assertEqual(capacity.snapshot()['active'], 0)
        with background_scope(None), inference_call('durable', 'torch'):
            pass
