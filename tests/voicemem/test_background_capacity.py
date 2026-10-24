"""Exercise native capacity through production timing and lifetime boundaries."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
from unittest.mock import patch

from voicemem.engine import (BackgroundCapacity, BackgroundLane, Service, background_scope,
                             device_lock, inference_call, model_metrics)
from voicemem.engine.scheduling.admission import ResourceAdmission, SpeechFeedback
from voicemem.engine.scheduling.capacity import _configured_capacity


class BackgroundCapacityTests(unittest.TestCase):
    def setUp(self):
        self.capacity = BackgroundCapacity(enabled=True)
        self.enterContext(patch("voicemem.engine.telemetry.metrics.background_capacity", self.capacity))
        model_metrics(reset=True)

    def wait_for(self, predicate):
        deadline = time.monotonic()+2
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.001)
        self.assertTrue(predicate())

    def native(self, entered, release, *, device=None, lock=None, service=Service.BACKGROUND):
        with inference_call("capacity_fixture", "torch", lock=lock, device=device, service=service):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Synthetic native work was not released")
            return b"unchanged fixture"

    def test_disabled_policy_preserves_baseline_concurrent_results(self):
        self.capacity.configure(False)
        events = [threading.Event(), threading.Event()]
        release = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(self.native, event, release, lock=threading.RLock()) for event in events]
            try:
                self.assertTrue(all(event.wait(1) for event in events))
                self.assertEqual(self.capacity.snapshot()["active"], 0)
            finally:
                release.set()
            self.assertEqual([job.result(timeout=1) for job in jobs], [b"unchanged fixture"]*2)

    def test_unrelated_model_locks_share_background_capacity_and_keep_all_results(self):
        first, second, release = (threading.Event() for _ in range(3))
        with ThreadPoolExecutor(max_workers=2) as pool:
            one = pool.submit(self.native, first, release, lock=threading.RLock())
            try:
                self.assertTrue(first.wait(1))
                two = pool.submit(self.native, second, release, lock=threading.RLock())
                self.wait_for(lambda: self.capacity.snapshot()["waiting"] == 1)
                self.assertFalse(second.is_set())
                self.assertEqual(self.capacity.snapshot()["active"], 1)
            finally:
                release.set()
            self.assertEqual(one.result(timeout=1), two.result(timeout=1))
        self.assertEqual(self.capacity.snapshot(), {"enabled": True, "active": 0, "waiting": 0})

    def test_foreground_never_queues_for_capacity_or_holds_waiters_model_lock(self):
        first, second, release = (threading.Event() for _ in range(3))
        lock = threading.RLock()
        with ThreadPoolExecutor(max_workers=2) as pool:
            one = pool.submit(self.native, first, release)
            try:
                self.assertTrue(first.wait(1))
                two = pool.submit(self.native, second, release, lock=lock)
                self.wait_for(lambda: self.capacity.snapshot()["waiting"] == 1)
                for service in (Service.AUDIO, Service.INTERACTIVE):
                    with inference_call("foreground_fixture", "torch", lock=lock, service=service):
                        self.assertFalse(second.is_set())
                self.assertEqual(self.capacity.snapshot()["waiting"], 1)
            finally:
                release.set()
            one.result(timeout=1); two.result(timeout=1)

    def test_known_cuda_devices_run_independently(self):
        first, second, release = (threading.Event() for _ in range(3))
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(self.native, first, release, device="cuda:0"),
                    pool.submit(self.native, second, release, device="cuda:1")]
            try:
                self.assertTrue(first.wait(1) and second.wait(1))
                self.assertEqual(self.capacity.snapshot()["active"], 2)
            finally:
                release.set()
            for job in jobs:
                job.result(timeout=1)

    def test_cpu_mps_same_cuda_and_unknown_placement_conflict(self):
        for left, right in (("cpu", "mps"), ("cuda", "cuda:0"), (None, "cuda:1")):
            with self.subTest(left=left, right=right), ThreadPoolExecutor(max_workers=2) as pool:
                first, second, release = (threading.Event() for _ in range(3))
                one = pool.submit(self.native, first, release, device=left)
                try:
                    self.assertTrue(first.wait(1))
                    two = pool.submit(self.native, second, release, device=right)
                    self.wait_for(lambda: self.capacity.snapshot()["waiting"] == 1)
                    self.assertFalse(second.is_set())
                finally:
                    release.set()
                one.result(timeout=1); two.result(timeout=1)

    def test_older_unknown_waiter_is_not_starved_by_new_device_work(self):
        entered, release, next_entered, next_release, last = (threading.Event() for _ in range(5))
        with ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(self.native, entered, release, device="cuda:0")
            try:
                self.assertTrue(entered.wait(1))
                second = pool.submit(self.native, next_entered, next_release)
                self.wait_for(lambda: self.capacity.snapshot()["waiting"] == 1)
                third = pool.submit(self.native, last, release, device="cuda:1")
                self.wait_for(lambda: self.capacity.snapshot()["waiting"] == 2)
                self.assertFalse(last.is_set())
                release.set()
                self.assertTrue(next_entered.wait(1))
                self.assertFalse(last.is_set())
            finally:
                release.set(); next_release.set()
            for job in (first, second, third):
                job.result(timeout=1)

    def test_failures_in_model_and_lock_resolution_release_capacity(self):
        def fail():
            raise ValueError("synthetic failure")

        for locks in (None, (threading.RLock(), fail)):
            with self.subTest(locks=locks), self.assertRaisesRegex(ValueError, "synthetic failure"):
                with inference_call("capacity_fixture", "torch", lock=locks, service=Service.BACKGROUND):
                    fail()
            self.assertEqual(self.capacity.snapshot()["active"], 0)
        with inference_call("capacity_fixture", "torch", service=Service.BACKGROUND):
            pass

    def test_nested_native_call_inherits_lease_without_repeating_speech_wait(self):
        with patch("voicemem.engine.telemetry.metrics.resources.wait", return_value=(0, False)) as wait:
            with background_scope(None), inference_call("outer_fixture", "torch"):
                with inference_call("inner_fixture", "torch"):
                    self.assertEqual(self.capacity.snapshot()["active"], 1)
            wait.assert_called_once_with("outer_fixture")
        self.assertEqual(self.capacity.snapshot()["active"], 0)

    def test_existing_device_owner_bypasses_capacity_with_a_different_model_lock(self):
        for device in ("mps", "cuda:2"):
            with self.subTest(device=device), device_lock(device), background_scope(None), \
                    patch.object(self.capacity, "reserve", side_effect=AssertionError("capacity wait under device lock")), \
                    patch("voicemem.engine.telemetry.metrics.resources.wait",
                          side_effect=AssertionError("speech wait under device lock")):
                with inference_call("nested_fixture", "torch", lock=threading.RLock()):
                    self.assertEqual(self.capacity.snapshot()["active"], 0)

    def test_capacity_wait_is_measured_separately_from_native_and_speech_wait(self):
        entered, release = threading.Event(), threading.Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(self.native, entered, release)
            self.assertTrue(entered.wait(1))
            timer = threading.Timer(.04, release.set)
            timer.start()
            try:
                with inference_call("queued_fixture", "torch", service=Service.BACKGROUND):
                    pass
            finally:
                release.set(); timer.cancel()
            first.result(timeout=1)
        row = next(row for row in model_metrics() if row["model"] == "queued_fixture")
        self.assertGreater(row["capacity_ms"], 10)
        self.assertLess(row["run_ms"], row["capacity_ms"])
        self.assertFalse(row["forced"])

    def test_speech_inventory_is_rechecked_after_capacity_queue(self):
        admission, feedback = ResourceAdmission(max_defer_s=2), SpeechFeedback()
        first, second, release = (threading.Event() for _ in range(3))
        feedback.checkpoint(240000, 24000, "paused")
        with patch("voicemem.engine.telemetry.metrics.resources", admission), ThreadPoolExecutor(max_workers=2) as pool:
            try:
                with admission.producing(feedback):
                    one = pool.submit(self.native, first, release)
                    self.assertTrue(first.wait(1))
                    two = pool.submit(self.native, second, release)
                    self.wait_for(lambda: self.capacity.snapshot()["waiting"] == 1)
                    feedback.checkpoint(0, 24000, "stalled")
                    release.set()
                    self.wait_for(lambda: admission.snapshot()["deferred"] == 1)
                    self.assertFalse(second.is_set())
                one.result(timeout=1); two.result(timeout=1)
            finally:
                release.set()
        self.assertFalse(admission.snapshot()["forced"])

    def test_configuration_cannot_change_with_outstanding_owners(self):
        with self.capacity.reserve("cpu"):
            with self.assertRaisesRegex(RuntimeError, "still in use"):
                self.capacity.configure(False)
            with self.assertRaisesRegex(RuntimeError, "resource domain"):
                with self.capacity.reserve("cuda:0"):
                    pass
        self.capacity.configure(False)
        self.assertFalse(self.capacity.snapshot()["enabled"])

    def test_interrupted_capacity_wait_removes_its_ticket(self):
        entered, release = threading.Event(), threading.Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            owner = pool.submit(self.native, entered, release)
            try:
                self.assertTrue(entered.wait(1))
                with patch.object(self.capacity._condition, "wait", side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        with self.capacity.reserve():
                            self.fail("Interrupted reservation entered native work")
                self.assertEqual(self.capacity.snapshot()["waiting"], 0)
            finally:
                release.set()
            owner.result(timeout=1)

    def test_config_is_opt_in_and_rejects_malformed_values(self):
        for value, expected in (("0", False), ("1", True)):
            with patch.dict("os.environ", {"VOICEMEM_ENGINE_BACKGROUND_CAPACITY": value}):
                self.assertEqual(_configured_capacity().snapshot()["enabled"], expected)
        with patch.dict("os.environ", {"VOICEMEM_ENGINE_BACKGROUND_CAPACITY": "invalid"}):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                _configured_capacity()


class CapacityLifetimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_caller_retains_started_native_capacity_until_cleanup(self):
        capacity, lane = BackgroundCapacity(enabled=True), BackgroundLane()
        entered, release = threading.Event(), threading.Event()

        def native():
            with background_scope(None), inference_call("lifetime_fixture", "torch"):
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("Synthetic native work was not released")

        async def idle():
            pass

        async def operation():
            await asyncio.to_thread(native)

        with patch("voicemem.engine.telemetry.metrics.background_capacity", capacity):
            caller = asyncio.create_task(lane.run(idle, operation))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                caller.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await caller
                self.assertEqual(capacity.snapshot()["active"], 1)
                self.assertTrue(lane.tasks)
            finally:
                release.set()
                await asyncio.gather(caller, *tuple(lane.tasks), return_exceptions=True)
            self.assertEqual(capacity.snapshot()["active"], 0)
