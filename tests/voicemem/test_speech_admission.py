"""Cooperative cross-backend admission and metadata-only timing regressions."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import ctypes
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from voicemem.engine import (InferenceLock, Service, background_scope, inference_call,
                             model_metrics, speech_scope)
from voicemem.engine.scheduling.admission import ResourceAdmission, SpeechFeedback, _speech
from voicemem.engine.scheduling.admission import _budget
from voicemem.engine.telemetry.system import EngineObserver, paging_rates
from voicemem.engine.telemetry.darwin import DarwinCounters, VMStatistics64


class SpeechAdmissionTests(unittest.TestCase):
    def aged_fixture(self):
        now = [20.]
        admission = ResourceAdmission(max_defer_s=.1, clock=lambda: now[0])
        feedback = SpeechFeedback(clock=lambda: 20.)
        feedback.observe(.16, .08, first=True)
        feedback.observe(.16, .08)
        feedback.checkpoint(7200, 24000, "playing")
        admission.observe("fixture", .03)
        def advance(seconds):
            now[0] += seconds
        return admission, feedback, advance

    def test_one_job_accumulates_waits_and_requires_audio_progress_between_aged_calls(self):
        admission, feedback, advance = self.aged_fixture()
        with patch.object(admission._condition, "wait", side_effect=advance), \
                admission.producing(feedback), background_scope(None):
            first, forced = admission.wait("fixture")
            self.assertAlmostEqual(first, .1)
            self.assertFalse(forced)
            self.assertEqual(admission.snapshot()["aged"], 1)
            feedback.observe(.16, .08)
            second, forced = admission.wait("fixture")
            self.assertEqual(second, 0)
            self.assertFalse(forced)
            self.assertEqual(admission.snapshot()["aged"], 2)
            third, forced = admission.wait("fixture")
            self.assertAlmostEqual(third, .1)
            self.assertTrue(forced)
        self.assertEqual(admission._aged_at, {})

    def test_new_job_does_not_inherit_previous_wait_credit(self):
        admission, feedback, advance = self.aged_fixture()
        with patch.object(admission._condition, "wait", side_effect=advance), admission.producing(feedback):
            for _ in range(2):
                feedback.observe(.16, .08)
                with background_scope(None):
                    elapsed, forced = admission.wait("fixture")
                    self.assertAlmostEqual(elapsed, .1)
                    self.assertFalse(forced)

    def test_aging_never_bypasses_stall_native_cost_or_fresh_memory_pressure(self):
        for reason in ("empty", "stalled", "unknown_speed", "slow", "long_audio_step",
                       "long_stage", "swap", "cuda_pressure"):
            with self.subTest(reason=reason):
                admission, feedback, advance = self.aged_fixture()
                device = "cuda:0" if reason == "cuda_pressure" else None
                if reason == "empty":
                    feedback.checkpoint(0, 24000, "playing")
                elif reason == "stalled":
                    feedback.checkpoint(7200, 24000, "stalled")
                elif reason == "unknown_speed":
                    feedback._rtf = None
                elif reason == "slow":
                    feedback.observe(.16, 1.)
                elif reason == "long_audio_step":
                    feedback.observe(.5, .4)
                elif reason == "long_stage":
                    admission.observe("fixture", 1.)
                elif reason == "swap":
                    admission.paging(20.)
                else:
                    admission.device_pressure(device, True)
                with patch.object(admission._condition, "wait", side_effect=advance), \
                        admission.producing(feedback, device=device), background_scope(None):
                    _budget.get().charge(1.)
                    elapsed, forced = admission.wait("fixture", device=device)
                self.assertAlmostEqual(elapsed, .1)
                self.assertTrue(forced)
                self.assertEqual(admission.snapshot()["aged"], 0)

    def test_budget_follows_fact_worker_context_and_nested_scopes_share_it(self):
        from voicemem.lang import contextualize
        with background_scope(None):
            budget = _budget.get()
            budget.charge(.5)
            with background_scope(None):
                self.assertIs(_budget.get(), budget)
                worker = contextualize(lambda: _budget.get())
        self.assertIsNone(_budget.get())
        with ThreadPoolExecutor(max_workers=1) as pool:
            self.assertIs(pool.submit(worker).result(timeout=1), budget)
        self.assertEqual(budget.spent(), .5)

    def test_browser_inventory_decays_and_pause_preserves_it(self):
        now = [10.]
        feedback = SpeechFeedback(clock=lambda: now[0])
        feedback.checkpoint(24000, 24000, "playing")
        now[0] += .4
        self.assertAlmostEqual(feedback.snapshot()["buffer_s"], .6)
        feedback.checkpoint(12000, 24000, "paused")
        now[0] += 2
        self.assertEqual(feedback.snapshot()["buffer_s"], .5)
        feedback.checkpoint(0, 24000, "stalled")
        self.assertEqual(feedback.snapshot()["buffer_s"], 0)

    def test_invalid_browser_inventory_does_not_poison_state(self):
        feedback = SpeechFeedback()
        feedback.checkpoint(24000, 24000, "paused")
        for samples, rate in ((float("nan"), 24000), (-1, 24000), (1, 0), ("bad", 1)):
            feedback.checkpoint(samples, rate, "playing")
            self.assertEqual(feedback.snapshot()["buffer_s"], 1)

    def test_native_inventory_cannot_overwrite_reported_browser_inventory(self):
        feedback = SpeechFeedback()
        feedback.checkpoint(0, 24000, "stalled")
        feedback.observe(.16, .1)
        self.assertEqual(feedback.snapshot()["buffer_s"], 0)

    def test_each_segment_prefill_is_excluded_from_steady_speed(self):
        feedback = SpeechFeedback()
        feedback.observe(.16, 2, first=True)
        feedback.observe(.16, .08)
        feedback.observe(.16, 4, first=True)
        self.assertEqual(feedback.snapshot()["rtf"], .5)

    def test_no_foreground_speech_does_not_defer_background(self):
        admission = ResourceAdmission()
        self.assertFalse(admission.wait("fixture")[1])
        self.assertEqual(admission.snapshot()["deferred"], 0)

    def test_headroom_speed_and_fresh_swap_rates_control_admission(self):
        now = [20.]
        admission = ResourceAdmission(clock=lambda: now[0])
        feedback = SpeechFeedback(clock=lambda: now[0])
        with admission.producing(feedback):
            self.assertTrue(admission._blocked("fixture", now[0]))
            feedback.checkpoint(24000, 24000, "playing")
            self.assertFalse(admission._blocked("fixture", now[0]))
            admission.observe("fixture", .8)
            self.assertTrue(admission._blocked("fixture", now[0]))
            feedback.checkpoint(48000, 24000, "playing")
            self.assertFalse(admission._blocked("fixture", now[0]))
            admission.paging(20)
            self.assertTrue(admission._blocked("fixture", now[0]))
            now[0] += 4
            feedback.checkpoint(48000, 24000, "paused")
            self.assertFalse(admission._blocked("fixture", now[0]))

    def test_slow_native_production_blocks_even_with_inventory(self):
        admission, feedback = ResourceAdmission(), SpeechFeedback()
        feedback.observe(.16, 2)
        self.assertIsNone(feedback.snapshot()["rtf"])
        feedback.observe(.16, .32)
        feedback.checkpoint(240000, 24000, "playing")
        with admission.producing(feedback):
            self.assertTrue(admission._blocked("fixture", admission.clock()))

    def test_completion_wakes_waiter_without_leaking_producer(self):
        admission, feedback = ResourceAdmission(max_defer_s=1), SpeechFeedback()
        with ThreadPoolExecutor(max_workers=1) as pool:
            with admission.producing(feedback):
                pending = pool.submit(admission.wait, "fixture")
                deadline = time.monotonic()+1
                while not admission.deferred and time.monotonic() < deadline:
                    time.sleep(.001)
                self.assertFalse(pending.done())
            elapsed, forced = pending.result(timeout=1)
        self.assertFalse(forced)
        self.assertLess(elapsed, 1)
        self.assertEqual(admission.snapshot()["speech_jobs"], 0)

    def test_continuous_speech_has_bounded_background_progress(self):
        admission = ResourceAdmission(max_defer_s=.02)
        with admission.producing(SpeechFeedback()):
            elapsed, forced = admission.wait("fixture")
        self.assertTrue(forced)
        self.assertGreaterEqual(elapsed, .02)
        self.assertEqual(admission.snapshot()["forced"], 1)

    def test_nested_scope_and_failure_release_output_ownership(self):
        admission, feedback = ResourceAdmission(), SpeechFeedback()
        with patch("voicemem.engine.scheduling.admission.resources", admission):
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with speech_scope(feedback, producing=False):
                    self.assertEqual(admission.snapshot()["speech_jobs"], 0)
                    with speech_scope() as nested:
                        self.assertIs(nested, feedback)
                        with speech_scope():
                            self.assertEqual(admission.snapshot()["speech_jobs"], 1)
                            raise RuntimeError("fixture")
            self.assertIsNone(_speech.get())
            self.assertEqual(admission.snapshot()["speech_jobs"], 0)

    def test_output_reserves_between_segments_after_first_local_producer(self):
        admission, feedback = ResourceAdmission(), SpeechFeedback()
        with patch("voicemem.engine.scheduling.admission.resources", admission):
            with speech_scope(feedback, producing=False):
                self.assertEqual(admission.snapshot()["speech_jobs"], 0)
                with speech_scope():
                    pass
                self.assertEqual(admission.snapshot()["speech_jobs"], 1)
            self.assertEqual(admission.snapshot()["speech_jobs"], 0)

    def test_admission_precedes_model_lock_and_failure_keeps_lock_usable(self):
        lock = InferenceLock()
        model_metrics(reset=True)

        def admit(model):
            self.assertEqual(model, "fixture")
            self.assertFalse(lock._is_owned())
            return .25, True

        with patch("voicemem.engine.telemetry.metrics.resources.wait", side_effect=admit):
            with background_scope(None), self.assertRaisesRegex(ValueError, "fixture"):
                with inference_call("fixture", "torch", lock=lock):
                    state = lock.snapshot()
                    self.assertEqual((state["model"], state["service"]), ("fixture", "background"))
                    raise ValueError("fixture")
        self.assertFalse(lock._is_owned())
        row = next(r for r in model_metrics() if r["model"] == "fixture")
        self.assertEqual((row["errors"], row["forced"], row["admission_ms"]), (1, 1, 250))

    def test_audio_does_not_wait_for_background_admission(self):
        with patch("voicemem.engine.telemetry.metrics.resources.wait") as wait:
            with background_scope(None), inference_call("fixture", "torch", service=Service.AUDIO):
                pass
            wait.assert_not_called()

    def test_reentrant_native_call_never_waits_while_holding_outer_lock(self):
        lock = InferenceLock()
        with patch("voicemem.engine.telemetry.metrics.resources.wait") as wait:
            with background_scope(None), lock, inference_call("fixture", "torch", lock=lock):
                self.assertTrue(lock._is_owned())
            wait.assert_not_called()

    def test_lock_wait_is_separate_from_native_runtime(self):
        lock, acquired, release = InferenceLock(), threading.Event(), threading.Event()

        def holder():
            with lock:
                acquired.set()
                release.wait(1)

        thread = threading.Thread(target=holder)
        thread.start()
        self.assertTrue(acquired.wait(1))
        timer = threading.Timer(.03, release.set)
        timer.start()
        try:
            with inference_call("waiting_fixture", "torch", lock=lock):
                pass
        finally:
            release.set(); thread.join(1); timer.cancel()
        row = next(r for r in model_metrics() if r["model"] == "waiting_fixture")
        self.assertGreater(row["lock_ms"], 10)
        self.assertLess(row["run_ms"], row["lock_ms"])

    def test_swap_rates_use_counter_deltas_and_handle_resets(self):
        counters = dict(swap_in=1048576, swap_out=2097152, page_in=4194304,
                        page_out=8388608, compress=1048576, decompress=2097152, free_mib=4)
        self.assertEqual(counters["swap_in"], 1048576)
        self.assertEqual(paging_rates(None, counters, 1), {})
        rates = paging_rates({k: 0 for k in counters}, counters, 2)
        self.assertEqual(rates["swap_out_mib_s"], 1)
        self.assertEqual(rates["decompress_mib_s"], 1)
        self.assertEqual(counters["free_mib"], 4)
        self.assertNotIn("free_mib_mib_s", rates)
        self.assertEqual(paging_rates(counters, {k: 0 for k in counters}, 1)["swap_in_mib_s"], 0)

    def test_missing_os_sampling_does_not_suppress_model_observability(self):
        observer = EngineObserver()
        row = dict(model="fixture", backend="torch", service="audio", calls=1,
                   run_ms=1, max_run_ms=1, lock_ms=0, admission_ms=0,
                   queue_ms=0, forced=0, errors=0)
        with patch("voicemem.engine.telemetry.system._counters", side_effect=NotImplementedError), \
                patch("voicemem.engine.telemetry.system.time.monotonic", side_effect=[0, 6]), \
                patch("voicemem.engine.telemetry.system.model_metrics", return_value=[row]), \
                patch.object(observer._stop, "wait", side_effect=lambda _: observer._stop.set()), \
                patch("builtins.print") as output:
            observer._run()
        messages = [call.args[0] for call in output.call_args_list]
        self.assertTrue(any("sampling_unavailable" in message for message in messages))
        self.assertTrue(any("[engine-model] model=fixture" in message for message in messages))

    def test_mach_rev1_reader_uses_native_page_size_without_forking(self):
        def sample(host, flavor, pointer, count):
            self.assertEqual((host, flavor), (7, 4))
            stats = ctypes.cast(pointer, ctypes.POINTER(VMStatistics64)).contents
            stats.swapins, stats.swapouts, stats.free_count = 64, 128, 256
            return 0

        from unittest.mock import Mock
        library = SimpleNamespace(mach_host_self=Mock(return_value=7),
                                  host_statistics64=Mock(side_effect=sample))
        with patch("voicemem.engine.telemetry.darwin.os.sysconf", return_value=16384, create=True):
            reader = DarwinCounters(library)
        result = reader.sample()
        self.assertEqual(ctypes.sizeof(VMStatistics64), 152)
        self.assertEqual((result["swap_in"], result["swap_out"], result["free_mib"]),
                         (1048576, 2097152, 4))
        library.host_statistics64.return_value = 1
        library.host_statistics64.side_effect = None
        with self.assertRaises(OSError):
            reader.sample()
