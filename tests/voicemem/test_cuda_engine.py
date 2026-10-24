"""Hardware-independent CUDA admission, graph exclusion and native lifetimes."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
from contextvars import ContextVar
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from voicemem.engine import (CudaWorker, Service, SpeechFeedback, cuda_initialization,
                             device_lock, inference_call, model_metrics, speech_scope,
                             torch_snapshot)
from voicemem.engine.scheduling.admission import ResourceAdmission, _speech
from voicemem.engine.scheduling.context import _service
from voicemem.engine.backends.torch import InferenceLock, _capture_gate
from voicemem.leftbrain.local_embedder import _LockedEncoder


class CudaAdmissionTests(unittest.TestCase):
    def test_same_device_aliases_share_one_lock_and_other_devices_progress(self):
        with patch.dict("sys.modules", {"torch": None}):
            first, second = device_lock("cuda"), device_lock("cuda:1")
        self.assertIs(first, device_lock("cuda:00"))
        self.assertIsNot(first, second)
        def acquire(lock):
            acquired = lock.acquire(timeout=.02)
            if acquired:
                lock.release()
            return acquired
        with ThreadPoolExecutor(max_workers=2) as pool, first:
            self.assertFalse(pool.submit(acquire, first).result(timeout=1))
            self.assertTrue(pool.submit(acquire, second).result(timeout=1))

    def test_capture_waits_for_native_work_and_excludes_all_cuda_devices(self):
        entered, release = threading.Event(), threading.Event()
        first, second = device_lock("cuda:0"), device_lock("cuda:1")
        def capture():
            with cuda_initialization():
                entered.set()
                self.assertTrue(release.wait(2))
        with ThreadPoolExecutor(max_workers=1) as pool:
            with first:
                task = pool.submit(capture)
                deadline = time.monotonic()+1
                while not _capture_gate.waiting and time.monotonic() < deadline:
                    time.sleep(.001)
                self.assertFalse(entered.is_set())
            try:
                self.assertTrue(entered.wait(1))
                self.assertFalse(second.acquire(False))
                self.assertFalse(first.acquire(timeout=.01))
            finally:
                release.set()
            task.result(timeout=1)
        self.assertTrue(second.acquire(False))
        second.release()

    def test_capture_failure_releases_gate_and_read_to_write_upgrade_is_rejected(self):
        lock = device_lock("cuda:0")
        with self.assertRaisesRegex(ValueError, "fixture"):
            with cuda_initialization(), lock:
                raise ValueError("fixture")
        with lock:
            with self.assertRaisesRegex(RuntimeError, "precede"):
                with cuda_initialization():
                    pass
        self.assertEqual(_capture_gate.readers, {})

    def test_cuda_memory_is_sampled_on_owner_and_snapshot_uses_cached_values(self):
        cuda = SimpleNamespace(mem_get_info=Mock(return_value=(100, 10000)),
                               memory_allocated=Mock(return_value=300),
                               memory_reserved=Mock(return_value=700),
                               max_memory_allocated=Mock(return_value=500))
        lock = InferenceLock(device="cuda:2")
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
            with lock:
                pass
            row = lock.snapshot()
            self.assertEqual((row["allocated_bytes"], row["reserved_bytes"], row["free_bytes"]),
                             (300, 700, 100))
            for _ in range(3):
                lock.snapshot()
                torch_snapshot()
            cuda.mem_get_info.assert_called_once_with("cuda:2")

    def test_cuda_sampling_failure_cannot_orphan_device_admission(self):
        cuda = SimpleNamespace(mem_get_info=Mock(side_effect=RuntimeError("fixture")))
        lock = device_lock("cuda:3")
        lock._memory_sampled = 0
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}), lock:
            pass
        self.assertTrue(lock.snapshot()["sampling_unavailable"])
        self.assertTrue(lock.acquire(False))
        lock.release()
        self.assertEqual(_capture_gate.readers, {})

    def test_speech_and_low_memory_admission_are_device_local(self):
        now = [10.]
        admission = ResourceAdmission(clock=lambda: now[0])
        feedback = SpeechFeedback(clock=lambda: now[0])
        with admission.producing(feedback, device="cuda:0"):
            self.assertTrue(admission._blocked("fixture", now[0], "cuda:0"))
            self.assertFalse(admission._blocked("fixture", now[0], "cuda:1"))
            self.assertTrue(admission._blocked("fixture", now[0]))
            feedback.checkpoint(24000, 24000, "playing")
            self.assertFalse(admission._blocked("fixture", now[0], "cuda:0"))
            admission.device_pressure("cuda:0", True)
            self.assertTrue(admission._blocked("fixture", now[0], "cuda:0"))
            self.assertFalse(admission._blocked("fixture", now[0], "cuda:1"))
            now[0] += 4
            feedback.checkpoint(24000, 24000, "playing")
            self.assertFalse(admission._blocked("fixture", now[0], "cuda:0"))
        self.assertFalse(admission._devices)

    def test_cost_and_model_metrics_do_not_mix_distinct_devices(self):
        now = [10.]
        admission = ResourceAdmission(clock=lambda: now[0])
        feedback = SpeechFeedback(clock=lambda: now[0])
        feedback.checkpoint(24000, 24000, "playing")
        admission.observe("fixture", 2., device="cuda:1")
        with admission.producing(feedback, device="cuda:0"):
            self.assertFalse(admission._blocked("fixture", now[0], "cuda:0"))
        model_metrics(reset=True)
        for device in ("cuda:0", "cuda:1"):
            with inference_call("device_fixture", "cuda", lock=device_lock(device)):
                pass
        rows = [r for r in model_metrics() if r["model"] == "device_fixture"]
        self.assertEqual({r["device"] for r in rows}, {"cuda:0", "cuda:1"})

    def test_cuda_alias_captures_current_device_before_crossing_threads(self):
        cuda = SimpleNamespace(is_initialized=lambda: True, current_device=lambda: 1)
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
            self.assertIs(device_lock("cuda"), device_lock("cuda:1"))

    def test_mutable_encoder_placement_is_resolved_under_model_ownership(self):
        entered = threading.Event()
        model = SimpleNamespace(device="cuda:0")
        def encode(*args, **kwargs):
            self.assertTrue(device_lock(kwargs.get("device") or model.device)._is_owned())
            return "encoded"
        model.encode = encode
        encoder = _LockedEncoder(model)
        def run():
            entered.set()
            return encoder.encode(["fixture"])
        with ThreadPoolExecutor(max_workers=1) as pool:
            with encoder._lock:
                task = pool.submit(run)
                self.assertTrue(entered.wait(1))
                model.device = "cuda:1"
            self.assertEqual(task.result(timeout=1), "encoded")
        self.assertEqual(encoder.encode(["fixture"], device="cuda:0"), "encoded")

    def test_failed_device_resolution_releases_model_mutex(self):
        lock = threading.RLock()
        def resolve():
            raise ValueError("placement fixture")
        with self.assertRaisesRegex(ValueError, "placement fixture"):
            with inference_call("fixture", "torch", lock=(lock, resolve)):
                pass
        self.assertFalse(lock._is_owned())


class CudaWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.worker = CudaWorker("cuda:0", max_buffer=2)
        self.addAsyncCleanup(self.worker.aclose)

    async def test_initialization_retries_on_owner_and_native_failure_is_visible(self):
        attempts, threads = [], []
        def initialize():
            threads.append(threading.get_ident())
            self.assertEqual(_capture_gate.owner, threads[-1])
            attempts.append(True)
            if len(attempts) == 1:
                raise ValueError("initialization fixture")
        def generate(cancelled):
            threads.append(threading.get_ident())
            self.assertTrue(device_lock("cuda:0")._is_owned())
            yield b"\x00\x00"
            raise RuntimeError("generation fixture")
        with self.assertRaisesRegex(ValueError, "initialization fixture"):
            async for _ in self.worker.stream(generate, initialize=initialize):
                pass
        received = []
        with self.assertRaisesRegex(RuntimeError, "generation fixture"):
            async for pcm in self.worker.stream(generate, initialize=initialize):
                received.append(pcm)
        self.assertEqual(received, [b"\x00\x00"])
        self.assertEqual(len(set(threads)), 1)
        self.assertNotEqual(threads[0], threading.get_ident())
        self.assertTrue(self.worker._initialized)

    async def test_context_and_output_ownership_survive_cancelled_consumer(self):
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        marker = ContextVar("fixture_marker", default=None)
        feedback, admission = SpeechFeedback(), ResourceAdmission()
        def generate(cancelled):
            self.assertEqual(marker.get(), "captured")
            self.assertIs(_speech.get(), feedback)
            self.assertEqual(_service.get(), Service.AUDIO)
            try:
                entered.set()
                release.wait(2)
                yield b"\x00\x00"
            finally:
                self.assertTrue(device_lock("cuda:0")._is_owned())
                closed.set()
        with patch("voicemem.engine.scheduling.admission.resources", admission):
            with speech_scope(feedback, producing=False):
                token = marker.set("captured")
                stream = self.worker.stream(generate)
                task = asyncio.create_task(anext(stream))
                marker.reset(token)
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    await stream.aclose()
                    self.assertEqual(admission.snapshot()["speech_jobs"], 1)
                finally:
                    release.set()
                    self.assertTrue(await asyncio.to_thread(closed.wait, 1))
                    await self.worker.aclose()
            self.assertEqual(admission.snapshot()["speech_jobs"], 0)

    async def test_slow_consumer_keeps_pcm_bounded_and_releases_device(self):
        produced, full = [], threading.Event()
        def generate(cancelled):
            for i in range(100):
                produced.append(i)
                if len(produced) == 4:
                    full.set()
                yield b"\x00\x00"
        async with aclosing(self.worker.stream(generate)) as stream:
            await anext(stream)
            self.assertTrue(await asyncio.to_thread(full.wait, 1))
            self.assertLessEqual(len(produced), 4)
            def other_model():
                lock = device_lock("cuda:0")
                acquired = lock.acquire(timeout=.2)
                if acquired:
                    lock.release()
                return acquired
            self.assertTrue(await asyncio.to_thread(other_model))

    async def test_cancelled_queued_factory_never_starts(self):
        entered, release = threading.Event(), threading.Event()
        second = Mock(return_value=iter([b"\x00\x00"]))
        def first(cancelled):
            entered.set()
            release.wait(2)
            yield b"\x00\x00"
        stream1, stream2 = self.worker.stream(first), self.worker.stream(second)
        task1 = asyncio.create_task(anext(stream1))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            task2 = asyncio.create_task(anext(stream2))
            await asyncio.sleep(0)
            task2.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task2
        finally:
            release.set()
            await asyncio.wait_for(task1, 1)
            await stream1.aclose()
            await stream2.aclose()
            await self.worker.aclose()
        second.assert_not_called()

    async def test_waiting_asr_runs_between_speech_steps(self):
        entered, release = threading.Event(), threading.Event()
        order = []
        lock = device_lock("cuda:0")
        def generate(cancelled):
            order.append("speech_first")
            entered.set()
            release.wait(2)
            yield b"\x00\x00"
            order.append("speech_second")
            yield b"\x00\x00"
        def recognize():
            with inference_call("streaming_asr", "torch", lock=lock, service=Service.AUDIO):
                order.append("asr")
        async def consume():
            return [pcm async for pcm in self.worker.stream(generate)]
        task = asyncio.create_task(consume())
        self.assertTrue(await asyncio.to_thread(entered.wait, 1))
        asr = asyncio.create_task(asyncio.to_thread(recognize))
        try:
            for _ in range(100):
                if lock.snapshot()["waiting"]:
                    break
                await asyncio.sleep(.001)
            self.assertTrue(lock.snapshot()["waiting"])
        finally:
            release.set()
        self.assertEqual(len(await asyncio.wait_for(task, 1)), 2)
        await asyncio.wait_for(asr, 1)
        self.assertEqual(order, ["speech_first", "asr", "speech_second"])

    async def test_delivery_does_not_occupy_default_executor_and_preserves_order(self):
        def generate(cancelled):
            yield b"\x01\x00"
            yield b"\x02\x00"
        with patch("voicemem.engine.backends.cuda.asyncio.to_thread", side_effect=AssertionError("blocking bridge")):
            chunks = [pcm async for pcm in self.worker.stream(generate)]
        self.assertEqual(chunks, [b"\x01\x00", b"\x02\x00"])

    async def test_cleanup_error_is_visible_and_next_job_can_run(self):
        def generate(cancelled):
            try:
                yield b"\x00\x00"
            finally:
                raise ValueError("cleanup fixture")
        with self.assertRaisesRegex(ValueError, "cleanup fixture"):
            async for _ in self.worker.stream(generate):
                pass
        self.assertEqual([p async for p in self.worker.stream(lambda _: iter([b"\x01\x00"]))],
                         [b"\x01\x00"])

    async def test_closed_worker_rejects_new_jobs(self):
        await self.worker.aclose()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await anext(self.worker.stream(lambda _: iter([])))
