"""Owner-thread CUDA speech execution with device admission and bounded delivery."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import queue
import threading
import time

from ..scheduling.admission import observe_speech, speech_scope
from ..scheduling.context import Service, inference_scope
from .devices import device_key
from ..telemetry.metrics import inference_call
from .torch import cuda_initialization, device_lock


class CudaWorker:
    """Serialize one mutable speech model on its native owner, not the async loop.

    Factories return PCM16 iterators and are deferred until device admission.
    Each iterator step must materialize host PCM before returning. The worker
    releases device admission before waiting for bounded consumer credits.
    Cancellation skips queued factories and retains native ownership through
    generator cleanup; running native calls cannot be preempted.
    """

    def __init__(self, device, *, model="tts", sample_rate=24000, max_buffer=4):
        if sample_rate <= 0 or max_buffer < 1:
            raise ValueError("sample_rate and max_buffer must be positive")
        self.device = device_key(device)
        if not self.device or not self.device.startswith("cuda:"):
            raise ValueError("CudaWorker requires an explicit CUDA device")
        self.model, self.sample_rate, self.max_buffer = model, sample_rate, max_buffer
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-cuda")
        self._requests = {}
        self._closed = self._initialized = False

    async def stream(self, factory, *, initialize=None):
        """Yield ordered host PCM; propagate native errors and close on cancellation.

        Optional initialization runs once successfully on this owner, excluding
        other registered CUDA work while graphs are captured. Closing this async
        iterator cancels delivery; ``aclose`` also drains all native cleanup.
        """
        if self._closed:
            raise RuntimeError("CUDA worker is closed")
        loop = asyncio.get_running_loop()
        cancelled, ready = threading.Event(), asyncio.Event()
        output = queue.Queue(maxsize=self.max_buffer)
        submitted = time.perf_counter()

        def publish(kind, value=None, *, final=False):
            while final or not cancelled.is_set():
                try:
                    output.put((kind, value), timeout=.05)
                    try:
                        loop.call_soon_threadsafe(ready.set)
                    except RuntimeError:
                        cancelled.set()
                    return
                except queue.Full:
                    if final and cancelled.is_set():
                        try:
                            output.get_nowait()
                        except queue.Empty:
                            pass

        def work():
            chunks = None
            count = audio_bytes = 0
            compute = maximum = 0.
            first = None
            status = "cancelled"
            queue_wait = time.perf_counter()-submitted
            queue_s = queue_wait
            lock = device_lock(self.device)
            try:
                if cancelled.is_set():
                    return
                with inference_scope(Service.AUDIO), speech_scope(device=self.device):
                    try:
                        if initialize is not None and not self._initialized:
                            with cuda_initialization():
                                if cancelled.is_set():
                                    return
                                with inference_call(self.model+"_load", "cuda", lock=lock):
                                    initialize()
                                    self._initialized = True
                        while not cancelled.is_set():
                            with inference_call(self.model, "cuda", lock=lock, queue_s=queue_s):
                                queue_s = 0.
                                if cancelled.is_set():
                                    break
                                started = time.perf_counter()
                                if chunks is None:
                                    chunks = iter(factory(cancelled))
                                try:
                                    pcm = next(chunks)
                                except StopIteration:
                                    status = "complete"
                                    break
                                elapsed = time.perf_counter()-started
                            if not pcm:
                                continue
                            if len(pcm) % 2:
                                raise ValueError("CUDA speech must return sample-aligned PCM16")
                            count += 1
                            audio_bytes += len(pcm)
                            compute += elapsed
                            maximum = max(maximum, elapsed)
                            if first is None:
                                first = time.perf_counter()-submitted
                            observe_speech(len(pcm)/(self.sample_rate*2), elapsed, first=count == 1)
                            publish("pcm", pcm)
                    finally:
                        if chunks is not None and hasattr(chunks, "close"):
                            with inference_call(self.model+"_cleanup", "cuda", lock=lock):
                                chunks.close()
            except BaseException as exc:
                status = "error"
                if cancelled.is_set():
                    print(f"[engine-cuda] model={self.model} device={self.device}"
                          f" cleanup_error={type(exc).__name__}", flush=True)
                publish("error", exc)
            finally:
                if count:
                    print(f"[engine-cuda] model={self.model} device={self.device}"
                          f" queue_ms={queue_wait*1000:.0f}"
                          f" first_pcm_ms={(first or 0)*1000:.0f} compute_ms={compute*1000:.0f}"
                          f" max_step_ms={maximum*1000:.0f} audio_ms={audio_bytes/(self.sample_rate*2)*1000:.0f}"
                          f" chunks={count} status={status}", flush=True)
                publish("done", final=True)

        future = self._worker.submit(copy_context().run, work)
        self._requests[cancelled] = future

        def finished(done):
            # Observe unexpected executor failures even after a consumer detaches.
            error = done.exception()
            if error is not None:
                print(f"[engine-cuda] worker_error={type(error).__name__}", flush=True)
            try:
                loop.call_soon_threadsafe(self._requests.pop, cancelled, None)
            except RuntimeError:
                pass

        future.add_done_callback(finished)
        try:
            while True:
                ready.clear()
                try:
                    kind, value = output.get_nowait()
                except queue.Empty:
                    await ready.wait()
                    continue
                if kind == "done":
                    return
                if kind == "error":
                    raise value
                yield value
        finally:
            cancelled.set()

    async def aclose(self):
        """Cancel queued delivery and drain owner-thread inference and cleanup."""
        self._closed = True
        for cancelled in tuple(self._requests):
            cancelled.set()
        await asyncio.to_thread(self._worker.shutdown, wait=True)
