"""Serial EOT inference, event-loop responsiveness and obsolete result guards."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
import unittest

import numpy as np

from voicemem.stream import VoiceStream, _EOT_EXECUTOR


class EotSchedulingTests(unittest.IsolatedAsyncioTestCase):
    def stream(self, score):
        stream = VoiceStream(SimpleNamespace(), src_rate=16000, confirm_s=999,
                             spec_min_chars=10000, gamble_s=999,
                             eot=SimpleNamespace(score=score, threshold=.5))
        stream._asr_w = SimpleNamespace(push=lambda _: "synthetic speech",
                                       reset=lambda: None, close=lambda: None,
                                       t=SimpleNamespace(join=lambda _: None))
        stream._vad = SimpleNamespace(is_speech=lambda _: True)
        stream._eot_wait = .02
        return stream

    async def feed(self, stream):
        return await stream.feed(np.zeros(320, np.int16).tobytes())

    async def test_inference_leaves_event_loop_responsive_with_default_pool_busy(self):
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        busy, entered = asyncio.Event(), asyncio.Event()
        release_pool, release_eot = threading.Event(), threading.Event()
        caller_thread = threading.get_ident()

        def occupy():
            loop.call_soon_threadsafe(busy.set)
            release_pool.wait(2)

        def score(_):
            self.assertNotEqual(threading.get_ident(), caller_thread)
            loop.call_soon_threadsafe(entered.set)
            release_eot.wait(2)
            return .1

        pool_job = asyncio.create_task(asyncio.to_thread(occupy))
        stream = self.stream(score)
        task = None
        try:
            await asyncio.wait_for(busy.wait(), 1)
            task = asyncio.create_task(self.feed(stream))
            await asyncio.wait_for(entered.wait(), 1)
            tick = asyncio.Event()
            loop.call_soon(tick.set)
            await asyncio.wait_for(tick.wait(), 1)
            self.assertFalse(task.done())
            release_eot.set()
            self.assertFalse((await asyncio.wait_for(task, 1)).turn)
        finally:
            release_pool.set()
            release_eot.set()
            await pool_job
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            await stream.aclose()

    async def test_close_and_reset_discard_native_results_and_errors(self):
        for action in ("close", "reset"):
            for failure in (False, True):
                with self.subTest(action=action, failure=failure):
                    loop = asyncio.get_running_loop()
                    entered, release = asyncio.Event(), threading.Event()

                    def score(_):
                        loop.call_soon_threadsafe(entered.set)
                        release.wait(2)
                        if failure:
                            raise RuntimeError("synthetic EOT failure")
                        return .99

                    stream = self.stream(score)
                    detector = stream.eot
                    task = asyncio.create_task(self.feed(stream))
                    try:
                        await asyncio.wait_for(entered.wait(), 1)
                        if action == "close":
                            await asyncio.wait_for(stream.aclose(), 1)
                        else:
                            stream._reset_turn()
                        release.set()
                        state = await asyncio.wait_for(task, 1)
                        self.assertIsNone(state.turn)
                        self.assertEqual(stream._eot_score, 0)
                        self.assertIs(stream.eot, detector)
                    finally:
                        release.set()
                        await asyncio.gather(task, return_exceptions=True)
                        await stream.aclose()

    async def test_cancelling_a_queued_probe_never_enters_the_provider(self):
        loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), threading.Event()
        calls = []

        def score(_):
            loop.call_soon_threadsafe(entered.set)
            release.wait(2)
            return .1

        first = self.stream(score)
        second = self.stream(lambda _: calls.append(True) or .1)
        active = asyncio.create_task(self.feed(first))
        queued = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            queued = asyncio.create_task(self.feed(second))
            await asyncio.sleep(0)
            queued.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await queued
            release.set()
            await asyncio.wait_for(active, 1)
            await loop.run_in_executor(_EOT_EXECUTOR, lambda: None)
            self.assertEqual(calls, [])
        finally:
            release.set()
            await asyncio.gather(active, *([queued] if queued else []), return_exceptions=True)
            await first.aclose()
            await second.aclose()

    async def test_current_failure_disables_eot_and_preserves_timeout_fallback(self):
        def fail(_):
            raise RuntimeError("synthetic EOT failure")
        stream = self.stream(fail)
        try:
            state = await self.feed(stream)
            self.assertIsNone(state.turn)
            self.assertIsNone(stream.eot)
        finally:
            await stream.aclose()


if __name__ == "__main__":
    unittest.main()
