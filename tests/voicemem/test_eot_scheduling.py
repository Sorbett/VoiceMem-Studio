"""EOT probes cannot block capture or confirm uncovered or obsolete speech."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

import numpy as np

from tests.helpers.probes import ManualExecutor
from voicemem.engine import LatestProbe
from voicemem.stream import Turn, VoiceStream, _EOT_EXECUTOR


class EotSchedulingTests(unittest.IsolatedAsyncioTestCase):
    def stream(self, score, *, manual=False):
        stream = VoiceStream(SimpleNamespace(), src_rate=16000, confirm_s=999,
                             spec_min_chars=10000, gamble_s=999,
                             gate=lambda _: "backchannel",
                             eot=SimpleNamespace(score=score, threshold=.5))
        stream.frames = []

        def push(frame):
            stream.frames.append(frame.copy())
            return "synthetic speech?"

        stream._asr_w = SimpleNamespace(push=push, reset=lambda: None, close=lambda: None,
                                       t=SimpleNamespace(join=lambda _: None))
        stream._vad = SimpleNamespace(is_speech=lambda _: True)
        stream._eot_wait = .02
        stream._finish_asr = AsyncMock()
        stream._confirm = AsyncMock(return_value=Turn("synthetic speech?", None))
        if manual:
            stream.executor = ManualExecutor()
            stream._eot_probe = LatestProbe(stream.executor, clock=lambda: 0)
        self.addAsyncCleanup(stream.aclose)
        return stream

    async def feed(self, stream, value=0):
        return await stream.feed(np.full(320, value, np.int16).tobytes())

    async def drain(self):
        await asyncio.get_running_loop().run_in_executor(_EOT_EXECUTOR, lambda: None)

    async def test_capture_advances_with_default_pool_busy_and_native_eot_blocked(self):
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        busy, entered = asyncio.Event(), asyncio.Event()
        release_pool, release_eot = threading.Event(), threading.Event()
        caller_thread = threading.get_ident()
        snapshots = []

        def occupy():
            loop.call_soon_threadsafe(busy.set)
            release_pool.wait(3)

        def score(pcm):
            self.assertNotEqual(threading.get_ident(), caller_thread)
            snapshots.append(pcm)
            loop.call_soon_threadsafe(entered.set)
            release_eot.wait(3)
            return .1

        pool_job = asyncio.create_task(asyncio.to_thread(occupy))
        stream = self.stream(score)
        try:
            await asyncio.wait_for(busy.wait(), 1)
            state = await asyncio.wait_for(self.feed(stream, 1), .5)
            await asyncio.wait_for(entered.wait(), 1)
            self.assertIsNone(state.turn)
            for value in range(2, 52):
                state = await asyncio.wait_for(self.feed(stream, value), .5)
                self.assertIsNone(state.turn)
            self.assertFalse(release_eot.is_set())
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(stream._eot_probe.snapshot()["submitted"], 1)
            self.assertEqual(len(stream.frames), 51)
            for value, frame in enumerate(stream.frames, 1):
                np.testing.assert_array_equal(frame, np.full(320, value / 32768, np.float32))
            np.testing.assert_array_equal(snapshots[0], stream.frames[0])
            self.assertEqual(sum(map(len, stream._pcm)), 51 * 320)
        finally:
            release_pool.set()
            release_eot.set()
            await pool_job
            await self.drain()

    async def test_close_and_reset_discard_native_results_and_errors(self):
        for action in ("close", "reset"):
            for failure in (False, True):
                with self.subTest(action=action, failure=failure):
                    loop = asyncio.get_running_loop()
                    entered, release = asyncio.Event(), threading.Event()

                    def score(_):
                        loop.call_soon_threadsafe(entered.set)
                        release.wait(3)
                        if failure:
                            raise RuntimeError("synthetic EOT failure")
                        return .99

                    stream = self.stream(score)
                    detector = stream.eot
                    try:
                        await self.feed(stream)
                        await asyncio.wait_for(entered.wait(), 1)
                        if action == "close":
                            await asyncio.wait_for(stream.aclose(), 1)
                        else:
                            stream._reset_turn()
                        self.assertFalse(stream._eot_probe.ready)
                        release.set()
                        await self.drain()
                        self.assertIsNone(stream._eot_probe.poll())
                        self.assertEqual(stream._eot_score, 0)
                        self.assertIs(stream.eot, detector)
                        self.assertEqual(stream._eot_probe.snapshot()["discarded"], 1)
                    finally:
                        release.set()
                        await self.drain()

    async def test_reset_or_close_queued_probe_never_enters_the_provider(self):
        for action in ("reset", "close"):
            with self.subTest(action=action):
                loop = asyncio.get_running_loop()
                entered, release = asyncio.Event(), threading.Event()
                calls = []

                def score(_):
                    loop.call_soon_threadsafe(entered.set)
                    release.wait(3)
                    return .1

                first = self.stream(score)
                second = self.stream(lambda _: calls.append(True) or .1)
                try:
                    await self.feed(first)
                    await asyncio.wait_for(entered.wait(), 1)
                    await self.feed(second)
                    if action == "reset":
                        second._reset_turn()
                    else:
                        await second.aclose()
                    release.set()
                    await self.drain()
                    self.assertEqual(calls, [])
                finally:
                    release.set()
                    await self.drain()

    async def test_current_failure_disables_eot_and_preserves_timeout_fallback(self):
        def fail(_):
            raise RuntimeError("synthetic EOT failure")
        stream = self.stream(fail, manual=True)
        await self.feed(stream)
        stream.executor.run_next()
        self.assertIsNone((await self.feed(stream)).turn)
        self.assertIsNone(stream.eot)
        stream.confirm_s = .04
        stream._vad.is_speech = lambda _: False
        self.assertIsNone((await self.feed(stream)).turn)
        self.assertIsNotNone((await self.feed(stream)).turn)

    async def test_speech_score_cannot_confirm_without_a_silence_snapshot(self):
        stream = self.stream(lambda _: .99, manual=True)
        await self.feed(stream)
        stream.executor.run_next()
        stream._vad.is_speech = lambda _: False
        self.assertIsNone((await self.feed(stream)).turn)
        self.assertGreater(stream._eot_score, 0)
        self.assertIsNone((await self.feed(stream)).turn)
        stream.executor.run_next()
        self.assertIsNotNone((await self.feed(stream)).turn)

    async def test_prefix_score_cannot_confirm_speech_missing_from_its_snapshot(self):
        stream = self.stream(lambda _: .99, manual=True)
        await self.feed(stream)
        stream.executor.run_next()
        self.assertIsNone((await self.feed(stream)).turn)
        stream._vad.is_speech = lambda _: False
        self.assertIsNone((await self.feed(stream)).turn)
        stream.executor.run_next()
        self.assertIsNotNone((await self.feed(stream)).turn)

    async def test_current_silence_score_preserves_minimum_silence_and_pause_guard(self):
        stream = self.stream(lambda _: .99, manual=True)
        stream.eot_min_s = .08
        stream.turn_end_guard = lambda *_: False
        await self.feed(stream)
        stream.executor.run_next()
        stream._vad.is_speech = lambda _: False
        await self.feed(stream)
        await self.feed(stream)
        stream.executor.run_next()
        self.assertIsNone((await self.feed(stream)).turn)
        self.assertIsNone((await self.feed(stream)).turn)
        stream.executor.run_next()
        stream.turn_end_guard = None
        self.assertIsNotNone((await self.feed(stream)).turn)

    async def test_private_speculation_score_does_not_override_disabled_eot_confirmation(self):
        stream = self.stream(lambda _: .99, manual=True)
        stream.eot_ends_turn = False
        await self.feed(stream)
        stream.executor.run_next()
        stream._vad.is_speech = lambda _: False
        await self.feed(stream)
        await self.feed(stream)
        stream.executor.run_next()
        state = await self.feed(stream)
        self.assertGreater(state.eot_score, 0)
        self.assertIsNone(state.turn)

    async def test_speech_resumption_invalidates_a_completed_pause_result(self):
        stream = self.stream(lambda _: .99, manual=True)
        await self.feed(stream)
        stream.executor.run_next()
        stream._vad.is_speech = lambda _: False
        await self.feed(stream)
        await self.feed(stream)
        stream.executor.run_next()
        stream._vad.is_speech = lambda _: True
        state = await self.feed(stream)
        self.assertIsNone(state.turn)
        self.assertEqual(stream._eot_score, 0)
        self.assertEqual(stream._eot_probe.snapshot()["discarded"], 1)
        stream._vad.is_speech = lambda _: False
        self.assertIsNone((await self.feed(stream)).turn)
        stream.executor.run_next()
        self.assertIsNotNone((await self.feed(stream)).turn)

    async def test_detector_replacement_rejects_old_result_or_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                def score(_):
                    if failure:
                        raise RuntimeError("synthetic obsolete failure")
                    return .99
                stream = self.stream(score, manual=True)
                await self.feed(stream)
                stream.executor.run_next()
                detector = stream.eot = SimpleNamespace(score=lambda _: .1, threshold=.5)
                stream._vad.is_speech = lambda _: False
                self.assertIsNone((await self.feed(stream)).turn)
                self.assertEqual(stream._eot_score, 0)
                self.assertIs(stream.eot, detector)

    async def test_cached_result_cannot_end_a_turn_after_provider_replacement(self):
        stream = self.stream(lambda _: .99, manual=True)
        stream.turn_end_guard = lambda *_: False
        await self.feed(stream)
        stream.executor.run_next()
        stream._vad.is_speech = lambda _: False
        await self.feed(stream)
        await self.feed(stream)
        stream.executor.run_next()
        self.assertIsNone((await self.feed(stream)).turn)
        self.assertGreater(stream._eot_score, 0)
        stream.eot = SimpleNamespace(score=lambda _: .1, threshold=.5)
        stream.turn_end_guard = None
        self.assertIsNone((await self.feed(stream)).turn)
        self.assertEqual(stream._eot_score, 0)


if __name__ == "__main__":
    unittest.main()
