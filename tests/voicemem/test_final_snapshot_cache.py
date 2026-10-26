"""Only successful final recognition of exactly matching audio may be reused."""
import asyncio
import threading
from types import SimpleNamespace
import unittest

import numpy as np

from voicemem.stream import VoiceStream


class FinalSnapshotCacheTests(unittest.IsolatedAsyncioTestCase):
    def test_invalid_preference_window_is_rejected_before_starting_workers(self):
        for value in (-.01, float('nan'), float('inf'), 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                VoiceStream(SimpleNamespace(), final_asr_grace_s=value)

    async def test_short_preference_accepts_ready_flush_without_cancelling_native_work(self):
        stream = self.make_stream()
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def transcribe(_):
            entered.set()
            try:
                release.wait(2)
                return 'late full transcript'
            finally:
                finished.set()

        stream._final_asr.transcribe = transcribe
        stream.final_asr_grace_s = .03

        async def flush():
            return 'complete streaming transcript'

        stream._asr_w.flush = flush
        task = asyncio.create_task(stream._finish_asr(stream._pcm[0]))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            await asyncio.wait_for(task, .2)
            self.assertEqual(stream._text, 'complete streaming transcript')
            self.assertFalse(finished.is_set())
        finally:
            release.set()
            await asyncio.to_thread(finished.wait, 1)

    def make_stream(self):
        self.calls = self.flushes = self.resets = 0

        def transcribe(pcm):
            self.calls += 1
            return f'complete-{self.calls}'

        async def flush():
            self.flushes += 1
            return ''

        def reset():
            self.resets += 1

        stream = VoiceStream(SimpleNamespace(), gate=lambda _: 'shallow')
        stream._final_asr = SimpleNamespace(transcribe=transcribe)
        stream._asr_w = SimpleNamespace(flush=flush, reset=reset)
        stream._text = 'partial'
        stream._pcm = [np.arange(160, dtype=np.float32)]
        return stream

    async def test_completed_exact_snapshot_avoids_second_decode_and_flush(self):
        stream = self.make_stream()
        self.assertEqual(await stream.refine_current_snapshot(), 'complete-1')
        timing = {}
        await stream._finish_asr(stream._pcm[0].copy(), timing=timing)
        self.assertEqual((stream._text, self.calls, self.flushes), ('complete-1', 1, 0))
        self.assertEqual(self.resets, 1)
        self.assertEqual(timing['snapshot_reused'], 1)

    async def test_changed_samples_or_extra_silence_require_complete_decode(self):
        for append_silence in (False, True):
            with self.subTest(append_silence=append_silence):
                stream = self.make_stream()
                await stream.refine_current_snapshot()
                pcm = stream._pcm[0].copy()
                if append_silence:
                    pcm = np.concatenate((pcm, np.zeros(16, dtype=pcm.dtype)))
                else:
                    pcm[-1] += 1
                timing = {}
                await stream._finish_asr(pcm, timing=timing)
                self.assertEqual((stream._text, self.calls), ('complete-2', 2))
                self.assertEqual(timing['snapshot_reused'], 0)

    async def test_turn_reset_and_provider_replacement_reject_cached_text(self):
        stream = self.make_stream()
        pcm = stream._pcm[0].copy()
        await stream.refine_current_snapshot()
        stream._reset_turn()
        await stream._finish_asr(pcm)
        self.assertEqual((stream._text, self.calls), ('complete-2', 2))
        stream._pcm = [pcm]
        await stream.refine_current_snapshot()
        stream._final_asr = SimpleNamespace(transcribe=lambda _: 'replacement-provider')
        await stream._finish_asr(pcm)
        self.assertEqual(stream._text, 'replacement-provider')

    async def test_frozen_partial_fallback_is_not_a_complete_audio_result(self):
        stream = self.make_stream()
        stream._final_asr = SimpleNamespace(transcribe=lambda _: '')
        self.assertEqual(await stream.refine_current_snapshot(), 'partial')
        stream._final_asr.transcribe = lambda _: 'complete'
        await stream._finish_asr(stream._pcm[0])
        self.assertEqual(stream._text, 'complete')

    async def test_reset_during_native_decode_cannot_publish_a_cache_entry(self):
        stream = self.make_stream()
        entered, release = threading.Event(), threading.Event()

        def transcribe(_):
            entered.set()
            release.wait(2)
            return 'obsolete'

        stream._final_asr.transcribe = transcribe
        task = stream.refine_current_snapshot()
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            stream._reset_turn()
        finally:
            release.set()
        self.assertEqual(await task, 'obsolete')
        self.assertIsNone(stream._refined_snapshot)
        self.assertEqual(stream._text, '')

    async def test_snapshot_from_another_stream_is_not_shared(self):
        first = self.make_stream()
        await first.refine_current_snapshot()
        second = self.make_stream()
        second._final_asr = first._final_asr
        await second._finish_asr(second._pcm[0])
        self.assertEqual(self.calls, 1)


if __name__ == '__main__':
    unittest.main()
