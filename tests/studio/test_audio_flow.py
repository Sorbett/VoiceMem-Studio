"""Bounded audio delivery, speculative commit and feedback fallback."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from studio.core.utils.contracts.component import ReplySink
from studio.core.utils.conversation.component import Conversation
from studio.core.utils.audio_timeline.component import AudioTimeline
from studio.core.utils.tts.buffering import PCMBuffer
from studio.web.audio_flow import PlaybackBudget


class AudioFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_error_needs_no_queue_capacity_and_preserves_chunks(self):
        queue = PCMBuffer(24000)
        pcm = b'\0\0' * 24000
        await queue.put(pcm)
        error = RuntimeError('synthetic')
        queue.close(error)
        self.assertEqual(await queue.get(), pcm)
        self.assertIs(await queue.get(), error)
        self.assertIsNone(await queue.get())

    async def test_one_oversized_chunk_and_timestamps_are_preserved(self):
        queue = PCMBuffer(24000)
        chunk = SimpleNamespace(pcm=b'\0\0' * 48000, timestamps=('synthetic',))
        await queue.put(chunk)
        pending = asyncio.create_task(queue.put(b'\1\0'))
        await asyncio.sleep(0)
        self.assertFalse(pending.done())
        self.assertIs(await queue.get(), chunk)
        await asyncio.wait_for(pending, 1)
        self.assertEqual(await queue.get(), b'\1\0')

    async def test_speculative_audio_waits_outside_lock_and_commit_preserves_order(self):
        events = []

        async def send(message):
            events.append(message)

        sink = ReplySink(send, send)
        first, second = b'\0\0' * 24000, b'\1\0' * 24000
        await sink.send({'type': 'user_transcript', 'text': 'draft'})
        await sink.send_audio(first)
        pending = asyncio.create_task(sink.send_audio(second))
        await asyncio.sleep(0)
        self.assertFalse(pending.done())
        self.assertEqual(sink.buffered_ms, 1000)
        self.assertEqual(events, [])
        await asyncio.wait_for(sink.wait_for_output(), 1)
        await asyncio.wait_for(sink.commit(final_user_text='confirmed'), 1)
        await asyncio.wait_for(pending, 1)
        self.assertEqual(events, [{'type': 'user_transcript', 'text': 'confirmed'}, first, second])
        self.assertEqual(sink.buffered_ms, 0)

    async def test_rejected_speculation_can_cancel_at_the_budget(self):
        send = AsyncMock()
        sink = ReplySink(send, send)
        await sink.send_audio(b'\0\0' * 24000)
        pending = asyncio.create_task(sink.send_audio(b'\1\0'))
        await asyncio.sleep(0)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        send.assert_not_awaited()
        self.assertFalse(sink._lock.locked())

    async def test_playback_low_watermark_pause_resume_and_stall(self):
        now = [0.0]
        budget = PlaybackBudget(clock=lambda: now[0])
        budget.checkpoint(72000, 24000, 'paused')
        pending = asyncio.create_task(budget.wait())
        await asyncio.sleep(0)
        self.assertFalse(pending.done())
        budget.checkpoint(36000, 24000, 'playing')
        await asyncio.sleep(0)
        self.assertFalse(pending.done())
        budget.checkpoint(12000, 24000, 'playing')
        await asyncio.wait_for(pending, 1)
        budget.checkpoint(0, 24000, 'stalled')
        await asyncio.wait_for(budget.wait(), 1)

    async def test_missing_invalid_or_stale_reports_cannot_hang_legacy_delivery(self):
        now = [0.0]
        budget = PlaybackBudget(clock=lambda: now[0])
        budget.sent(20)
        await budget.wait()
        for samples, rate in ((float('nan'), 24000), (1, 0), (1, float('inf')), (-1, 24000)):
            budget.checkpoint(samples, rate, 'paused')
            self.assertIsNone(budget._updated)
        budget.checkpoint(72000, 24000, 'paused')
        now[0] = 3
        await asyncio.wait_for(budget.wait(), 1)
        budget = PlaybackBudget(stale_s=.02)
        budget.checkpoint(72000, 24000, 'paused')
        await asyncio.wait_for(budget.wait(), .5)

    async def test_send_counts_new_audio_and_cancel_releases_waiter(self):
        budget = PlaybackBudget()
        budget.checkpoint(24000, 24000, 'paused')
        budget.sent(2)
        pending = asyncio.create_task(budget.wait())
        await asyncio.sleep(0)
        self.assertFalse(pending.done())
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending

    async def test_old_output_cannot_change_budget_or_send_after_output_switch(self):
        conversation = Conversation.__new__(Conversation)
        conversation.agent = SimpleNamespace(MIC_RATE=24000, BARGE_DEBUG=False)
        timeline = AudioTimeline(track_delivery=True)
        timeline.delivery_budget = budget = PlaybackBudget()
        conversation.turn = dict(timeline=timeline, play_started=True, until=0)
        conversation.sock = SimpleNamespace(send_bytes=AsyncMock())
        await conversation.playback_checkpoint(dict(output_id='old', rendered_samples=0,
            buffered_samples=72000, sample_rate=24000, state='paused'))
        self.assertIsNone(budget._updated)
        await conversation.playback_checkpoint(dict(output_id=timeline.output_id, rendered_samples=0,
            buffered_samples=72000, sample_rate=24000, state='paused'))
        pending = asyncio.create_task(conversation.send_audio(b'\0\0', timeline))
        await asyncio.sleep(0)
        self.assertFalse(pending.done())
        conversation.turn['timeline'] = AudioTimeline()
        budget.checkpoint(0, 24000, 'playing')
        await asyncio.wait_for(pending, 1)
        conversation.sock.send_bytes.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
