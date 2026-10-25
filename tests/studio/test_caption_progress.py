"""Playback caption boundaries reuse the authoritative heard-source mapping."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from studio.core.utils.audio_timeline.component import AudioTimeline
from studio.core.utils.conversation.component import Conversation


class CaptionProgressTests(unittest.IsolatedAsyncioTestCase):
    def session(self, *, managed=True):
        session = Conversation.__new__(Conversation)
        session.agent = SimpleNamespace(MIC_RATE=24000, BARGE_DEBUG=False)
        session.sock = SimpleNamespace(send_json=AsyncMock())
        timeline = AudioTimeline(track_delivery=True)
        timeline.context_managed = managed
        timeline.append_text('你好😀，这是合成测试。')
        segment = timeline.begin_segment(0, len(timeline.generated_text))
        timeline.append_audio(b'\0\0' * 24000)
        timeline.mark_sent(24000)
        timeline.finish_segment(segment)
        session.turn = dict(timeline=timeline, play_started=True, until=1)
        return session, timeline

    async def checkpoint(self, session, timeline, samples, *, state='playing', output_id=None):
        await session.playback_checkpoint(dict(output_id=output_id or timeline.output_id,
            rendered_samples=samples, sample_rate=24000, state=state))

    async def test_progress_is_monotonic_deduplicated_and_matches_heard_source(self):
        session, timeline = self.session()
        await self.checkpoint(session, timeline, 12000)
        session.sock.send_json.assert_awaited_once_with(dict(type='answer_progress',
            output_id=timeline.output_id, text_end=len(timeline.heard_text())))
        await self.checkpoint(session, timeline, 12000, state='paused')
        await self.checkpoint(session, timeline, 12000, state='stalled')
        self.assertEqual(session.sock.send_json.await_count, 1)
        await self.checkpoint(session, timeline, 24000, state='drained')
        self.assertEqual(session.sock.send_json.call_args.args[0]['text_end'], len(timeline.generated_text))
        self.assertEqual(session.turn['until'], 0)

    async def test_stale_frozen_interrupted_and_unmanaged_outputs_publish_nothing(self):
        for kind in ('stale', 'frozen', 'interrupted', 'unmanaged'):
            with self.subTest(kind=kind):
                session, timeline = self.session(managed=kind != 'unmanaged')
                if kind == 'frozen':
                    timeline.freeze_playback()
                if kind == 'interrupted':
                    timeline.mark_interrupted()
                await self.checkpoint(session, timeline, 24000,
                                      output_id='old' if kind == 'stale' else None)
                session.sock.send_json.assert_not_awaited()

    async def test_new_output_has_independent_progress_even_with_shorter_text(self):
        session, timeline = self.session()
        await self.checkpoint(session, timeline, 24000)
        second = AudioTimeline(track_delivery=True)
        second.context_managed = True
        second.append_text('好')
        second.append_audio(b'\0\0' * 24000)
        second.mark_sent(24000)
        second.generation_complete = True
        session.turn['timeline'] = second
        await self.checkpoint(session, second, 24000)
        session.sock.send_json.assert_awaited_with(dict(type='answer_progress',
            output_id=second.output_id, text_end=1))
