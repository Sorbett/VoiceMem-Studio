"""Speech candidates and confirmed interruptions with synthetic input only."""
import ast
import asyncio
from pathlib import Path
import textwrap
import time
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from tests.helpers.capture import anticipate_namespace, state
from tests.helpers.conversation import ConversationFixture
from studio.core.utils.audio_timeline.component import AudioTimeline
from studio.core.utils.conversation.input_turn import InputContinuation


class CaptureCandidateTests(unittest.IsolatedAsyncioTestCase):
    async def capture(self, snapshots, reference='synthetic assistant reply'):
        ns = anticipate_namespace()
        ns['CANDIDATE_MIN_SPEECH_S'] = .12
        snapshots = iter(snapshots)
        current = None
        sent = []
        pause, resume, interrupt = AsyncMock(), AsyncMock(), AsyncMock()

        class Socket:
            async def receive(self):
                nonlocal current
                current = next(snapshots, None)
                if current is None:
                    return {'type': 'websocket.disconnect'}
                return {'bytes': np.zeros(960, np.int16).tobytes()}

            async def send_json(self, message):
                sent.append(message)

        async def feed(_):
            return current

        stream = types.SimpleNamespace(feed=feed, confirm_s=.2, aclose=AsyncMock())
        ns['vm'] = types.SimpleNamespace(stream=lambda **kw: stream)
        with patch('studio.core.utils.turn_taking.backchannel.emitting', return_value=False), \
                patch('studio.core.utils.turn_taking.backchannel.Backchannel.offer', return_value=None):
            turns = [turn async for turn in ns['anticipate'](
                Socket(), is_busy=lambda: True, said=lambda: reference,
                on_candidate=pause, on_candidate_reject=resume, on_speech=interrupt)]
        stream.aclose.assert_awaited_once()
        return pause, resume, interrupt, sent, turns

    async def test_sustained_speech_pauses_before_the_first_asr_text(self):
        pause, resume, interrupt, _, _ = await self.capture([state('') for _ in range(6)])
        pause.assert_awaited_once()
        resume.assert_not_awaited()
        interrupt.assert_not_awaited()

    async def test_subthreshold_sound_does_not_pause(self):
        pause, _, interrupt, _, _ = await self.capture([state(''), state('')])
        pause.assert_not_awaited()
        interrupt.assert_not_awaited()

    async def test_delayed_transcript_confirms_one_interruption(self):
        pause, resume, interrupt, _, _ = await self.capture([
            *[state('') for _ in range(4)], state('新的提问'), state('新的提问还有补充'),
            state('新的提问还有更多补充')])
        pause.assert_awaited_once()
        resume.assert_not_awaited()
        interrupt.assert_awaited_once()

    async def test_acknowledgement_and_echo_resume_the_same_output(self):
        for text, event in [('嗯', 'user_backchannel'), ('synthetic assistant reply', None)]:
            with self.subTest(text=text):
                pause, resume, interrupt, sent, turns = await self.capture([
                    *[state('') for _ in range(4)], state(text, final=True)])
                pause.assert_awaited_once()
                resume.assert_awaited_once()
                interrupt.assert_not_awaited()
                self.assertEqual(turns, [])
                self.assertEqual(sum(m['type'] == 'user_backchannel' for m in sent), int(event is not None))

    async def test_empty_candidate_times_out_without_a_reply(self):
        quiet = [state('') for _ in range(6)]
        for snapshot in quiet:
            snapshot.state = '<silence>'
        pause, resume, interrupt, _, turns = await self.capture([
            *[state('') for _ in range(4)], *quiet, state('', final=True)])
        pause.assert_awaited_once()
        resume.assert_awaited_once()
        interrupt.assert_not_awaited()
        self.assertEqual(turns, [])


class ConfirmedInterruptionTests(ConversationFixture, unittest.IsolatedAsyncioTestCase):
    async def test_confirmed_capture_interrupt_bypasses_reply_start_grace(self):
        self.block_model = True
        self.auto_playback = False
        task = await self.start()
        await asyncio.wait_for(self.audio_ready.wait(), 1)
        self.agent._session_anticipate = Mock()
        self.session.listen()
        interrupt = self.agent._session_anticipate.call_args.kwargs['on_speech']
        self.session.turn['t0'] = time.monotonic()
        await self.session.pause_candidate()
        await interrupt()
        self.assertTrue(task.done())
        self.assertTrue(self.model_closed.is_set())
        self.assertTrue(self.session.turn['timeline'].interrupted)
        self.assertFalse(self.session.candidate_paused)
        events = [m for m in self.messages if m['type'] == 'answer_interrupt']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['heard_text'], '')
        self.agent.queue_remember_turn.assert_called_once()

    async def test_pause_and_resume_carry_the_output_identity(self):
        self.block_model = True
        await self.start()
        await asyncio.wait_for(self.audio_ready.wait(), 1)
        timeline = self.session.turn['timeline']
        await self.session.pause_candidate()
        await self.session.resume_candidate()
        events = [m for m in self.messages if m['type'] in {'answer_pause', 'answer_resume'}]
        self.assertEqual([m['output_id'] for m in events], [timeline.output_id] * 2)
        self.assertEqual(self.model_calls, 1)


class RealtimeInterruptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirmed_realtime_interrupt_bypasses_reply_start_grace(self):
        path = Path(__file__).resolve().parents[2] / 'studio/core/utils/realtime_session/component.py'
        node = next(n for n in ast.walk(ast.parse(path.read_text()))
                    if isinstance(n, ast.AsyncFunctionDef) and n.name == 'on_speech')
        wrapper = ('def handler():\n    candidate_paused = True\n    candidate_paused_at = 0.0\n'
                   + textwrap.indent(ast.unparse(node), '    ') + '\n    return on_speech\n')
        timeline = AudioTimeline(track_delivery=True)
        idle = asyncio.Event()
        idle.set()
        sock = types.SimpleNamespace(send_json=AsyncMock())
        close = Mock()
        ns = dict(asyncio=asyncio, time=time, hearing=lambda: True,
                  self=types.SimpleNamespace(BARGE_DEBUG=False, BARGE_GRACE_MS=500),
                  turn=dict(t0=time.monotonic(), until=0, live=True, timeline=timeline,
                            provider_item_id=''), response_idle=idle, sock=sock, close_turn=close,
                  input_continuation=InputContinuation(), owner={})
        exec(compile(wrapper, str(path), 'exec'), ns)
        await ns['handler']()()
        close.assert_called_once_with(interrupted=True)
        sock.send_json.assert_awaited_once_with(dict(
            type='answer_interrupt', output_id=timeline.output_id, heard_text=''))
        self.assertTrue(timeline.interrupted)
