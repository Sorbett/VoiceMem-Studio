"""Logical speech turns across unheard cancellation, using synthetic providers."""
import asyncio
from contextlib import asynccontextmanager
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.helpers.conversation import ConversationFixture
from studio.core.utils.conversation.input_turn import can_continue_input, join_input, refresh_continuation
from studio.core.utils.routing.component import Routing
from studio.core.utils.contracts.component import Pending
from studio.core.utils.realtime_session.component import RealtimeSession
from studio.core.utils.session_context.component import SessionBuffer
from voicemem import gate
from voicemem.stream import empty_result


class InputContinuationTests(ConversationFixture, unittest.IsolatedAsyncioTestCase):
    def test_joined_audio_features_use_complete_span_without_mutating_prior_inputs(self):
        base = self.pending('Synthetic first', input_audio_span=(0, 1), audio_state={'span_s': 1})
        tail = self.pending('Synthetic tail', input_audio_span=(1.2, 2.2),
                            audio_state={'span_s': 1, 'silence_s': .2})
        combined = join_input(base, tail)
        self.assertEqual(combined.audio_state, {'span_s': 2.2, 'silence_s': .2})
        self.assertEqual(tail.audio_state, {'span_s': 1, 'silence_s': .2})
        self.assertEqual(base.audio_state, {'span_s': 1})

    def setUp(self):
        super().setUp()
        self.block_model = True
        self.auto_playback = False
        self.agent._ensure_pending_memory = AsyncMock()
        self.agent._is_explicit_interrupt = lambda text: text.startswith('停一下')
        self.input_audio_cursor = 0.0

    def pending(self, text='最终确认的输入', **kwargs):
        span = kwargs.pop('input_audio_span',
                          (self.input_audio_cursor, self.input_audio_cursor + .4))
        if span is not None:
            self.input_audio_cursor = span[1] + .25
        return super().pending(text, input_audio_span=span, **kwargs)

    async def begin(self, text='我想做一个账号。', input_id='first'):
        pending = self.pending(text, input_turn_id=input_id)
        await self.session.publish_user_input(pending)
        task = await self.start(pending)
        await asyncio.wait_for(self.audio_ready.wait(), 1)
        return pending, task, self.session.turn['timeline']

    async def continue_input(self, text='主要做美妆。', input_id='second'):
        await self.session.on_user_speech()
        pending = await self.session.merge_continuation(
            self.pending(text, input_turn_id=input_id, early_ok=True))
        await self.session.publish_user_input(pending)
        return pending

    async def test_unheard_cancel_merges_ui_model_and_one_memory_turn(self):
        base, old_task, old_timeline = await self.begin()
        merged = await self.continue_input()
        self.assertTrue(old_task.done())
        self.agent.queue_remember_turn.assert_not_called()
        self.assertEqual(self.context.messages(self.session.context_session, 'fixture'), [])
        self.assertEqual(merged.text, '我想做一个账号。 主要做美妆。')
        self.assertEqual(merged.replace_input_turn_id, base.input_turn_id)
        self.assertFalse(merged.early_ok)
        self.model_waiting.clear()
        task = await self.start(merged)
        await asyncio.wait_for(self.model_waiting.wait(), 1)
        await self.session.stop_reply(force=True)
        self.assertTrue(task.done())
        self.assertTrue(old_timeline.context_saved)
        self.assertEqual(self.model_args[-1][0], merged.text)
        self.agent.queue_remember_turn.assert_called_once()
        self.assertIs(self.agent.queue_remember_turn.call_args.args[0], merged)
        self.assertEqual(self.context.messages(self.session.context_session, 'fixture'),
                         [{'role': 'user', 'content': merged.text}])
        finals = [m for m in self.messages if m['type'] == 'user_transcript']
        self.assertEqual(len(finals), 2)
        self.assertEqual(finals[-1]['replace_input_turn_id'], 'first')

    async def test_three_unheard_segments_keep_all_words_and_one_commit(self):
        await self.begin('我想做账号')
        second = await self.continue_input('主要做美妆')
        await self.start(second)
        third = await self.continue_input('美妆，要具体计划。', 'third')
        self.assertEqual(third.text, '我想做账号主要做美妆美妆，要具体计划。')
        self.assertEqual(third.replace_input_turn_id, 'second')
        self.agent.queue_remember_turn.assert_not_called()
        await self.session.close_session()
        await self.session.close_session()
        self.agent.queue_remember_turn.assert_called_once()
        self.assertIs(self.agent.queue_remember_turn.call_args.args[0], third)

    async def test_disconnect_between_cancellation_and_final_asr_saves_prefix(self):
        original, _, _ = await self.begin()
        await self.session.on_user_speech()
        await self.session.on_user_speech()
        await self.session.close_session()
        await self.session.close_session()
        self.agent.queue_remember_turn.assert_called_once()
        self.assertIs(self.agent.queue_remember_turn.call_args.args[0], original)

    async def test_failure_after_merge_before_output_adoption_saves_complete_input(self):
        await self.begin()
        merged = await self.continue_input()
        await self.session.close_session()
        self.agent.queue_remember_turn.assert_called_once()
        self.assertIs(self.agent.queue_remember_turn.call_args.args[0], merged)

    async def test_cancellation_during_merged_retrieval_preserves_accepted_text(self):
        await self.begin()
        await self.session.on_user_speech()
        entered = asyncio.Event()

        async def retrieve(*_):
            entered.set()
            await asyncio.Event().wait()

        self.agent._ensure_pending_memory.side_effect = retrieve
        task = asyncio.create_task(self.session.merge_continuation(
            self.pending('主要做美妆', input_turn_id='second')))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await self.session.close_session()
        self.agent.queue_remember_turn.assert_called_once()
        self.assertEqual(self.agent.queue_remember_turn.call_args.args[0].text,
                         '我想做一个账号。 主要做美妆')

    async def test_space_change_during_retrieval_does_not_move_old_prefix(self):
        original, _, _ = await self.begin()
        await self.session.on_user_speech()
        incoming = self.pending('主要做美妆', input_turn_id='second')

        async def retrieve(*_):
            self.agent.ACTIVE_SPACE = 'other'

        self.agent._ensure_pending_memory.side_effect = retrieve
        self.assertIs(await self.session.merge_continuation(incoming), incoming)
        self.agent.queue_remember_turn.assert_called_once()
        self.assertIs(self.agent.queue_remember_turn.call_args.args[0], original)
        self.assertEqual(self.agent._push_history.call_args.args[1], 'fixture')
        self.assertEqual(incoming.text, '主要做美妆')

    async def test_final_input_without_an_earlier_capture_candidate_also_merges(self):
        await self.begin()
        merged = await self.session.merge_continuation(
            self.pending('主要做美妆', input_turn_id='second'))
        self.assertEqual(merged.replace_input_turn_id, 'first')
        self.agent.queue_remember_turn.assert_not_called()

    async def test_long_silence_starts_a_new_input_even_when_output_is_unheard(self):
        original, _, _ = await self.begin()
        original.input_audio_span = (0.0, 1.0)
        incoming = self.pending('我还有另一个问题', input_turn_id='second')
        incoming.input_audio_span = (5.0, 6.0)
        await self.session.on_user_speech()
        self.assertIs(await self.session.merge_continuation(incoming), incoming)
        self.agent.queue_remember_turn.assert_called_once()
        self.assertIs(self.agent.queue_remember_turn.call_args.args[0], original)
        self.assertFalse(incoming.replace_input_turn_id)
        self.model_waiting.clear()
        await self.start(incoming)
        await asyncio.wait_for(self.model_waiting.wait(), 1)
        self.assertEqual(self.model_args[-1][0], incoming.text)
        await self.session.stop_reply(force=True)
        self.assertEqual(self.agent.queue_remember_turn.call_count, 2)

    async def test_short_gap_merges_even_when_the_new_utterance_is_long(self):
        original, _, _ = await self.begin()
        original.input_audio_span = (0, 1)
        incoming = self.pending('我想详细补充几个条件', input_turn_id='second',
                                input_audio_span=(1.4, 16))
        merged = await self.session.merge_continuation(incoming)
        self.assertEqual(merged.replace_input_turn_id, 'first')
        self.assertEqual(merged.input_audio_span, (0, 16))
        self.agent.queue_remember_turn.assert_not_called()

    async def test_unknown_capture_timing_does_not_reopen_an_old_bubble(self):
        await self.begin()
        incoming = self.pending('我还有一个问题', input_turn_id='second', input_audio_span=None)
        self.assertIs(await self.session.merge_continuation(incoming), incoming)
        self.agent.queue_remember_turn.assert_called_once()

    async def test_unfinished_followup_does_not_merge_after_a_long_pause(self):
        self.session.cached_ack = lambda _: None
        base = self.pending('因为', input_turn_id='first', input_audio_span=(0, 1))
        await self.session.defer_unfinished_reply(base)
        incoming = self.pending('我想问别的', input_turn_id='second', input_audio_span=(5, 6))
        self.assertIs(await self.session.merge_continuation(incoming), incoming)
        self.agent.queue_remember_turn.assert_called_once()
        self.assertIs(self.agent.queue_remember_turn.call_args.args[0], base)

    async def test_gap_after_a_merged_turn_uses_its_latest_voiced_end(self):
        await self.begin('我想做账号')
        second = await self.continue_input('主要做美妆')
        await self.start(second)
        third = self.pending('我还有一个问题', input_turn_id='third',
                             input_audio_span=(second.input_audio_span[1] + 3, 8))
        self.assertIs(await self.session.merge_continuation(third), third)
        self.agent.queue_remember_turn.assert_called_once()
        self.assertIs(self.agent.queue_remember_turn.call_args.args[0], second)

    async def test_started_main_audio_is_a_boundary_even_before_text_maps_to_audio(self):
        for checkpoint in ({'rendered_samples': 1}, {'rendered_samples': 0, 'event': 'started'}):
            with self.subTest(checkpoint=checkpoint):
                if self.session.prewarm['closed']:
                    super().setUp()
                    self.block_model = True
                    self.auto_playback = False
                original, _, timeline = await self.begin()
                await self.session.playback_checkpoint(dict(
                    output_id=timeline.output_id, sample_rate=24000, state='playing', **checkpoint))
                await self.session.on_user_speech()
                incoming = self.pending('新问题', input_turn_id='second')
                self.assertIs(await self.session.merge_continuation(incoming), incoming)
                self.agent.queue_remember_turn.assert_called_once()
                self.assertIs(self.agent.queue_remember_turn.call_args.args[0], original)
                await self.session.close_session()

    async def test_candidate_rejection_resumes_without_holding_or_persisting(self):
        await self.begin()
        await self.session.pause_candidate()
        await self.session.resume_candidate()
        self.assertIsNone(self.session.input_continuation.held)
        self.agent.queue_remember_turn.assert_not_called()
        self.assertFalse(self.session.turn['timeline'].interrupted)

    async def test_backchannel_is_not_a_main_audio_boundary(self):
        await self.begin()
        await self.session.emit_filler('嗯', bytes(4800))
        merged = await self.continue_input()
        self.assertEqual(merged.replace_input_turn_id, 'first')
        self.agent.queue_remember_turn.assert_not_called()

    async def test_privacy_scope_text_and_explicit_stop_never_merge(self):
        for change in ('space', 'memory', 'speaker', 'stranger', 'typed', 'stop'):
            with self.subTest(change=change):
                self.setUp()
                original, _, _ = await self.begin()
                await self.session.on_user_speech()
                incoming = self.pending('新问题', input_turn_id='second')
                if change == 'space':
                    self.agent.ACTIVE_SPACE = 'other'
                elif change == 'memory':
                    self.agent.vm = object()
                elif change == 'speaker':
                    self.session.owner['last'] = 'other-speaker'
                elif change == 'stranger':
                    incoming.stranger = True
                elif change == 'typed':
                    incoming.spoken = False
                else:
                    incoming.text = '停一下'
                self.assertIs(await self.session.merge_continuation(incoming), incoming)
                self.agent.queue_remember_turn.assert_called_once()
                self.assertIs(self.agent.queue_remember_turn.call_args.args[0], original)
                self.assertIs(self.agent.queue_remember_turn.call_args.kwargs['memory_vm'], self.memory)
                self.assertEqual(self.agent._push_history.call_args.args[1], 'fixture')
                await self.session.close_session()

    async def test_stale_playback_and_repeated_finalizers_cannot_commit_an_old_fragment(self):
        original, _, timeline = await self.begin()
        merged = await self.continue_input()
        await self.start(merged)
        await self.session.playback_checkpoint(dict(
            output_id=timeline.output_id, rendered_samples=24000, state='drained'))
        self.session.save_reply_context(original, timeline, self.memory, 'fixture')
        self.agent.queue_remember_turn.assert_not_called()
        await self.session.stop_reply(force=True)
        self.agent.queue_remember_turn.assert_called_once()

    async def test_unfinished_wait_and_later_continuation_do_not_duplicate_prefix(self):
        self.session.cached_ack = lambda _: None
        base = self.pending('因为', input_turn_id='first')
        await self.session.publish_user_input(base)
        await self.session.defer_unfinished_reply(base)
        merged = await self.session.merge_continuation(
            self.pending('我想试试', input_turn_id='second'))
        self.assertEqual(merged.text, '因为我想试试')
        await self.session.close_session()
        self.agent.queue_remember_turn.assert_called_once()
        self.assertIs(self.agent.queue_remember_turn.call_args.args[0], merged)

    async def test_continuation_invalidates_private_early_reply_and_fragment_retrieval(self):
        await self.begin()
        sink, _, early_task = await self.early()
        merged = await self.continue_input()
        self.assertTrue(early_task.done())
        self.assertFalse(sink.live)
        self.assertIsNone(self.session.early['task'])
        self.agent._ensure_pending_memory.assert_awaited_once_with(merged, self.memory)


class ContinuationMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_query_is_retrieved_even_when_depth_router_is_disabled(self):
        memory = types.SimpleNamespace(
            classify=lambda text: types.SimpleNamespace(slots=['work'], entities=[]),
            search=lambda text, **kw: empty_result())
        agent = Routing()
        agent._replay_id = lambda *args: ''
        agent._THINKING_ROUTER_ON = False
        pending = ConversationFixture.pending(self, '我想讨论上次的计划，美妆账号怎么做？')
        pending.memory_context, pending.replay = 'stale fragment context', 'stale-replay'
        with patch.object(memory, 'search', wraps=memory.search) as search:
            await refresh_continuation(agent, pending, memory)
        self.assertEqual(search.call_args.args[0], pending.text)
        self.assertEqual(pending.route, gate.DEEP)
        self.assertEqual(pending.replay, '')
        self.assertNotIn('stale', pending.memory_context)

    async def test_memory_failure_keeps_complete_text_and_clears_stale_hits(self):
        agent = types.SimpleNamespace(_ensure_pending_memory=AsyncMock(side_effect=RuntimeError('synthetic')))
        pending = ConversationFixture.pending(self, '我想继续上次的计划')
        pending.memory_context = 'stale'
        await refresh_continuation(agent, pending, object())
        self.assertEqual(pending.text, '我想继续上次的计划')
        self.assertEqual(pending.memory_context, '')
        self.assertEqual(pending.result.hits, [])


class ContinuationWindowTests(unittest.TestCase):
    def test_window_boundary_epoch_and_invalid_spans(self):
        base = Pending('首句', '', empty_result(), input_audio_span=(0, 1))
        for span, expected in (((1.0, 2), True), ((2.2, 3), True), ((2.21, 3), False),
                               ((.9, 2), False), ((2, 1), False),
                               ((float('nan'), 2), False), ((2, float('inf')), False),
                               (None, False)):
            with self.subTest(span=span):
                incoming = Pending('次句', '', empty_result(), input_audio_span=span)
                self.assertEqual(can_continue_input(base, incoming), expected)
        incoming = Pending('重开麦克风', '', empty_result(),
                           input_audio_span=(1.2, 2), input_audio_epoch=1)
        self.assertFalse(can_continue_input(base, incoming))


class RealtimeContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def run_session(self, *, heard=False, gap_s=.25):
        events = asyncio.Queue()
        audio_delivered = asyncio.Event()
        messages, instructions = [], []

        class Connection:
            def __aiter__(self):
                return self

            async def __anext__(self):
                return await events.get()

        conn = Connection()

        async def cancel():
            await events.put(types.SimpleNamespace(type='response.cancelled'))

        conn.response = types.SimpleNamespace(create=AsyncMock(), cancel=AsyncMock(side_effect=cancel))
        conn.session = types.SimpleNamespace(update=AsyncMock())
        conn.input_audio_buffer = types.SimpleNamespace(commit=AsyncMock())
        conn.truncate_output = AsyncMock()

        @asynccontextmanager
        async def connect(_):
            yield conn

        async def send(message):
            messages.append(message)

        async def send_audio(_):
            audio_delivered.set()

        memory = object()
        agent = RealtimeSession()
        agent.ACTIVE_SPACE, agent.vm = 'fixture', memory
        agent.BARGE_DEBUG, agent.MIC_RATE, agent.REPLY = False, 24000, object()
        agent._turn_detection = lambda: {}
        agent._SESSION_CONTEXT = SessionBuffer()
        agent._push_history = Mock(return_value='history')
        agent.queue_remember_turn = Mock()
        agent._is_backchannel = lambda _: False
        agent._is_explicit_interrupt = lambda _: False
        agent._ensure_pending_memory = AsyncMock()
        agent.note_hits = Mock()
        agent.fill_tags = lambda *a, **kw: {}
        agent.audio_of = agent.hit_cluster = None
        agent._kick_acoustic = lambda *a, **kw: None
        agent._realtime_instructions = lambda *a, **kw: instructions.append(kw['text']) or 'fixture'

        async def capture(_sock, **callbacks):
            yield Pending('我想做账号', '', empty_result(), input_turn_id='first',
                          input_audio_span=(0, 1))
            await events.put(types.SimpleNamespace(
                type='response.output_audio_transcript.delta', delta='可以', item_id='provider-first'))
            await events.put(types.SimpleNamespace(
                type='response.output_audio.delta', delta='AAA='))
            await asyncio.wait_for(audio_delivered.wait(), 1)
            first_output = next(m['output_id'] for m in messages if m['type'] == 'answer_start')
            if heard:
                await callbacks['on_playback_checkpoint'](dict(
                    output_id=first_output, rendered_samples=1, event='started', state='playing'))
            await callbacks['on_speech']()
            if not heard:
                agent.queue_remember_turn.assert_not_called()
            yield Pending('主要做美妆', '', empty_result(), input_turn_id='second',
                          input_audio_span=(1 + gap_s, 2 + gap_s))

        agent.anticipate = capture
        sock = types.SimpleNamespace(send_json=send, send_bytes=send_audio)
        with patch('studio.core.utils.realtime_session.component.utils.realtime_connect', connect), \
                patch('studio.core.utils.realtime_session.component.utils.hits_payload', return_value={}):
            await asyncio.wait_for(agent.realtime_session(sock), 2)
        return agent, conn, messages, instructions

    async def test_realtime_preserves_provider_audio_and_commits_one_combined_input(self):
        agent, conn, messages, instructions = await self.run_session()
        self.assertEqual(instructions, ['我想做账号', '我想做账号主要做美妆'])
        self.assertEqual(conn.input_audio_buffer.commit.await_count, 2)
        conn.response.cancel.assert_awaited_once()
        conn.truncate_output.assert_awaited_once()
        finals = [m for m in messages if m['type'] == 'user_transcript']
        self.assertEqual(finals[-1]['replace_input_turn_id'], 'first')
        self.assertEqual(finals[-1]['text'], instructions[-1])
        agent.queue_remember_turn.assert_called_once()
        self.assertEqual(agent.queue_remember_turn.call_args.args[0].text, instructions[-1])

    async def test_realtime_heard_reply_keeps_the_next_request_separate(self):
        agent, conn, messages, instructions = await self.run_session(heard=True)
        self.assertEqual(instructions, ['我想做账号', '主要做美妆'])
        self.assertEqual(agent.queue_remember_turn.call_count, 2)
        finals = [m for m in messages if m['type'] == 'user_transcript']
        self.assertNotIn('replace_input_turn_id', finals[-1])

    async def test_realtime_long_pause_keeps_inputs_and_memory_turns_separate(self):
        agent, conn, messages, instructions = await self.run_session(gap_s=4)
        self.assertEqual(instructions, ['我想做账号', '主要做美妆'])
        self.assertEqual(agent.queue_remember_turn.call_count, 2)
        finals = [m for m in messages if m['type'] == 'user_transcript']
        self.assertNotIn('replace_input_turn_id', finals[-1])
        self.assertEqual(conn.input_audio_buffer.commit.await_count, 2)


if __name__ == '__main__':
    unittest.main()
