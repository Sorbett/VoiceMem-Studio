"""Capture reuses known emotion without starting acoustic model work."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from tests.helpers.capture import anticipate_namespace, state


def paused(text='测试这一轮'):
    snapshot = state(text)
    snapshot.state, snapshot.silence = '<silence>', .1
    snapshot._pcm = np.ones(1600, dtype=np.float32)
    return snapshot


class CaptureSession:
    def __init__(self, frames):
        self.ns = anticipate_namespace()
        self.frames, self.current = iter(frames), None
        self.playback = {'busy': False, 'text': ''}
        self.sent, self.turns = [], []
        self.detector = Mock(side_effect=AssertionError('unexpected emotion inference'))
        self.memory = SimpleNamespace(
            preprocess=Mock(side_effect=AssertionError('unexpected full perception')),
            _o=SimpleNamespace(_get_emotion_detector=self.detector))
        self.snapshot = Mock(side_effect=AssertionError('unexpected audio snapshot'))
        self.stream = SimpleNamespace(
            vm=self.memory, feed=self.feed, confirm_s=.2, aclose=AsyncMock(),
            audio_snapshot=self.snapshot, feed_text=self.feed_text)
        self.memory.stream = lambda **kwargs: self.stream
        self.ns['vm'] = self.memory
        self.owner = {'id': '', 'last': '', 'miss': 0, 'emotion': 'known'}

    async def feed(self, raw):
        self.current[2]._vm = self.memory
        return self.current[2]

    async def feed_text(self, text):
        return state(text, final=True).turn

    async def receive(self):
        await asyncio.sleep(0)
        item = next(self.frames, None)
        if callable(item):
            item = await item()
        if item is None:
            return {'type': 'websocket.disconnect'}
        if isinstance(item, dict):
            return item
        self.current = item
        self.playback.update(busy=item[0], text=item[1])
        return {'bytes': np.zeros(960, dtype=np.int16).tobytes()}

    async def send_json(self, message):
        self.sent.append(message)

    async def run(self):
        with patch('studio.core.utils.turn_taking.backchannel.emitting', return_value=False), \
                patch('studio.core.utils.turn_taking.backchannel.Backchannel.offer', return_value=None):
            async for turn in self.ns['anticipate'](
                    self, owner=self.owner, is_busy=lambda: self.playback['busy'],
                    said=lambda: self.playback['text']):
                self.turns.append(turn)
        self.memory.preprocess.assert_not_called()
        self.detector.assert_not_called()
        self.snapshot.assert_not_called()
        self.stream.aclose.assert_awaited_once()
        return self.turns


class CaptureEmotionTests(unittest.IsolatedAsyncioTestCase):
    async def test_pauses_resumption_and_final_audio_reuse_known_emotion(self):
        final = state('继续说完的这一轮', final=True)
        final._pcm = np.ones(3200, dtype=np.float32)
        session = CaptureSession([
            (False, '', paused()),
            (False, '', state('继续说')),
            (False, '', paused('继续说完的这一轮')),
            (False, '', final),
            (False, '', paused('下一轮问题')),
            (False, '', state('下一轮问题', final=True))])
        turns = await asyncio.wait_for(session.run(), 1)
        self.assertEqual([turn.emotion for turn in turns], ['known', 'known'])
        self.assertEqual(session.owner['emotion'], 'known')

    async def test_missing_known_emotion_does_not_trigger_refresh(self):
        session = CaptureSession([
            (False, '', paused()),
            (False, '', state('测试这一轮', final=True))])
        session.owner.pop('emotion')
        turns = await session.run()
        self.assertEqual([turn.emotion for turn in turns], [''])

    async def test_handoff_uses_latest_known_emotion(self):
        session = CaptureSession([])

        async def confirm():
            session.owner['emotion'] = 'updated'
            return (False, '', state('测试这一轮', final=True))

        session.frames = iter([(False, '', paused()), confirm])
        turns = await session.run()
        self.assertEqual([turn.emotion for turn in turns], ['updated'])

    async def test_rejected_final_paths_do_not_affect_next_turn(self):
        for reason in ('echo', 'backchannel', 'candidate', 'discarded-candidate', 'replay', 'emission'):
            with self.subTest(reason=reason):
                busy = reason in {'echo', 'backchannel', 'candidate', 'discarded-candidate'}
                final_text = {'echo': '测试助手的回声', 'backchannel': 'ok.',
                              'candidate': '', 'discarded-candidate': '',
                              'replay': '', 'emission': '测试附和的回声'}[reason]
                session = CaptureSession([
                    (busy, '测试助手的回声' if busy else '', paused('不完整输入')),
                    (busy, '测试助手的回声' if busy else '', state(final_text, final=True)),
                    (False, '', state('下一轮问题', final=True))])
                if reason in {'candidate', 'discarded-candidate'}:
                    session.ns['CANDIDATE_MIN_SPEECH_S'] = 0
                    session.ns['BARGE_STABLE_UPDATES'] = 3
                    session.frames = iter([
                        (True, '测试助手的回声', state('有内容的插话')),
                        *list(session.frames)])
                    if reason == 'discarded-candidate':
                        initial = next(session.frames)
                        pause = next(session.frames)
                        session.frames = iter([
                            initial, pause,
                            (True, '测试助手的回声', paused('测试助手的回声')),
                            *list(session.frames)])
                if reason == 'replay':
                    session.ns['_replaying_now'] = lambda: True
                if reason == 'emission':
                    from studio.core.utils.turn_taking.component import TurnTakingStateMachine
                    machine = TurnTakingStateMachine()
                    machine.record_emission(final_text)
                    session.ns['TurnTakingStateMachine'] = lambda **kwargs: machine
                turns = await session.run()
                self.assertEqual([turn.text for turn in turns], ['下一轮问题'])
                self.assertEqual([turn.emotion for turn in turns], ['known'])

    async def test_text_input_after_audio_pause_reuses_known_emotion(self):
        session = CaptureSession([
            (False, '', paused()),
            {'text': json.dumps({'type': 'user_text', 'text': '文字问题'})}])
        turns = await session.run()
        self.assertEqual([turn.text for turn in turns], ['文字问题'])
        self.assertFalse(turns[0].spoken)
        self.assertEqual(turns[0].emotion, 'known')

    async def test_disconnect_during_pause_leaves_no_hint_tasks(self):
        session = CaptureSession([(False, '', paused())])
        self.assertEqual(await asyncio.wait_for(session.run(), 1), [])
        self.assertFalse(any(task.get_name().startswith('capture-emotion-')
                             for task in asyncio.all_tasks()))

    async def test_stranger_filter_remains_active(self):
        session = CaptureSession([
            (False, '', paused()),
            (False, '', state('测试这一轮', final=True))])
        session.ns['SPEAKER_GATE'] = True
        session.owner.update(id='synthetic-speaker', miss=1)
        turns = await session.run()
        self.assertEqual([turn.emotion for turn in turns], ['known'])
        self.assertTrue(turns[0].stranger)
        self.assertEqual(turns[0].memory_context, '')
        self.assertEqual(turns[0].replay, '')


if __name__ == '__main__':
    unittest.main()
