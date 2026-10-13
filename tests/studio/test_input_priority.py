"""Input/reply admission and background-memory priority without model inference."""
import asyncio
from contextlib import aclosing
import json
import types
import unittest
from unittest.mock import patch

from studio.core.utils.memory.component import Memory
from tests.helpers.capture import anticipate_namespace, state
from tests.helpers.conversation import ConversationFixture


def priority_owner():
    owner = Memory()
    owner._HOT = {'n': 0}
    owner._IDLE = asyncio.Event()
    owner._IDLE.set()
    owner._IDLE_MAX_WAIT_S = .5
    owner.BARGE_DEBUG = False
    return owner


class CapturePriorityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner = priority_owner()
        self.ns = anticipate_namespace()
        self.ns['hot_path_enter'] = self.owner.hot_path_enter
        self.ns['hot_path_exit'] = self.owner.hot_path_exit
        self.enterContext(patch('studio.core.utils.turn_taking.backchannel.emitting', return_value=False))
        self.enterContext(patch('studio.core.utils.turn_taking.backchannel.Backchannel.offer', return_value=None))

    def capture(self, messages, frames=(), *, said=lambda: ''):
        messages = iter(messages)
        frames = iter(frames)

        async def receive():
            return next(messages, {'type': 'websocket.disconnect'})

        async def feed(_):
            return next(frames)

        async def feed_text(text):
            self.assertFalse(self.owner._IDLE.is_set())
            return state(text, final=True).turn

        stream = types.SimpleNamespace(feed=feed, feed_text=feed_text)
        self.ns['vm'] = types.SimpleNamespace(stream=lambda **kw: stream)
        sock = types.SimpleNamespace(receive=receive, send_json=self.send)
        return self.ns['anticipate'](sock, said=said), sock

    async def send(self, _):
        pass

    async def test_voice_input_blocks_background_until_confirmed_turn_is_consumed(self):
        turns, _ = self.capture([{'bytes': bytes(1920)}, {'bytes': bytes(1920)}],
                                [state('synthetic input'), state('synthetic input', final=True)])
        async with aclosing(turns):
            pending = await anext(turns)
            self.assertEqual(pending.text, 'synthetic input')
            self.assertEqual(self.owner._HOT['n'], 1)
            idle = asyncio.create_task(self.owner.wait_idle())
            try:
                await asyncio.sleep(0)
                self.assertFalse(idle.done())
                with self.assertRaises(StopAsyncIteration):
                    await anext(turns)
                await asyncio.wait_for(idle, 1)
            finally:
                idle.cancel()
                await asyncio.gather(idle, return_exceptions=True)
        self.assertEqual(self.owner._HOT['n'], 0)

    async def test_text_input_is_protected_before_memory_lookup_and_released_on_close(self):
        turns, _ = self.capture([{'text': json.dumps({'type': 'user_text', 'text': 'synthetic text'})}])
        async with aclosing(turns):
            self.assertEqual((await anext(turns)).text, 'synthetic text')
            self.assertFalse(self.owner._IDLE.is_set())
        self.assertEqual(self.owner._HOT['n'], 0)
        self.assertTrue(self.owner._IDLE.is_set())

    async def test_echo_rejection_and_disconnect_release_input_priority(self):
        turns, _ = self.capture([{'bytes': bytes(1920)}, {'bytes': bytes(1920)}],
                                [state('synthetic echo'), state('synthetic echo', final=True)],
                                said=lambda: 'synthetic echo')
        async with aclosing(turns):
            self.assertEqual([pending async for pending in turns], [])
        self.assertEqual(self.owner._HOT['n'], 0)
        self.assertTrue(self.owner._IDLE.is_set())

    async def test_cancellation_during_speech_does_not_leave_memory_waiting(self):
        turns, sock = self.capture([{'bytes': bytes(1920)}], [state('synthetic speech')])
        waiting = asyncio.Event()
        receive = sock.receive
        count = 0

        async def wait_receive():
            nonlocal count
            count += 1
            if count == 1:
                return await receive()
            waiting.set()
            await asyncio.Event().wait()

        sock.receive = wait_receive
        task = asyncio.create_task(anext(turns))
        try:
            await asyncio.wait_for(waiting.wait(), 1)
            self.assertEqual(self.owner._HOT['n'], 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(self.owner._HOT['n'], 0)
            self.assertTrue(self.owner._IDLE.is_set())
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await turns.aclose()

    async def test_background_starvation_limit_remains_bounded(self):
        self.owner._IDLE_MAX_WAIT_S = .01
        token = self.owner.hot_path_enter()
        try:
            await asyncio.wait_for(self.owner.wait_idle(), .5)
            self.assertFalse(self.owner._IDLE.is_set())
        finally:
            self.owner.hot_path_exit(token)


class ReplyHandoffPriorityTests(ConversationFixture, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.priority = priority_owner()
        self.agent.hot_path_enter = self.priority.hot_path_enter
        self.agent.hot_path_exit = self.priority.hot_path_exit

    async def test_reply_is_reserved_before_child_can_start(self):
        task = await self.start()
        self.assertEqual(self.priority._HOT['n'], 1)
        await asyncio.wait_for(task, 1)
        await asyncio.sleep(0)
        self.assertEqual(self.priority._HOT['n'], 0)

    async def test_cancel_before_reply_child_starts_releases_reservation(self):
        task = await self.start()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
        self.assertEqual(self.model_calls, 0)
        self.assertEqual(self.priority._HOT['n'], 0)
        self.assertTrue(self.priority._IDLE.is_set())

    async def test_provider_failure_releases_reply_priority(self):
        async def fail(*_args, **_kwargs):
            raise RuntimeError('synthetic provider failure')

        self.agent.voicemem_llm_tts = fail
        task = await self.start()
        with self.assertRaisesRegex(RuntimeError, 'synthetic provider failure'):
            await task
        await asyncio.sleep(0)
        self.assertEqual(self.priority._HOT['n'], 0)
        self.assertTrue(self.priority._IDLE.is_set())


if __name__ == '__main__':
    unittest.main()
