"""Background admission and native lifetime using synthetic jobs only."""
import asyncio
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from studio.core.utils.memory.background import BackgroundLane, background_lane
from studio.core.utils.memory.component import Memory
from studio.core.utils.perception.component import Perception


class BackgroundSchedulingTests(unittest.IsolatedAsyncioTestCase):
    def owner(self):
        class Agent(Memory, Perception):
            pass
        agent = Agent()
        agent._HOT = {"n": 0}
        agent._IDLE = asyncio.Event()
        agent._IDLE.set()
        agent._NATIVE_IDLE = threading.Event()
        agent._NATIVE_IDLE.set()
        agent._REMEMBER_TASKS = set()
        agent.BARGE_DEBUG = False
        agent._finish_history_turn = Mock()
        agent.ACOUSTIC_MIN_SCORE = .92
        agent.ACOUSTIC_TRUST = {"开心"}
        return agent

    async def test_durable_completion_bounds_ingest_and_acoustic_concurrency(self):
        agent = self.owner()
        callbacks, seen = [], []
        entered = asyncio.Event()
        loop = asyncio.get_running_loop()

        def ingest(text, **kwargs):
            seen.append(text)
            callbacks.append(kwargs["on_complete"])
            loop.call_soon_threadsafe(entered.set)
            return {"speaker_id": ""}

        agent.vm = SimpleNamespace(ingest=ingest)
        pending = SimpleNamespace(text="first fixture", audio_path="", language="en")
        agent.queue_remember_turn(pending, "", {}, memory_vm=agent.vm)
        await asyncio.wait_for(entered.wait(), 1)
        agent.queue_remember_turn(SimpleNamespace(**{**vars(pending), "text": "second fixture"}),
                                  "", {}, memory_vm=agent.vm)
        agent._acoustic_emotion = Mock(return_value=("开心", .99))
        send = Mock()
        acoustic = agent._kick_acoustic(send, "synthetic.wav", output_id="old",
                                       is_current=lambda: False)
        await asyncio.sleep(.02)
        self.assertEqual(seen, ["first fixture"])
        self.assertTrue(agent._MEMORY_WRITES)
        agent._acoustic_emotion.assert_not_called()
        entered.clear()
        callbacks[0]({"persistent_memory_created": True})
        await asyncio.wait_for(entered.wait(), 1)
        self.assertEqual(seen, ["first fixture", "second fixture"])
        callbacks[1]({"persistent_memory_created": False})
        await asyncio.wait_for(asyncio.gather(*tuple(agent._REMEMBER_TASKS), acoustic), 1)
        self.assertFalse(agent._MEMORY_WRITES)
        agent._acoustic_emotion.assert_not_called()

    async def test_cancelling_started_native_work_keeps_lane_until_native_returns(self):
        lane = BackgroundLane()
        loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), threading.Event()
        second_entered = asyncio.Event()

        def native():
            loop.call_soon_threadsafe(entered.set)
            release.wait(2)

        async def idle():
            return

        async def next_job():
            second_entered.set()

        first = asyncio.create_task(lane.run(idle, lambda: asyncio.to_thread(native)))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(lane.run(idle, next_job))
            await asyncio.sleep(.02)
            self.assertFalse(second_entered.is_set())
            release.set()
            await asyncio.wait_for(second, 1)
        finally:
            release.set()
            await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
            await asyncio.gather(*tuple(lane.tasks), return_exceptions=True)

    async def test_cancelling_queued_job_or_changing_owner_does_not_run_native_work(self):
        lane = BackgroundLane()
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def idle():
            entered.set()
            await release.wait()

        async def work():
            calls.append(True)

        waiting = asyncio.create_task(lane.run(idle, work))
        await asyncio.wait_for(entered.wait(), 1)
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        release.set()
        await lane.run(idle, work, is_current=lambda: False)
        self.assertEqual(calls, [])
        await lane.run(idle, work)
        self.assertEqual(calls, [True])

    async def test_failure_releases_lane_for_the_next_job(self):
        lane = BackgroundLane()

        async def idle():
            pass

        async def fail():
            raise RuntimeError("synthetic native failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic native failure"):
            await lane.run(idle, fail)
        self.assertEqual(await lane.run(idle, lambda: asyncio.sleep(0, result="ok")), "ok")

    async def test_foreground_native_event_follows_balanced_input_tokens(self):
        agent = self.owner()
        one, two = agent.hot_path_enter(), agent.hot_path_enter()
        self.assertFalse(agent._NATIVE_IDLE.is_set())
        agent.hot_path_exit(one)
        self.assertFalse(agent._NATIVE_IDLE.is_set())
        agent.hot_path_exit(two)
        self.assertTrue(agent._NATIVE_IDLE.is_set())
        self.assertIs(background_lane(agent), background_lane(agent))

    async def test_confirmed_memory_progresses_during_continuous_input_and_drains_on_shutdown(self):
        agent = self.owner()
        agent.BACKGROUND_DEFER_S = .01
        token = agent.hot_path_enter()
        callbacks, seen = [], []
        entered = asyncio.Event()
        loop = asyncio.get_running_loop()

        def ingest(text, **kwargs):
            callbacks.append(kwargs['on_complete'])
            seen.append(text)
            loop.call_soon_threadsafe(entered.set)
            return {}

        agent.vm = SimpleNamespace(ingest=ingest)
        pending = SimpleNamespace(text='first fixture', audio_path='', language='en')
        agent.queue_remember_turn(pending, '', {})
        await asyncio.wait_for(entered.wait(), 1)
        self.assertEqual(agent._HOT['n'], 1)
        self.assertFalse(agent._IDLE.is_set())
        agent.queue_remember_turn(SimpleNamespace(**{**vars(pending), 'text': 'second fixture'}), '', {})
        drain = asyncio.create_task(agent.drain_memory())
        await asyncio.sleep(.01)
        self.assertFalse(drain.done())
        entered.clear()
        callbacks[0]({'persistent_memory_created': True})
        await asyncio.wait_for(entered.wait(), 1)
        self.assertEqual(seen, ['first fixture', 'second fixture'])
        self.assertFalse(drain.done())
        callbacks[1]({'persistent_memory_created': True})
        await asyncio.wait_for(drain, 1)
        self.assertFalse(agent._MEMORY_WRITES)
        self.assertFalse(agent._REMEMBER_TASKS)
        agent.hot_path_exit(token)


if __name__ == "__main__":
    unittest.main()
