"""Ordered SDK ingestion overlaps bounded perception without losing native writes."""
import asyncio
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from studio.core.utils.memory.component import Memory
from studio.core.utils.perception.component import Perception
from voicemem.core import VoiceMem
from voicemem.lang import memory_language
from voicemem.orchestrator import Orchestrator


class MemoryPipelineTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, *, fail_preparation="", fail_write=""):
        class Agent(Memory, Perception):
            pass

        agent = Agent()
        agent._IDLE = asyncio.Event()
        agent._IDLE.set()
        agent._NATIVE_IDLE = threading.Event()
        agent._NATIVE_IDLE.set()
        agent._REMEMBER_TASKS = set()
        agent.BARGE_DEBUG = False
        agent._finish_history_turn = Mock()
        agent.ACOUSTIC_MIN_SCORE = .92
        agent.ACOUSTIC_TRUST = {"开心"}
        prepared = {name: asyncio.Event() for name in ("first", "second", "third")}
        writing = {name: asyncio.Event() for name in prepared}
        release = {name: threading.Event() for name in prepared}
        writes, outcomes = [], []
        loop = asyncio.get_running_loop()
        runtime = Orchestrator.__new__(Orchestrator)
        runtime.memory_language = "en"
        runtime.follow_input_language = True
        runtime.remember_reply = Mock()
        runtime.last_agent_reply = lambda **kwargs: "captured preceding reply"
        runtime._clap_memory_enabled = lambda: False

        def perceive(text, *args):
            loop.call_soon_threadsafe(prepared[text].set)
            if text == fail_preparation:
                raise RuntimeError("synthetic preparation failure")
            return SimpleNamespace(
                speaker="user", emotion="", environment="", environment_hint="",
                scene_tag="", scene_raw_labels=[], person_id="", tune_result=None,
                abnormal_hits=[], detection=None)

        def persist(context):
            text = context["text"]
            writes.append((text, memory_language(), context["prior_agent_reply"]))
            loop.call_soon_threadsafe(writing[text].set)
            if not release[text].wait(3):
                raise TimeoutError("synthetic write not released")
            if text == fail_write:
                raise RuntimeError("synthetic write failure")
            return {"persistent_memory_created": True, "memory_ids": [text]}

        runtime.preprocess = perceive
        runtime._finish_ingest = persist
        memory = VoiceMem.__new__(VoiceMem)
        memory._o = runtime
        agent.vm = memory

        async def cleanup():
            for event in release.values():
                event.set()
            await asyncio.wait_for(agent.drain_memory(), 2)

        self.addAsyncCleanup(cleanup)

        def enqueue(text, *, language="en", owner=None):
            pending = SimpleNamespace(text=text, audio_path=None, language=language)
            before = set(agent._REMEMBER_TASKS)
            agent.queue_remember_turn(pending, "captured heard reply", {} if owner is None else owner, text,
                                      memory, lambda result: outcomes.append((text, result)))
            return (agent._REMEMBER_TASKS - before).pop()

        return agent, enqueue, prepared, writing, release, writes, outcomes

    async def test_perception_overlaps_writes_and_acoustic_ui_has_independent_admission(self):
        agent, enqueue, prepared, writing, release, writes, outcomes = self.fixture()
        enqueue("first")
        await asyncio.wait_for(writing["first"].wait(), 1)
        enqueue("second", language="zh")
        enqueue("third")
        await asyncio.wait_for(prepared["second"].wait(), 1)
        await asyncio.sleep(.01)
        self.assertFalse(writing["second"].is_set())
        self.assertFalse(prepared["third"].is_set())
        self.assertEqual(len(agent._MEMORY_WRITES), 2)

        agent._acoustic_emotion = Mock(return_value=("开心", .99))
        send = AsyncMock()
        acoustic = agent._kick_acoustic(send, "synthetic.wav", output_id="current",
                                       is_current=lambda: True)
        await asyncio.wait_for(acoustic, 1)
        send.assert_awaited_once()
        self.assertEqual(send.call_args.args[0]["output_id"], "current")

        # A later active Space cannot redirect already-captured work.
        agent.vm = object()
        release["first"].set()
        await asyncio.wait_for(writing["second"].wait(), 1)
        await asyncio.wait_for(prepared["third"].wait(), 1)
        self.assertFalse(writing["third"].is_set())
        release["second"].set()
        await asyncio.wait_for(writing["third"].wait(), 1)
        release["third"].set()
        await asyncio.wait_for(agent.drain_memory(), 1)
        self.assertEqual(writes, [
            ("first", "en", "captured preceding reply"),
            ("second", "zh", "captured preceding reply"),
            ("third", "en", "captured preceding reply"),
        ])
        self.assertEqual([text for text, result in outcomes], ["first", "second", "third"])
        self.assertFalse(agent._MEMORY_WRITES)

    async def test_cancelled_caller_keeps_native_write_and_shutdown_ownership(self):
        agent, enqueue, prepared, writing, release, writes, outcomes = self.fixture()
        first = enqueue("first")
        await asyncio.wait_for(writing["first"].wait(), 1)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        enqueue("second")
        await asyncio.wait_for(prepared["second"].wait(), 1)
        drain = asyncio.create_task(agent.drain_memory())
        try:
            await asyncio.sleep(.01)
            self.assertFalse(drain.done())
            self.assertFalse(writing["second"].is_set())
            release["first"].set()
            await asyncio.wait_for(writing["second"].wait(), 1)
            self.assertFalse(drain.done())
            release["second"].set()
            await asyncio.wait_for(drain, 1)
            self.assertEqual([text for text, language, reply in writes], ["first", "second"])
            self.assertFalse(agent._MEMORY_WRITES)
        finally:
            for event in release.values():
                event.set()
            await asyncio.gather(drain, return_exceptions=True)

    async def test_failed_perception_reports_failure_and_preserves_following_write_order(self):
        agent, enqueue, prepared, writing, release, writes, outcomes = self.fixture(
            fail_preparation="second")
        enqueue("first")
        await asyncio.wait_for(writing["first"].wait(), 1)
        enqueue("second")
        enqueue("third")
        await asyncio.wait_for(prepared["second"].wait(), 1)
        await asyncio.sleep(.01)
        self.assertFalse(prepared["third"].is_set())
        release["first"].set()
        await asyncio.wait_for(writing["third"].wait(), 1)
        self.assertFalse(writing["second"].is_set())
        release["third"].set()
        await asyncio.wait_for(agent.drain_memory(), 1)
        self.assertEqual([text for text, language, reply in writes], ["first", "third"])
        second = next(result for text, result in outcomes if text == "second")
        self.assertTrue(second["error"])
        self.assertFalse(agent._MEMORY_WRITES)

    async def test_metadata_failure_retains_started_write_before_releasing_order(self):
        class BrokenMetadata(dict):
            def __setitem__(self, key, value):
                raise RuntimeError("synthetic metadata failure")

        agent, enqueue, prepared, writing, release, writes, outcomes = self.fixture()
        first = enqueue("first", owner=BrokenMetadata())
        await asyncio.wait_for(writing["first"].wait(), 1)
        enqueue("second")
        await asyncio.wait_for(prepared["second"].wait(), 1)
        self.assertFalse(first.done())
        self.assertFalse(writing["second"].is_set())
        release["first"].set()
        with self.assertRaisesRegex(RuntimeError, "synthetic metadata failure"):
            await asyncio.wait_for(first, 1)
        await asyncio.wait_for(writing["second"].wait(), 1)
        release["second"].set()
        await asyncio.wait_for(agent.drain_memory(), 1)
        self.assertEqual([text for text, result in outcomes], ["first", "second"])

    async def test_failed_native_write_reports_failure_and_unblocks_next_turn(self):
        agent, enqueue, prepared, writing, release, writes, outcomes = self.fixture(fail_write="first")
        enqueue("first")
        await asyncio.wait_for(writing["first"].wait(), 1)
        enqueue("second")
        await asyncio.wait_for(prepared["second"].wait(), 1)
        self.assertFalse(writing["second"].is_set())
        release["first"].set()
        await asyncio.wait_for(writing["second"].wait(), 1)
        release["second"].set()
        await asyncio.wait_for(agent.drain_memory(), 1)
        self.assertTrue(outcomes[0][1]["error"])
        self.assertTrue(outcomes[1][1]["persistent_memory_created"])
        self.assertFalse(agent._MEMORY_WRITES)


if __name__ == "__main__":
    unittest.main()
