"""Disposable input work must release admission without losing memory eligibility."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from voicemem.engine import inference_call
from voicemem.engine.backends.torch import InferenceLock
from voicemem.engine.scheduling.cancellation import (
    InferenceCancelled, cancellation_scope, run_cancellable,
)
from voicemem.stream import Turn, VoiceStream, empty_result


class StreamInputCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_executor_backlog_cannot_block_confirmed_input(self):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        workers = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(workers)

        def occupy_default():
            loop.call_soon_threadsafe(entered.set)
            release.wait(2)

        blocked = loop.run_in_executor(None, occupy_default)
        await entered.wait()
        stream, result = self.stream(Mock(return_value='deep'))
        try:
            turn = await asyncio.wait_for(stream._confirm(), .3)
            self.assertIs(turn.result, result)
            self.assertFalse(blocked.done())
        finally:
            release.set()
            await blocked
            workers.shutdown(wait=True)

    def stream(self, gate, *, timeout=None):
        result = SimpleNamespace(hits=[])
        memory = SimpleNamespace(
            classify=Mock(return_value=SimpleNamespace(slots=[], entities=[])),
            search=Mock(return_value=result))
        stream = VoiceStream(memory, gate=gate, gate_timeout_s=timeout)
        stream._text = 'complete input'
        return stream, result

    async def test_cancelled_speculation_never_enters_queued_native_work_or_search(self):
        lock = InferenceLock()
        entered, finished = threading.Event(), threading.Event()
        called = []

        def gate(text):
            entered.set()
            try:
                with inference_call('gate-fixture', 'torch', lock=lock):
                    called.append(text)
                    return 'deep'
            finally:
                finished.set()

        stream, _ = self.stream(gate)
        lock.acquire()
        task = asyncio.create_task(stream._speculate('obsolete input'))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            lock.release()
        self.assertTrue(await asyncio.to_thread(finished.wait, 1))
        self.assertEqual(called, [])
        stream.vm.classify.assert_not_called()
        stream.vm.search.assert_not_called()

    async def test_confirmed_deep_route_is_not_classified_again(self):
        gate = Mock(return_value='deep')
        stream, result = self.stream(gate)
        turn = await stream._confirm()
        self.assertIs(turn.result, result)
        gate.assert_called_once_with('complete input')
        stream.vm.search.assert_called_once()

    async def test_deadline_preserves_memory_and_does_not_repeat_the_slow_gate(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        calls = []

        def gate(text):
            calls.append(text)
            entered.set()
            try:
                release.wait(2)
                return 'shallow'
            finally:
                finished.set()

        stream, result = self.stream(gate, timeout=.03)
        detail = {}
        try:
            turn = await asyncio.wait_for(stream._confirm(timing=detail), .3)
            self.assertTrue(entered.is_set())
            self.assertEqual(turn.route, 'deep')
            self.assertIs(turn.result, result)
            stream.vm.search.assert_called_once()
            self.assertEqual(detail['gate_timeout'], 1)
            self.assertEqual(len(calls), 1)
        finally:
            release.set()
            await asyncio.to_thread(finished.wait, 1)
        self.assertEqual(stream._route, 'deep')

    async def test_expired_prefetch_gate_keeps_the_same_total_wait_budget(self):
        release, finished = threading.Event(), threading.Event()

        def gate(_):
            try:
                release.wait(2)
                return 'shallow'
            finally:
                finished.set()

        stream, result = self.stream(gate, timeout=.02)
        stream._gate_pre_text = stream._text
        pre = stream._gate_pre = asyncio.create_task(stream._input_work(
            lambda: stream._gate_locked(stream._text),
            timing=stream._gate_pre_timing, label='gate'))
        detail = {}
        try:
            turn = await asyncio.wait_for(stream._confirm(timing=detail), .3)
            self.assertIs(turn.result, result)
            self.assertTrue(pre.cancelled())
            self.assertEqual(detail['gate_timeout'], 1)
        finally:
            release.set()
            await asyncio.to_thread(finished.wait, 1)

    async def test_confirmed_deep_route_upgrades_a_completed_shallow_prefetch(self):
        stream, result = self.stream(Mock(return_value='deep'))
        stream._spec = asyncio.create_task(asyncio.sleep(
            0, result=Turn(stream._text, empty_result(), route='shallow')))
        turn = await stream._confirm()
        self.assertEqual(turn.route, 'deep')
        self.assertIs(turn.result, result)
        stream.vm.search.assert_called_once()

    async def test_cancelling_confirmation_cannot_restart_retrieval(self):
        stream, _ = self.stream(Mock(return_value='deep'))
        waiting = asyncio.Event()

        async def speculative():
            waiting.set()
            await asyncio.Event().wait()

        stream._spec = asyncio.create_task(speculative())
        await waiting.wait()
        task = asyncio.create_task(stream._confirm())
        await asyncio.sleep(.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        stream.vm.search.assert_not_called()

    async def test_cancelled_prefetch_can_retry_without_repeating_confirmed_gate(self):
        gate = Mock(return_value='deep')
        stream, result = self.stream(gate)
        stream._spec = asyncio.create_task(asyncio.sleep(10))
        stream._spec.cancel()
        await asyncio.gather(stream._spec, return_exceptions=True)
        turn = await stream._confirm()
        self.assertIs(turn.result, result)
        gate.assert_called_once_with(stream._text)
        stream.vm.search.assert_called_once()

    async def test_worker_keeps_parent_cancellation_predicate(self):
        cancelled = threading.Event()
        cancelled.set()
        operation = Mock()
        with cancellation_scope(cancelled.is_set):
            with self.assertRaises(InferenceCancelled):
                await run_cancellable(operation)
        operation.assert_not_called()


if __name__ == '__main__':
    unittest.main()
