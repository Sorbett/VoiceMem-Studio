"""Bounded deferred work preserves native lifetime and ordered side effects."""
import asyncio
import unittest

from voicemem.engine import BackgroundPipeline


class BackgroundPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_preparation_overlaps_completion_with_two_active_jobs(self):
        pipeline = BackgroundPipeline(max_active=2)
        prepared = [asyncio.Event() for _ in range(3)]
        release = [asyncio.Event() for _ in range(3)]
        barriers = []

        async def prepare(index, previous):
            barriers.append(previous)
            prepared[index].set()

        def job(index):
            return asyncio.create_task(pipeline.run(
                lambda: asyncio.sleep(0), lambda previous: prepare(index, previous),
                release[index].wait))

        jobs = [job(index) for index in range(3)]
        try:
            await asyncio.wait_for(prepared[1].wait(), 1)
            self.assertTrue(prepared[0].is_set())
            self.assertFalse(prepared[2].is_set())
            self.assertTrue(barriers[0].is_set())
            self.assertFalse(barriers[1].is_set())
            release[0].set()
            await asyncio.wait_for(prepared[2].wait(), 1)
            self.assertTrue(barriers[1].is_set())
            self.assertFalse(barriers[2].is_set())
        finally:
            for event in release:
                event.set()
            await asyncio.wait_for(asyncio.gather(*jobs), 1)

    async def test_cancelled_started_job_retains_capacity_until_native_completion(self):
        pipeline = BackgroundPipeline(max_active=1)
        entered, release, next_entered = (asyncio.Event() for _ in range(3))
        first = asyncio.create_task(pipeline.run(
            lambda: asyncio.sleep(0), lambda previous: asyncio.sleep(0, result=entered.set()),
            release.wait))
        await asyncio.wait_for(entered.wait(), 1)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(pipeline.run(
            lambda: asyncio.sleep(0), lambda previous: asyncio.sleep(0, result=next_entered.set()),
            lambda: asyncio.sleep(0)))
        try:
            await asyncio.sleep(0)
            self.assertFalse(next_entered.is_set())
            release.set()
            await asyncio.wait_for(second, 1)
        finally:
            release.set()
            await asyncio.gather(*tuple(pipeline.tasks), return_exceptions=True)

    async def test_failed_preparation_does_not_bypass_earlier_write(self):
        pipeline = BackgroundPipeline(max_active=3)
        entered, release, third_entered = (asyncio.Event() for _ in range(3))
        third_barrier = []

        async def fail(previous):
            raise RuntimeError("synthetic preparation failure")

        async def third(previous):
            third_barrier.append(previous)
            third_entered.set()

        first = asyncio.create_task(pipeline.run(
            lambda: asyncio.sleep(0), lambda previous: asyncio.sleep(0, result=entered.set()),
            release.wait))
        await asyncio.wait_for(entered.wait(), 1)
        second = asyncio.create_task(pipeline.run(
            lambda: asyncio.sleep(0), fail, lambda: asyncio.sleep(0)))
        last = asyncio.create_task(pipeline.run(
            lambda: asyncio.sleep(0), third, lambda: asyncio.sleep(0)))
        try:
            await asyncio.wait_for(third_entered.wait(), 1)
            self.assertFalse(third_barrier[0].is_set())
            self.assertFalse(second.done())
            release.set()
            await asyncio.wait_for(first, 1)
            with self.assertRaisesRegex(RuntimeError, "synthetic preparation failure"):
                await asyncio.wait_for(second, 1)
            await asyncio.wait_for(last, 1)
            self.assertTrue(third_barrier[0].is_set())
        finally:
            release.set()
            await asyncio.gather(first, second, last, return_exceptions=True)

    async def test_cancelling_queued_preparation_does_not_run_or_block_following_job(self):
        pipeline = BackgroundPipeline(max_active=1)
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        first = asyncio.create_task(pipeline.run(
            lambda: asyncio.sleep(0), lambda previous: asyncio.sleep(0, result=entered.set()),
            release.wait))
        await asyncio.wait_for(entered.wait(), 1)
        waiting = asyncio.create_task(pipeline.run(
            lambda: asyncio.sleep(0), lambda previous: asyncio.sleep(0, result=calls.append("cancelled")),
            lambda: asyncio.sleep(0)))
        await asyncio.sleep(0)
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        release.set()
        await asyncio.wait_for(first, 1)
        await pipeline.run(lambda: asyncio.sleep(0),
                           lambda previous: asyncio.sleep(0, result=calls.append(previous.is_set())),
                           lambda: asyncio.sleep(0))
        self.assertEqual(calls, [True])


if __name__ == "__main__":
    unittest.main()
