"""GPU admission, cooperative cancellation and async output regressions."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
import threading
import time
import unittest
from unittest.mock import Mock, patch

from tests.helpers.gpu import cpu_gpu_loop
from voicemem.engine import GpuLoop, Job


class GpuCancellationTests(unittest.TestCase):
    def test_allocator_observation_failure_does_not_stop_model_delivery(self):
        with cpu_gpu_loop() as gpu, \
                patch('mlx.core.get_active_memory', create=True, side_effect=RuntimeError('fixture')), \
                patch('mlx.core.get_cache_memory', create=True, return_value=0):
            job = gpu.iter(lambda: iter([1, 2]), label="fixture")
            self.assertEqual(job.out.get(timeout=1), ("ok", 1))
            self.assertEqual(job.out.get(timeout=1), ("ok", 2))
            self.assertIsNone(job.out.get(timeout=1))
            self.assertTrue(gpu.snapshot()["sampling_unavailable"])

    def test_initialization_failure_reaches_all_callers_without_running_factories(self):
        factory = Mock()
        with cpu_gpu_loop() as gpu, patch('mlx.core.set_cache_limit',
                                          side_effect=RuntimeError('synthetic device failure')):
            for _ in range(2):
                job = gpu.iter(factory)
                kind, error = job.out.get(timeout=1)
                self.assertEqual(kind, 'err')
                self.assertIsInstance(error, RuntimeError)
                self.assertIsNone(job.out.get(timeout=1))
            factory.assert_not_called()

    def test_compute_share_uses_time_instead_of_chunk_count(self):
        entered, release, sampled = (threading.Event() for _ in range(3))
        counts = {"cheap": 0, "expensive": 0}
        snapshots = []

        def occupied():
            entered.set()
            release.wait(2)
            yield None

        def cheap():
            while True:
                time.sleep(.001)
                counts["cheap"] += 1
                yield None

        def expensive():
            while True:
                time.sleep(.004)
                counts["expensive"] += 1
                if counts["expensive"] == 20:
                    snapshots.append(counts["cheap"])
                    sampled.set()
                yield None

        with cpu_gpu_loop() as gpu:
            blocker = gpu.iter(occupied)
            self.assertTrue(entered.wait(1))
            one, four = gpu.iter(cheap), gpu.iter(expensive, weight=4)
            try:
                release.set()
                self.assertTrue(sampled.wait(2))
            finally:
                one.cancel()
                four.cancel()
                release.set()
            self.assertIsNotNone(blocker.out.get(timeout=1))
            self.assertGreaterEqual(snapshots[0], 12)
            self.assertLessEqual(snapshots[0], 36)

    def test_cancelled_queued_factory_never_runs(self):
        job, factory, active = Job(), Mock(), []
        job.cancel()
        GpuLoop._begin((factory, job, 1), active)
        factory.assert_not_called()
        self.assertEqual(active, [])
        self.assertIsNone(job.out.get(timeout=1))

    def test_cancellation_is_checked_between_weighted_steps(self):
        entered, release, closed = (threading.Event() for _ in range(3))
        extra = Mock()

        def chunks():
            try:
                yield b'first'
                entered.set()
                if not release.wait(2):
                    raise TimeoutError('synthetic GPU step was not released')
                yield b'in-flight'
                extra()
                yield b'obsolete'
            finally:
                closed.set()

        with cpu_gpu_loop() as gpu:
            job = gpu.iter(chunks, weight=4)
            try:
                self.assertEqual(job.out.get(timeout=1), ('ok', b'first'))
                self.assertTrue(entered.wait(1))
                job.cancel()
            finally:
                release.set()
            self.assertEqual(job.out.get(timeout=1), ('ok', b'in-flight'))
            self.assertIsNone(job.out.get(timeout=1))
            self.assertTrue(closed.is_set())
            extra.assert_not_called()

    def test_cleanup_failure_is_reported_and_worker_accepts_next_job(self):
        entered, release = threading.Event(), threading.Event()

        def chunks():
            try:
                entered.set()
                release.wait(2)
                yield b'in-flight'
            finally:
                raise ValueError('synthetic cleanup failure')

        with cpu_gpu_loop() as gpu:
            job = gpu.iter(chunks)
            try:
                self.assertTrue(entered.wait(1))
                job.cancel()
            finally:
                release.set()
            self.assertEqual(job.out.get(timeout=1), ('ok', b'in-flight'))
            kind, error = job.out.get(timeout=1)
            self.assertEqual(kind, 'err')
            self.assertIsInstance(error, ValueError)
            self.assertIsNone(job.out.get(timeout=1))
            next_job = gpu.iter(lambda: iter([b'next']))
            self.assertEqual(next_job.out.get(timeout=1), ('ok', b'next'))
            self.assertIsNone(next_job.out.get(timeout=1))


class AsyncGpuTests(unittest.IsolatedAsyncioTestCase):
    async def test_backpressure_bounds_generation_without_blocking_other_jobs(self):
        produced, closed = [], threading.Event()

        def chunks():
            try:
                for index in range(100):
                    produced.append(index)
                    yield index
            finally:
                closed.set()

        with cpu_gpu_loop() as gpu:
            stream = gpu.aiter(chunks, max_pending=2)
            self.assertEqual(await anext(stream), 0)
            await asyncio.sleep(.02)
            self.assertEqual(produced, [0, 1])
            self.assertEqual([v async for v in gpu.aiter(lambda: iter(['next']))], ['next'])
            self.assertEqual(produced, [0, 1])
            await stream.aclose()
            self.assertTrue(await asyncio.to_thread(closed.wait, 1))
            self.assertEqual(produced, [0, 1])

    async def test_factory_is_deferred_until_selected_and_solo_first_is_preserved(self):
        entered, release = threading.Event(), threading.Event()
        order = []

        def occupied():
            entered.set()
            release.wait(2)
            yield None

        def factory():
            order.append('factory')
            return iter(['ordinary'])

        def audio():
            order.append('audio')
            yield 'audio'

        with cpu_gpu_loop() as gpu:
            blocker = gpu.iter(occupied)
            self.assertTrue(entered.wait(1))
            ordinary = gpu.iter(factory)
            first = gpu.iter(audio, solo_first=True)
            release.set()
            await asyncio.to_thread(first.out.get, True, 1)
            await asyncio.to_thread(ordinary.out.get, True, 1)
            self.assertEqual(order, ['audio', 'factory'])
            blocker.cancel()

    async def test_output_does_not_wait_for_default_executor(self):
        entered, release = threading.Event(), threading.Event()
        executor = ThreadPoolExecutor(max_workers=1)
        asyncio.get_running_loop().set_default_executor(executor)
        blocker = executor.submit(lambda: (entered.set(), release.wait(2)))
        self.assertTrue(entered.wait(1))

        async def consume(gpu):
            return [value async for value in gpu.aiter(lambda: iter([b'one', b'two']))]

        try:
            with cpu_gpu_loop() as gpu:
                self.assertEqual(await asyncio.wait_for(consume(gpu), 1), [b'one', b'two'])
        finally:
            release.set()
            blocker.result(timeout=1)

    async def test_cancellation_before_start_skips_factory(self):
        entered, release = threading.Event(), threading.Event()
        factory = Mock(return_value=iter([b'obsolete']))

        def occupied():
            entered.set()
            release.wait(2)
            yield b'busy'

        with cpu_gpu_loop() as gpu:
            blocker = gpu.iter(occupied)
            self.assertTrue(entered.wait(1))
            stream = gpu.aiter(factory)
            task = asyncio.create_task(anext(stream))
            try:
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
            self.assertEqual(blocker.out.get(timeout=1), ('ok', b'busy'))
            self.assertIsNone(blocker.out.get(timeout=1))
            self.assertEqual([value async for value in gpu.aiter(lambda: iter([b'next']))], [b'next'])
            factory.assert_not_called()

    async def test_partial_output_error_and_close_propagate_without_orphaning(self):
        closed = threading.Event()

        def failing():
            try:
                yield b'first'
                raise RuntimeError('synthetic GPU failure')
            finally:
                closed.set()

        with cpu_gpu_loop() as gpu:
            async with aclosing(gpu.aiter(failing)) as stream:
                self.assertEqual(await anext(stream), b'first')
                with self.assertRaisesRegex(RuntimeError, 'synthetic GPU failure'):
                    await anext(stream)
            self.assertTrue(closed.is_set())
            self.assertEqual([value async for value in gpu.aiter(lambda: iter([]))], [])


if __name__ == '__main__':
    unittest.main()
