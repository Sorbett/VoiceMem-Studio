"""Exercise MLX routing and speech on the real scheduler with synthetic work."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from studio.core.utils.reply_modes.component import FAST, SLOW
from studio.core.utils.reply_modes.mlx import MLXThinkingRouter
from studio.core.utils.tts.component import BreezeMLXTTS
from tests.helpers.gpu import cpu_gpu_loop


class LocalReplySchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_reply_uses_bounded_bridge_and_closes_on_consumer_exit(self):
        from studio.core.utils.llm import local
        llm = local.LocalLLM.__new__(local.LocalLLM)
        produced, closed = [], threading.Event()

        def tokens(*_):
            try:
                for index in range(100):
                    produced.append(index)
                    yield str(index)
            finally:
                closed.set()

        llm._gen_iter = tokens
        with cpu_gpu_loop() as gpu, patch.object(local, 'gpu_loop', return_value=gpu):
            stream = llm('synthetic input')
            self.assertEqual(await anext(stream), '0')
            await asyncio.sleep(.02)
            self.assertEqual(len(produced), 16)
            await stream.aclose()
            self.assertTrue(await asyncio.to_thread(closed.wait, 1))
            self.assertEqual([v async for v in gpu.aiter(lambda: iter(['next']))], ['next'])


class MlxRouterSchedulingTests(unittest.IsolatedAsyncioTestCase):
    def router(self):
        return MLXThinkingRouter(model='/synthetic-router')

    def long_input_runtime(self, before_forward, *, prefix_matches=True):
        router = self.router()
        caches, lengths = [], []

        def forward(tokens, *, cache):
            lengths.append(tokens.shape[1])
            caches.append(cache)
            before_forward()
            cache[0].state[0] += tokens.shape[1]
            token = 9 if tokens.shape[1] == 1 and tokens[0, 0] == 3 else 3
            logits = np.full((1, 1, 10), -1000.0)
            logits[0, -1, token] = 0
            return logits

        def tokenize(messages, *, add_generation_prompt, **kwargs):
            tail = [5] * 600 if add_generation_prompt else [5 if not messages[-1]['content'] else 6]
            prefix_token = 2 if add_generation_prompt and not prefix_matches else 1
            return [prefix_token] * 400 + tail

        tokenizer = types.SimpleNamespace(apply_chat_template=tokenize,
                                         eos_token_ids={9}, decode=lambda _: '否')
        router._load_mlx = lambda: (tokenizer, forward)
        cache_module = types.ModuleType('mlx_lm.models.cache')
        cache_module.make_prompt_cache = lambda _: [types.SimpleNamespace(state=np.array([0]))]
        modules = {'mlx_lm': types.ModuleType('mlx_lm'),
                   'mlx_lm.models': types.ModuleType('mlx_lm.models'),
                   'mlx_lm.models.cache': cache_module}
        return router, caches, lengths, modules

    async def test_sync_router_yields_to_audio_during_prefix_and_bounds_request_prefill(self):
        entered, release = threading.Event(), threading.Event()
        order = []

        def before_forward():
            order.append('forward')
            if len(order) == 1:
                entered.set()
                release.wait(2)

        router, caches, lengths, modules = self.long_input_runtime(before_forward)

        def audio():
            order.append('audio')
            yield b'pcm'

        with cpu_gpu_loop() as gpu, patch.dict('sys.modules', modules), \
                patch('voicemem.engine.gpu_loop', return_value=gpu):
            task = asyncio.create_task(asyncio.to_thread(router.classify, 'synthetic input'))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                speech = gpu.iter(audio, solo_first=True)
            finally:
                release.set()
            self.assertEqual((await task).level, FAST)
            self.assertEqual(speech.out.get(timeout=1), ('ok', b'pcm'))
            self.assertIsNone(speech.out.get(timeout=1))
        self.assertEqual(order[:3], ['forward', 'audio', 'forward'])
        self.assertLessEqual(max(lengths), 128)
        np.testing.assert_array_equal(router._prefix_cache[0].state, [400])
        np.testing.assert_array_equal(caches[-1][0].state, [1001])

    async def test_cancelled_prefix_stays_private_and_next_request_rebuilds_it(self):
        entered, release = threading.Event(), threading.Event()

        def before_forward():
            if not entered.is_set():
                entered.set()
                release.wait(2)

        router, caches, lengths, modules = self.long_input_runtime(before_forward)
        with cpu_gpu_loop() as gpu, patch.dict('sys.modules', modules), \
                patch('voicemem.engine.gpu_loop', return_value=gpu):
            task = asyncio.create_task(router.classify_async('cancelled input'))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
            await asyncio.to_thread(gpu.call, lambda: None)
            self.assertEqual(lengths, [128])
            self.assertIsNone(router._prefix_key)
            self.assertIsNone(router._prefix_cache)
            self.assertEqual(router._cache, {})
            self.assertEqual((await router.classify_async('replacement input')).level, FAST)
            np.testing.assert_array_equal(router._prefix_cache[0].state, [400])
            self.assertIsNot(caches[0], router._prefix_cache)

    async def test_token_prefix_mismatch_uses_bounded_private_fallback(self):
        router, caches, lengths, modules = self.long_input_runtime(
            lambda: None, prefix_matches=False)
        with cpu_gpu_loop() as gpu, patch.dict('sys.modules', modules), \
                patch('voicemem.engine.gpu_loop', return_value=gpu):
            self.assertEqual((await router.classify_async('synthetic input')).level, FAST)
        self.assertLessEqual(max(lengths), 128)
        np.testing.assert_array_equal(router._prefix_cache[0].state, [400])
        np.testing.assert_array_equal(caches[-1][0].state, [1001])
        self.assertIsNot(caches[-1], router._prefix_cache)

    async def test_queued_last_waiter_cancellation_skips_model_and_cache(self):
        entered, release = threading.Event(), threading.Event()
        router = self.router()
        router._prediction_steps = Mock(return_value=iter(['否']))

        def occupied():
            entered.set()
            release.wait(2)
            yield None

        with cpu_gpu_loop() as gpu, patch('voicemem.engine.gpu_loop', return_value=gpu):
            blocker = gpu.iter(occupied)
            self.assertTrue(entered.wait(1))
            task = asyncio.create_task(router.classify_async('synthetic input'))
            try:
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                self.assertEqual(gpu._jobs.qsize(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
            self.assertIsNotNone(blocker.out.get(timeout=1))
            self.assertIsNone(blocker.out.get(timeout=1))
            self.assertEqual([v async for v in gpu.aiter(lambda: iter(['next']))], ['next'])
        router._prediction_steps.assert_not_called()
        self.assertEqual(router._cache, {})
        self.assertEqual(router._classifications, {})

    async def test_cancelled_native_step_finishes_without_following_steps_or_cache(self):
        entered, release, closed = (threading.Event() for _ in range(3))
        following = Mock()
        router = self.router()

        def steps(*args):
            try:
                entered.set()
                release.wait(2)
                yield None
                following()
                yield '是'
            finally:
                closed.set()

        router._prediction_steps = steps
        with cpu_gpu_loop() as gpu, patch('voicemem.engine.gpu_loop', return_value=gpu):
            task = asyncio.create_task(router.classify_async('synthetic input'))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(closed.is_set())
            finally:
                release.set()
            self.assertEqual([v async for v in gpu.aiter(lambda: iter(['next']))], ['next'])
        self.assertTrue(closed.is_set())
        following.assert_not_called()
        self.assertEqual(router._cache, {})

    async def test_shared_waiter_survives_and_completed_decision_is_cached(self):
        entered, release = threading.Event(), threading.Event()
        router = self.router()
        calls = []

        def steps(*args):
            calls.append(threading.get_ident())
            entered.set()
            release.wait(2)
            yield None
            yield '是'

        router._prediction_steps = steps
        with cpu_gpu_loop() as gpu, patch('voicemem.engine.gpu_loop', return_value=gpu):
            first = asyncio.create_task(router.classify_async('synthetic input'))
            second = None
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                second = asyncio.create_task(router.classify_async('synthetic input'))
                await asyncio.sleep(0)
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                self.assertFalse(second.done())
            finally:
                release.set()
            self.assertEqual((await second).level, SLOW)
            self.assertEqual((await router.classify_async('synthetic input')).level, SLOW)
            self.assertEqual(calls, [gpu._thread.ident])

    async def test_real_greedy_steps_preserve_sync_output_and_private_prefix(self):
        router = self.router()
        threads, caches = [], []

        def forward(tokens, *, cache):
            threads.append(threading.get_ident())
            caches.append(cache)
            cache[0].state[0] += tokens.shape[1]
            token = 9 if tokens.shape[1] == 1 and tokens[0, 0] == 3 else 3
            logits = np.full((1, 1, 10), -1000.0)
            logits[0, -1, token] = 0
            return logits

        tokenizer = types.SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: [1, 2, 5 if not messages[-1]['content'] else 6],
            eos_token_ids={9}, decode=lambda tokens: '否' if tokens == [3] else '')
        router._load_mlx = lambda: (tokenizer, forward)
        cache_module = types.ModuleType('mlx_lm.models.cache')
        cache_module.make_prompt_cache = lambda _: [types.SimpleNamespace(state=np.array([0]))]
        modules = {'mlx_lm': types.ModuleType('mlx_lm'),
                   'mlx_lm.models': types.ModuleType('mlx_lm.models'),
                   'mlx_lm.models.cache': cache_module}
        with cpu_gpu_loop() as gpu, patch.dict('sys.modules', modules), \
                patch('voicemem.engine.gpu_loop', return_value=gpu):
            self.assertEqual((await router.classify_async('first synthetic input')).level, FAST)
            prefix = router._prefix_cache
            np.testing.assert_array_equal(prefix[0].state, [2])
            self.assertEqual((await asyncio.to_thread(router.classify, 'second synthetic input')).level, FAST)
            self.assertIs(router._prefix_cache, prefix)
            np.testing.assert_array_equal(prefix[0].state, [2])
            self.assertEqual(set(threads), {gpu._thread.ident})
            self.assertTrue(all(cache is not prefix for cache in caches[1:]))

    async def test_provider_failure_does_not_populate_cache(self):
        router = self.router()

        def fail(*args):
            raise RuntimeError('synthetic classification failure')
            yield

        router._prediction_steps = fail
        with cpu_gpu_loop() as gpu, patch('voicemem.engine.gpu_loop', return_value=gpu):
            with self.assertRaisesRegex(RuntimeError, 'synthetic classification failure'):
                await router.classify_async('synthetic input')
            self.assertEqual(router._cache, {})
            router._prediction_steps = lambda *args: iter(['否'])
            self.assertEqual((await router.classify_async('synthetic input')).level, FAST)


class MlxSpeechSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_pcm_delivery_ignores_saturated_default_pool(self):
        entered, release = threading.Event(), threading.Event()
        executor = ThreadPoolExecutor(max_workers=1)
        asyncio.get_running_loop().set_default_executor(executor)
        blocker = executor.submit(lambda: (entered.set(), release.wait(2)))
        self.assertTrue(entered.wait(1))
        tts = BreezeMLXTTS(model='synthetic-model')
        tts._segments = lambda *args: iter([b'first', b'second'])

        async def consume():
            return [value async for value in tts.stream('synthetic input')]

        try:
            with cpu_gpu_loop() as gpu, patch('voicemem.engine.gpu_loop', return_value=gpu):
                self.assertEqual(await asyncio.wait_for(consume(), 1), [b'first', b'second'])
        finally:
            release.set()
            blocker.result(timeout=1)

    async def test_interruption_cleans_codec_before_next_segment(self):
        entered, release = threading.Event(), threading.Event()
        events = []
        tts = BreezeMLXTTS(model='synthetic-model')

        def segments(text, instruction):
            events.append(('begin', text))
            try:
                if text == 'old':
                    entered.set()
                    release.wait(2)
                yield text.encode()
            finally:
                events.append(('close', text))

        tts._segments = segments
        with cpu_gpu_loop() as gpu, patch('voicemem.engine.gpu_loop', return_value=gpu):
            old = tts.stream('old')
            task = asyncio.create_task(anext(old))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await old.aclose()
            finally:
                release.set()
            self.assertEqual([value async for value in tts.stream('new')], [b'new'])
        self.assertEqual(events, [('begin', 'old'), ('close', 'old'),
                                  ('begin', 'new'), ('close', 'new')])

    async def test_error_after_pcm_propagates_and_releases_serial_lock(self):
        tts = BreezeMLXTTS(model='synthetic-model')

        def segments(*args):
            yield b'first'
            raise ValueError('synthetic codec failure')

        tts._segments = segments
        with cpu_gpu_loop() as gpu, patch('voicemem.engine.gpu_loop', return_value=gpu):
            async with aclosing(tts.stream('synthetic input')) as stream:
                self.assertEqual(await anext(stream), b'first')
                with self.assertRaisesRegex(ValueError, 'synthetic codec failure'):
                    await anext(stream)
            tts._segments = lambda *args: iter([b'recovered'])
            self.assertEqual([value async for value in tts.stream('next')], [b'recovered'])


if __name__ == '__main__':
    unittest.main()
