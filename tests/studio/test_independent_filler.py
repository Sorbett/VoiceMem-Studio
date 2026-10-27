"""Optional custom filler transport ownership and credential isolation."""
import asyncio
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from studio.core.utils.reply_modes import independent_filler


class IndependentFillerTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_and_cancellation_close_auxiliary_stream(self):
        for cancel in (False, True):
            entered = asyncio.Event()
            closed = AsyncMock()
            exited = AsyncMock()
            class Stream:
                async def __aiter__(self):
                    entered.set()
                    await asyncio.Event().wait()
                    yield None
                close = closed
            class Client:
                def __init__(self, **kwargs):
                    self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.request))
                async def request(self, **kwargs):
                    return Stream()
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    await exited()
            with self.subTest(cancel=cancel), \
                 patch.object(independent_filler, 'configuration', return_value=('openai', 'aux', 'http://127.0.0.1', 'fixture')), \
                 patch('openai.AsyncOpenAI', Client):
                task = asyncio.create_task(independent_filler.generate('system', 'prompt', timeout_s=.1))
                await asyncio.wait_for(entered.wait(), 1)
                if cancel:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                else:
                    self.assertEqual(await task, '')
                closed.assert_awaited_once()
                exited.assert_awaited_once()

    def test_different_provider_or_endpoint_does_not_reuse_reply_credentials(self):
        env = {'VOICEMEM_STUDIO_PROVIDER': 'deepseek', 'VOICEMEM_STUDIO_API_KEY': 'reply-fixture',
               'VOICEMEM_STUDIO_MODEL': 'reply-model', 'STUDIO_FILLER_PROVIDER': 'openai'}
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
            independent_filler.configuration()
        env['OPENAI_API_KEY'] = 'openai-fixture'
        with patch.dict(os.environ, env, clear=True):
            cfg = independent_filler.configuration()
            self.assertEqual((cfg[1], cfg[3]), ('gpt-4o', 'openai-fixture'))
        env['STUDIO_FILLER_BASE_URL'] = 'https://different.invalid/v1'
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
            independent_filler.configuration()

    def test_loopback_server_and_dedicated_credentials_are_supported(self):
        env = {'STUDIO_FILLER_PROVIDER': 'openai', 'STUDIO_FILLER_MODEL': 'auxiliary',
               'STUDIO_FILLER_BASE_URL': 'http://127.0.0.1:9999/v1', 'STUDIO_FILLER_API_KEY': ''}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(independent_filler.configuration(),
                             ('openai', 'auxiliary', env['STUDIO_FILLER_BASE_URL'], 'local'))
        env.update(STUDIO_FILLER_BASE_URL='https://example.invalid/v1', OPENAI_API_KEY='other')
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
            independent_filler.configuration()

    async def test_non_thinking_filler_stream_closes_without_touching_reply_settings(self):
        calls = []
        stream = SimpleNamespace(close=AsyncMock())
        async def chunks():
            for text in ('让我', '认真整理一下。', '不会读到这里'):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])
        class Stream:
            def __aiter__(self):
                return chunks()
            close = stream.close
        class Client:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.request))
            async def request(self, **kwargs):
                calls.append(kwargs)
                return Stream()
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_):
                pass
        with patch.object(independent_filler, 'configuration', return_value=('deepseek', 'aux', 'https://example.invalid', 'fixture')), \
             patch('openai.AsyncOpenAI', Client):
            self.assertEqual(await independent_filler.generate('system', 'prompt'), '让我认真整理一下。')
        self.assertEqual(calls[0]['extra_body'], {'thinking': {'type': 'disabled'}})
        stream.close.assert_awaited_once()

