"""API speech/reply cancellation and cross-loop client cleanup without network."""
import asyncio
from contextlib import aclosing
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from voicemem.tts import BaseTTS, OpenAITTS
from voicemem.reply import openai_reply


class ProviderLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_base_speech_retains_plain_async_iterator_and_partial_sample_contract(self):
        class Chunks:
            def __init__(self):
                self.parts = iter((b'\1',b'\2\3'))
            def __aiter__(self):
                return self
            async def __anext__(self):
                try:
                    return next(self.parts)
                except StopIteration:
                    raise StopAsyncIteration from None
        class Speech(BaseTTS):
            def _raw(self, text, instruction):
                return Chunks()
        self.assertEqual([chunk async for chunk in Speech().stream('hello')],[b'\1\2',b'\3\0'])

    async def test_tts_closure_and_cancellation_close_nested_stream(self):
        for cancel in (False,True):
            waiting, closed = asyncio.Event(), asyncio.Event()
            class Speech(BaseTTS):
                async def _raw(self, text, instruction):
                    try:
                        yield b'\0\0'
                        waiting.set()
                        await asyncio.Event().wait()
                    finally:
                        closed.set()
            stream = Speech().stream('hello')
            self.assertEqual(await anext(stream),b'\0\0')
            if cancel:
                task = asyncio.create_task(anext(stream))
                await asyncio.wait_for(waiting.wait(),1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            else:
                await stream.aclose()
            self.assertTrue(closed.is_set())

    async def test_speech_client_is_recreated_and_cache_identity_follows_voice_and_endpoint(self):
        speech = OpenAITTS(model='tts-model',voice='voice-a',base_url='https://api.example/v1',api_key='fixture')
        first, second = SimpleNamespace(close=AsyncMock()),SimpleNamespace(close=AsyncMock())
        with patch('openai.AsyncOpenAI',side_effect=[first,second]):
            self.assertIs(speech._cli(),first)
            await speech.aclose()
            await speech.aclose()
            self.assertIs(speech._cli(),second)
            await speech.aclose()
        first.close.assert_awaited_once_with()
        second.close.assert_awaited_once_with()
        for kwargs in ({'voice':'voice-b'},{'base_url':'https://other.example/v1'}):
            config = dict(model='tts-model',voice='voice-a',base_url='https://api.example/v1',api_key='fixture')
            config.update(kwargs)
            self.assertNotEqual(speech.cache_voice_id,OpenAITTS(**config).cache_voice_id)

    async def test_reply_probe_client_and_response_close_before_reuse(self):
        class Response:
            def __init__(self):
                self.close = AsyncMock()
            def __aiter__(self):
                return self.tokens()
            async def tokens(self):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='hello'))])
        responses = [Response(),Response()]
        clients = [SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))),close=AsyncMock()) for response in responses]
        with patch('openai.AsyncOpenAI',side_effect=clients):
            reply = openai_reply(model='fixture-model',api_key='fixture',base_url='https://api.example/v1',system='fixture')
            for response in responses:
                async with aclosing(reply('hello','',[])) as stream:
                    self.assertEqual(await anext(stream),'hello')
                await reply.aclose()
                response.close.assert_awaited_once_with()
        for client in clients:
            client.close.assert_awaited_once_with()


if __name__ == '__main__':
    unittest.main()
