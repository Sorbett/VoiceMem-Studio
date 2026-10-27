"""Component configuration and adapter contracts without models or network calls."""
import asyncio
from contextlib import aclosing
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from studio.core.utils.component_config import component as config
from studio.core.utils.component_config.contracts import checked


class ConfigurationTests(unittest.TestCase):
    def test_recommended_defaults_and_non_secret_public_contracts(self):
        with patch.dict(os.environ, {}, clear=True):
            for role in config.catalog():
                self.assertEqual(config.selection(role)['provider'], 'recommended')
        raw = json.dumps({'speech': {'provider': 'openai'}})
        with patch.dict(os.environ, {'STUDIO_COMPONENTS': raw, 'STUDIO_SPEECH_API_KEY': 'speech-fixture'}, clear=True):
            public = config.public_selections()
            self.assertEqual(config.selection('speech')['apiKey'], 'speech-fixture')
        self.assertNotIn('speech-fixture', json.dumps(public))
        self.assertNotIn('apiKey', public['speech'])

    def test_invalid_configuration_is_not_silently_ignored(self):
        for raw in ('[1]', '{', '{"unknown":{}}', '{"asr":{"provider":"unknown"}}',
                    '{"asr":{"provider":"deepgram","apiKey":"fixture"}}'):
            with self.subTest(raw=raw), patch.dict(os.environ, {'STUDIO_COMPONENTS': raw}, clear=True):
                with self.assertRaises(ValueError):
                    config.selection('asr')
        for options in ([], {'api_key':'fixture'}, {'nested':{'authorization':'fixture'}}, {'value':float('nan')}):
            with self.assertRaises(ValueError):
                config.validate('asr', {'provider':'adapter','factory':'example:create','options':options})
        valid = config.validate('router', {'provider':'adapter','factory':'example:create','options':{'max_tokens':40}})
        self.assertEqual(valid['options'], {'max_tokens':40})

    def test_endpoint_policy_matches_desktop_and_rejects_embedded_secrets(self):
        for url in ('http://external.example/v1', 'https://user:pass@api.example', 'https://api.example?key=fixture', 'file:///tmp/api'):
            with self.assertRaises(ValueError):
                config.endpoint(url)
        self.assertEqual(config.endpoint('http://127.0.0.1:1234/v1/'), 'http://127.0.0.1:1234/v1')

    def test_selected_factory_receives_only_its_configuration_and_is_checked(self):
        module = ModuleType('fixture_adapter')
        module.create = Mock(return_value=SimpleNamespace(transcribe=lambda audio:'transcript'))
        value = config.validate('asr_final', {'provider':'adapter','factory':'fixture_adapter:create','model':'custom','options':{'precision':'fp32'}})
        value['apiKey'] = 'adapter-fixture'
        with patch.dict('sys.modules', {'fixture_adapter':module}):
            adapter = config.custom('asr_final', value)
        self.assertEqual(adapter.transcribe([]), 'transcript')
        module.create.assert_called_once_with(model='custom',base_url='',api_key='adapter-fixture',options={'precision':'fp32'})
        self.assertFalse(hasattr(adapter,'new_stream'))

    def test_custom_speech_cache_is_scoped_to_model_options_and_factory(self):
        module = ModuleType('fixture_speech')
        async def stream(text, instruction):
            yield b'\0\0'
        module.create = lambda **kwargs: SimpleNamespace(stream=stream)
        identities = []
        with patch.dict('sys.modules',{'fixture_speech':module}):
            for voice in ('one','two'):
                value = config.validate('speech',{'provider':'adapter','factory':'fixture_speech:create','options':{'voice':voice}})
                identities.append(config.custom('speech',value).cache_voice_id)
        self.assertNotEqual(*identities)

    def test_custom_model_manifest_skips_only_replaced_artifacts(self):
        from studio.core.utils.cli.component import parse_args
        from studio.core.utils.models.initialize import models
        values = {role:{'provider':'adapter','factory':'example:create'} for role in config.catalog()}
        with patch.dict(os.environ, {'STUDIO_COMPONENTS':json.dumps(values)}, clear=True):
            selected = {model.name for model in models(parse_args(['--backend','mlx','--llm','openai']))}
        self.assertTrue({'E5 记忆向量','SenseVoice 感知','AST 环境感知','声纹','emotion2vec'} <= selected)
        self.assertFalse({'Breeze MLX','VoiceMem-Jev','Silero VAD','SenseVoice 最终转写','中英双语流式 ASR','Smart Turn EOT'} & selected)

    def test_credentials_follow_active_components_and_empty_role_does_not_reuse_another_key(self):
        from studio.core.utils.cli.component import parse_args
        from studio.core.utils.startup.initialize import required_credentials
        from studio.core.utils.llm.initialize import credential
        values = {'asr':{'provider':'deepgram'},'speech':{'provider':'openai'}}
        with patch.dict(os.environ, {'STUDIO_COMPONENTS':json.dumps(values),'STUDIO_ASR_API_KEY':'asr-fixture',
                                    'STUDIO_SPEECH_API_KEY':'speech-fixture','VOICEMEM_STUDIO_API_KEY':'',
                                    'OPENAI_API_KEY':'other-fixture'}, clear=True):
            self.assertEqual(credential('openai','reply'), '')
            required = required_credentials(parse_args(['--backend','mlx','--llm','openai']))
            preparation = required_credentials(parse_args(['--backend','mlx','--llm','openai']),stage='memory')
        self.assertEqual({entry[0] for entry in required if entry[1] in {'deepgram','openai'}},
                         {'VOICEMEM_MEMORY_API_KEY / OPENAI_API_KEY','VOICEMEM_STUDIO_API_KEY / OPENAI_API_KEY','STUDIO_ASR_API_KEY','STUDIO_SPEECH_API_KEY'})
        self.assertIn('STUDIO_ASR_API_KEY',{entry[0] for entry in preparation})
        self.assertNotIn('STUDIO_SPEECH_API_KEY',{entry[0] for entry in preparation})


class VADBridgeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        config._vad.cache_clear()
        self.addCleanup(config._vad.cache_clear)

    def open_memory(self):
        from studio.core.voicemem import open_memory
        from voicemem.orchestrator import Utils

        def memory(**kwargs):
            return SimpleNamespace(utils=Utils(
                'multi_modal', None, Path(self.directory.name), {'vad': kwargs['vad']}))

        with patch('studio.core.voicemem.VoiceMem', side_effect=memory), \
                patch('studio.core.utils.llm.initialize.create', return_value=object()), \
                patch('studio.core.utils.llm.initialize.credential', return_value='fixture'):
            return open_memory({'mode': 'text_mode',
                                'reply': {'llm': {'provider': 'deepseek', 'config': {'system': 'shared'}}}})

    def test_recommended_vad_resolves_lazily_and_passes_the_actual_warmup(self):
        from studio.core.voicemem import memory_warmups
        detector = SimpleNamespace(is_speech=Mock(return_value=False))
        with patch.dict(os.environ, {}, clear=True), \
                patch('voicemem.utils.audio.stream_io.make_vad', return_value=detector) as create:
            memory = self.open_memory()
            create.assert_not_called()
            dict(memory_warmups(memory))['VAD']()
            self.assertIs(memory.utils.get('vad'), detector)
            self.assertIs(config.vad(), detector)
            create.assert_called_once_with()
        samples = detector.is_speech.call_args.args[0]
        np.testing.assert_array_equal(samples, np.zeros(512, dtype=np.float32))

    def test_custom_vad_resolves_through_the_bridge_and_keeps_connection_state_private(self):
        from studio.core.voicemem import memory_warmups
        from voicemem.stream import VoiceStream
        children = []

        def new_stream():
            child = SimpleNamespace(is_speech=lambda samples: False, new_stream=new_stream)
            children.append(child)
            return child

        module = ModuleType('fixture_vad')
        module.create = Mock(return_value=SimpleNamespace(is_speech=lambda samples: False, new_stream=new_stream))
        values = {'vad': {'provider': 'adapter', 'factory': 'fixture_vad:create'}}
        with patch.dict(os.environ, {'STUDIO_COMPONENTS': json.dumps(values)}, clear=True), \
                patch.dict('sys.modules', {'fixture_vad': module}):
            memory = self.open_memory()
            module.create.assert_not_called()
            dict(memory_warmups(memory))['VAD']()
            first, second = VoiceStream(memory).vad, VoiceStream(memory).vad
            self.assertIsNot(first, second)
            self.assertIsNot(first.adapter, second.adapter)
            self.assertIs(first.adapter, children[0])
            self.assertIs(second.adapter, children[1])
            module.create.assert_called_once_with(model='', base_url='', api_key='', options={})

    def test_vad_construction_failure_still_fails_warmup(self):
        from studio.core.voicemem import memory_warmups
        with patch.dict(os.environ, {}, clear=True), \
                patch('voicemem.utils.audio.stream_io.make_vad', side_effect=RuntimeError('synthetic VAD failure')):
            memory = self.open_memory()
            with self.assertRaisesRegex(RuntimeError, 'synthetic VAD failure'):
                dict(memory_warmups(memory))['VAD']()


class ContractTests(unittest.IsolatedAsyncioTestCase):
    def test_missing_methods_and_shared_stream_state_are_rejected(self):
        with self.assertRaisesRegex(ValueError, '接口'):
            checked('asr', object())
        raw = SimpleNamespace(feed=lambda x:'text',flush=lambda:'text',reset=lambda:None)
        raw.new_stream = lambda: raw
        with self.assertRaisesRegex(ValueError, 'independent'):
            checked('asr',raw).new_stream()
        raw.new_stream = lambda: SimpleNamespace(feed=lambda x:'other',flush=lambda:'other',reset=lambda:None,new_stream=lambda:None)
        self.assertEqual(checked('asr',raw).new_stream().feed([]), 'other')
        raw.feed = lambda x: {'text':'invalid'}
        with self.assertRaises(TypeError):
            checked('asr',raw).feed([])

    def test_eot_probability_and_final_text_outputs_are_checked(self):
        for score in (float('nan'), -1, 2):
            adapter = checked('eot', SimpleNamespace(threshold=.5,score=lambda audio:score))
            with self.assertRaises(ValueError):
                adapter.score([])
        with self.assertRaises(ValueError):
            checked('eot',SimpleNamespace(threshold=2,score=lambda audio:.5))
        with self.assertRaises(TypeError):
            checked('asr_final',SimpleNamespace(transcribe=lambda audio:None)).transcribe([])

    async def test_speech_cancellation_closes_the_underlying_generator(self):
        closed = asyncio.Event()
        async def stream(text, instruction):
            try:
                yield b'\0\0'
                await asyncio.Event().wait()
            finally:
                closed.set()
        adapter = checked('speech', SimpleNamespace(stream=stream))
        async with aclosing(adapter.stream('hello')) as chunks:
            self.assertEqual(await anext(chunks), b'\0\0')
        self.assertTrue(closed.is_set())

    async def test_speech_invalid_samples_and_sample_rate_are_rejected(self):
        from voicemem.audio_timing import TimedAudioChunk
        for chunk in (b'x', 'text', TimedAudioChunk(pcm=b'\0\0',sample_rate=16000)):
            async def stream(text, instruction):
                yield chunk
            with self.assertRaises((TypeError,ValueError)):
                async for _ in checked('speech',SimpleNamespace(stream=stream)).stream('hello'):
                    pass

    async def test_router_keeps_classification_and_spoken_filler_contracts(self):
        from studio.core.utils.reply_modes.component import ThinkingDecision
        async def classify(*args, **kwargs):
            return ThinkingDecision('slow')
        async def filler(*args, **kwargs):
            return 'Let me think.'
        router = checked('router',SimpleNamespace(classify_async=classify,generate_short_text_async=filler,warmup=lambda:None))
        self.assertEqual((await router.classify_async('question')).level,'slow')
        self.assertEqual(await router.generate_short_text_async('system','prompt'),'Let me think.')
        with self.assertRaisesRegex(ValueError, 'generate_short_text_async'):
            checked('router',SimpleNamespace(classify_async=classify,warmup=lambda:None))


if __name__ == '__main__':
    unittest.main()
