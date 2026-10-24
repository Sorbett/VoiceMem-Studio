"""One CPU budget reaches every FunASR model without changing decoding inputs."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from voicemem.engine import cpu_thread_budget


class CpuBudgetTests(unittest.TestCase):
    def test_sdk_default_and_invalid_explicit_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(cpu_thread_budget(), 4)
        for value in ('0', '-1', '1.5', '', 'invalid'):
            with self.subTest(value=value), patch.dict(os.environ, {
                    'VOICEMEM_ENGINE_CPU_THREADS': value}, clear=True):
                with self.assertRaisesRegex(ValueError, 'CPU_THREADS'):
                    cpu_thread_budget()

    def test_streaming_sensevoice_and_emotion2vec_use_the_same_budget(self):
        from studio.core.utils.perception.component import emotion_inference
        from voicemem.utils.audio.asr import FunASRStreamingASR, Transcriber
        factory = Mock(side_effect=lambda **kwargs: SimpleNamespace(
            kwargs=kwargs, generate=Mock(return_value=[])))
        modules = {'funasr': SimpleNamespace(AutoModel=factory),
                   'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))}
        with patch.dict(os.environ, {'VOICEMEM_ENGINE_CPU_THREADS': '2'}, clear=True), \
                patch.dict('sys.modules', modules):
            streaming = FunASRStreamingASR(model='synthetic-streaming', device='cpu')
            transcriber = Transcriber('cpu', model='synthetic-emotion')
            cache = {}
            emotion_inference(cache, 'synthetic.wav')
        self.assertEqual([call.kwargs['ncpu'] for call in factory.call_args_list], [2, 2, 2])
        self.assertEqual(factory.call_args_list[0].kwargs['model'], 'synthetic-streaming')
        self.assertEqual(factory.call_args_list[1].kwargs['model'], 'synthetic-emotion')
        self.assertEqual(streaming.STRIDE, 9600)
        self.assertFalse(transcriber.model.generate.called)
        cache['m'].generate.assert_called_once_with(
            'synthetic.wav', granularity='utterance', extract_embedding=False)
