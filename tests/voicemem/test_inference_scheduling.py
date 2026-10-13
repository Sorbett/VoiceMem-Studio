"""Offline checks for CPU recognition and accelerator lock ownership."""
from concurrent.futures import ThreadPoolExecutor
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from voicemem import gate
from voicemem.stream import VoiceStream
from voicemem.utils.audio.asr import FunASRStreamingASR, Transcriber
from voicemem.utils.torch_lock import TORCH_LOCK


class InferenceSchedulingTests(unittest.TestCase):
    def transcriber(self, device, generate):
        factory = Mock(return_value=types.SimpleNamespace(generate=generate))
        with patch.dict('sys.modules', {'funasr': types.SimpleNamespace(AutoModel=factory)}):
            return Transcriber(device, model='synthetic-model')

    def test_cpu_emotion_does_not_block_streaming_asr_or_lexical_gate(self):
        started, release = threading.Event(), threading.Event()

        def cpu_generate(**_):
            started.set()
            if not release.wait(2):
                raise TimeoutError('synthetic CPU work was not released')
            return [{'text': '<|HAPPY|>hello'}]

        cpu = self.transcriber('cpu', cpu_generate)
        stream_asr = object.__new__(FunASRStreamingASR)
        stream_asr._cache, stream_asr._text = {}, ''

        def gpu_generate(**_):
            self.assertTrue(TORCH_LOCK._is_owned())
            return [{'text': 'streaming text'}]

        stream_asr.model = types.SimpleNamespace(generate=gpu_generate)
        stream = object.__new__(VoiceStream)
        stream.gate = gate.route
        with ThreadPoolExecutor(max_workers=3) as pool:
            cpu_job = pool.submit(cpu.run_with_emotion, 'synthetic-audio')
            try:
                self.assertTrue(started.wait(1))
                asr = pool.submit(stream_asr._run, np.zeros(10), False)
                lexical = pool.submit(stream._gate_locked, '我明天有什么安排？')
                self.assertEqual(asr.result(timeout=1), 'streaming text')
                self.assertEqual(lexical.result(timeout=1), gate.DEEP)
                self.assertFalse(cpu_job.done())
            finally:
                release.set()
            self.assertEqual(cpu_job.result(timeout=1), ('hello', '开心'))

    def test_shared_cpu_recognizer_still_serializes_its_own_requests(self):
        started, release, second_started = (threading.Event() for _ in range(3))

        def generate(**kwargs):
            if kwargs['input'] == 'first':
                started.set()
                release.wait(2)
            else:
                second_started.set()
            return [{'text': kwargs['input']}]

        cpu = self.transcriber('cpu', generate)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(cpu.run, 'first')
            try:
                self.assertTrue(started.wait(1))
                second = pool.submit(cpu.run, 'second')
                self.assertFalse(second_started.wait(.05))
            finally:
                release.set()
            self.assertEqual(first.result(timeout=1), 'first')
            self.assertEqual(second.result(timeout=1), 'second')

    def test_gpu_transcribers_and_injected_gate_keep_accelerator_lock(self):
        def generate(**_):
            self.assertTrue(TORCH_LOCK._is_owned())
            return [{'text': 'hello'}]

        for device in ('mps', 'cuda:0'):
            with self.subTest(device=device):
                self.assertEqual(self.transcriber(device, generate).run('fixture'), 'hello')
        stream = object.__new__(VoiceStream)

        def injected(text):
            self.assertTrue(TORCH_LOCK._is_owned())
            return gate.SHALLOW

        stream.gate = injected
        self.assertEqual(stream._gate_locked('fixture'), gate.SHALLOW)
        self.assertFalse(TORCH_LOCK._is_owned())

    def test_builtin_semantic_gate_keeps_encoder_serialization(self):
        from voicemem.leftbrain.local_embedder import _LockedEncoder

        def encode(*_args, **_kwargs):
            self.assertTrue(TORCH_LOCK._is_owned())
            return np.array([[1., 0.]])

        encoder = _LockedEncoder(types.SimpleNamespace(encode=encode))
        banks = (encoder, types.SimpleNamespace(query_prefix=''),
                 np.array([[1., 0.]] * 3), np.array([[0., 1.]] * 3))
        stream = object.__new__(VoiceStream)
        stream.gate = gate.route
        with patch.object(gate, '_banks', return_value=banks):
            self.assertEqual(stream._gate_locked('Explain gravity'), gate.DEEP)
        self.assertFalse(TORCH_LOCK._is_owned())

    def test_cpu_singleton_load_is_shared_without_waiting_for_gpu_lock(self):
        from voicemem.utils.audio.emotion import detector
        recognizer = object()
        factory = Mock(return_value=recognizer)
        with patch.object(detector, '_TRANSCRIBER', None), \
             patch('voicemem.utils.audio.asr.Transcriber', factory), \
             ThreadPoolExecutor(max_workers=2) as pool:
            with TORCH_LOCK:
                first = pool.submit(detector.shared_transcriber)
                second = pool.submit(detector.shared_transcriber)
                self.assertIs(first.result(timeout=1), recognizer)
                self.assertIs(second.result(timeout=1), recognizer)
            factory.assert_called_once()

    def test_cpu_failure_releases_lock_for_the_next_request(self):
        generate = Mock(side_effect=[RuntimeError('synthetic failure'), [{'text': 'recovered'}]])
        cpu = self.transcriber('cpu', generate)
        with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
            cpu.run('fixture')
        with ThreadPoolExecutor(max_workers=1) as pool:
            self.assertEqual(pool.submit(cpu.run, 'fixture').result(timeout=1), 'recovered')


if __name__ == '__main__':
    unittest.main()
