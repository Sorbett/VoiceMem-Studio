"""Offline checks for CPU recognition and accelerator lock ownership."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from voicemem import gate
from voicemem.stream import VoiceStream
from voicemem.utils.audio.asr import FunASRStreamingASR, Transcriber
from voicemem.utils.audio.environment.environment_detector_ast import ASTEnvironmentDetector
from voicemem.engine import TORCH_LOCK, device_lock


class InferenceSchedulingTests(unittest.TestCase):
    def ast(self, device, forward):
        vector = types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: np.ones(2)))
        torch = types.SimpleNamespace(inference_mode=nullcontext, sigmoid=lambda _: vector)
        self.enterContext(patch.dict('sys.modules', {'torch': torch}))
        detector = ASTEnvironmentDetector(device=device)
        detector._model = forward
        detector._processor = lambda *_a, **_k: {}
        detector._load_audio = lambda _: np.ones(160, np.float32)
        output = types.SimpleNamespace(logits=[0], hidden_states=[[
            types.SimpleNamespace(mean=lambda **_: vector)]])
        return detector, output

    def test_cpu_scene_inference_does_not_block_streaming_asr(self):
        entered, release = threading.Event(), threading.Event()
        def forward(**_):
            self.assertFalse(TORCH_LOCK._is_owned())
            entered.set()
            if not release.wait(2):
                raise TimeoutError('synthetic AST work was not released')
            return output
        detector, output = self.ast('cpu', forward)
        asr = object.__new__(FunASRStreamingASR)
        asr._cache, asr._text = {}, ''
        asr.model = types.SimpleNamespace(generate=lambda **_: [{'text': 'streaming fixture'}])
        with ThreadPoolExecutor(max_workers=2) as pool:
            scene = pool.submit(detector._infer, 'synthetic-audio')
            try:
                self.assertTrue(entered.wait(1))
                self.assertEqual(pool.submit(asr._run, np.ones(10), False).result(timeout=1),
                                 'streaming fixture')
                self.assertFalse(scene.done())
            finally:
                release.set()
            self.assertIsNotNone(scene.result(timeout=1))

    def test_scene_model_keeps_its_private_serialization_and_gpu_guard(self):
        entered, release, second_entered = (threading.Event() for _ in range(3))
        calls = []
        def forward(**_):
            calls.append(True)
            if len(calls) == 1:
                entered.set()
                release.wait(2)
            else:
                second_entered.set()
            return output
        detector, output = self.ast('cpu', forward)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(detector._infer, 'fixture')
            try:
                self.assertTrue(entered.wait(1))
                second = pool.submit(detector._infer, 'fixture')
                self.assertFalse(second_entered.wait(.02))
            finally:
                release.set()
            self.assertIsNotNone(first.result(timeout=1))
            self.assertIsNotNone(second.result(timeout=1))
        for device in ('mps', 'cuda:0'):
            def gpu_forward(**_):
                self.assertTrue(device_lock(device)._is_owned())
                return output
            detector, output = self.ast(device, gpu_forward)
            self.assertIsNotNone(detector._infer('fixture'))

    def test_concurrent_cpu_scene_loads_construct_one_model_without_gpu_lock(self):
        detector = ASTEnvironmentDetector(device='cpu')
        model = Mock()
        model.to.return_value = model.eval.return_value = model
        model.config = types.SimpleNamespace(num_labels=1, id2label={0: 'fixture'})
        factory, processor = Mock(return_value=model), Mock(return_value=object())
        module = types.SimpleNamespace(
            ASTForAudioClassification=types.SimpleNamespace(from_pretrained=factory),
            AutoProcessor=types.SimpleNamespace(from_pretrained=processor))
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with patch.dict('sys.modules', {'transformers': module, 'torch': torch}), \
                ThreadPoolExecutor(max_workers=2) as pool:
            with TORCH_LOCK:
                one, two = pool.submit(detector._load), pool.submit(detector._load)
                one.result(timeout=1)
                two.result(timeout=1)
        factory.assert_called_once()
        processor.assert_called_once()

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
            self.assertTrue(device_lock(device)._is_owned())
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
