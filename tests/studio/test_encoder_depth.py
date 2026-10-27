"""Task-specific depth encoder boundaries, without downloaded weights or dialogue."""
import json
import asyncio
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import BertConfig

from studio.core.utils.reply_modes.component import FAST, SLOW, ThinkingDecision
from studio.core.utils.reply_modes.encoder_depth import EncoderDepthRouter
from studio.core.utils.reply_modes.depth_network import DepthEncoder, features


class NetworkTests(unittest.TestCase):
    def test_absent_history_cannot_change_logits_through_padded_history_tokens(self):
        torch.manual_seed(7)
        config = BertConfig(vocab_size=20, hidden_size=8, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=16)
        model = DepthEncoder(config).eval()
        ids = torch.tensor([[1, 3, 2], [1, 4, 2]])
        mask = torch.ones_like(ids)
        observed = torch.zeros(1, 7)
        with torch.inference_mode():
            first = model(ids, mask, observed)
            changed = ids.clone()
            changed[1] = torch.tensor([1, 7, 2])
            second = model(changed, mask, observed)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        seen = []
        hook = model.classifier.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
        observed[:, -1] = 1
        with torch.inference_mode():
            model(ids, mask, observed)
        hook.remove()
        self.assertGreater(float(seen[0][0, 8:16].abs().sum()), 0)

    def test_acoustic_features_include_missingness_and_history_presence(self):
        _, _, empty = features('fixture')
        self.assertEqual(empty.shape, (7,))
        np.testing.assert_array_equal(empty, np.zeros(7, np.float32))
        _, _, actual = features('fixture', [{'role': 'user', 'content': 'previous fixture'}],
                                 {'eot_probability': .8})
        self.assertEqual(actual[0], np.float32(.8))
        self.assertEqual(actual[3], 1)
        self.assertEqual(actual[-1], 1)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.metadata = {'version': 1, 'task': 'voice-depth-encoder', 'features_version': 1,
                         'labels': ['ordinary', 'deep'],
                         'prototype': False, 'temperature': 1, 'max_length': 128,
                         'report': {'trained_audio_state': True}}
        self.save()
        (self.path/'model.safetensors').write_bytes(b'fixture; loader mocked')

    def save(self):
        (self.path/'depth_encoder.json').write_text(json.dumps(self.metadata))

    def test_opt_in_and_calibration_guards(self):
        self.metadata['prototype'] = True
        self.save()
        with self.assertRaisesRegex(ValueError, 'allow_prototype'):
            EncoderDepthRouter(self.path)
        self.metadata.update(prototype=False, temperature=0)
        self.save()
        with self.assertRaisesRegex(ValueError, 'calibration'):
            EncoderDepthRouter(self.path)

    def test_onnx_uses_only_graph_and_tokenizer_and_cache_tracks_audio(self):
        (self.path/'model.onnx').write_bytes(b'fixture graph')
        (self.path/'model.safetensors').unlink()
        tokenizer = Mock(return_value={'input_ids': np.ones((2, 3), np.int64),
                                       'attention_mask': np.ones((2, 3), np.int64)})
        session = Mock()
        session.run.return_value = [np.asarray([[0., 2.]], np.float32)]
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), \
             patch('onnxruntime.InferenceSession', return_value=session), \
             patch('voicemem.engine.device_lock', side_effect=AssertionError('CPU ONNX must not wait on MPS')), \
             patch('studio.core.utils.reply_modes.depth_network.DepthEncoder') as eager:
            router = EncoderDepthRouter(self.path, backend='onnx')
            self.assertEqual(router.classify('fixture', audio_state={'span_s': 1}).level, SLOW)
            router.classify('fixture', audio_state={'span_s': 1})
            router.classify('fixture', audio_state={'span_s': 2})
        self.assertEqual(session.run.call_count, 2)
        eager.assert_not_called()
        self.assertTrue(router.prefetch_filler_text)
        with self.assertRaisesRegex(ValueError, 'device=cpu'):
            EncoderDepthRouter(self.path, backend='onnx', device='cuda:0')

    def test_empty_text_avoids_model_load_and_invalid_logits_fail(self):
        router = EncoderDepthRouter(self.path)
        self.assertEqual(router.classify(' ').level, FAST)
        router._tokenizer = lambda *args, **kwargs: {'input_ids': np.ones((2, 3), np.int64),
                                                    'attention_mask': np.ones((2, 3), np.int64)}
        router._model = lambda **kwargs: torch.tensor([[float('nan'), 0.]])
        with self.assertRaisesRegex(ValueError, 'logits'):
            router.classify('fixture')


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    def router(self, classify):
        router = EncoderDepthRouter.__new__(EncoderDepthRouter)
        router.accepts_audio_state = True
        router._jobs = {}
        router.classify = classify
        return router

    async def test_shared_request_cancellation_keeps_other_waiter_and_captured_features(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def classify(text, hint, history, *, audio_state):
            calls.append((history, audio_state))
            entered.set()
            if not release.wait(2):
                raise TimeoutError('Synthetic worker not released')
            return ThinkingDecision(FAST, 'synthetic')

        router = self.router(classify)
        history = [{'role': 'user', 'content': 'Synthetic context'}]
        audio = {'span_s': 2, 'eot_probability': .8}
        first = asyncio.create_task(router.classify_async('Synthetic input', history=history, audio_state=audio))
        await asyncio.to_thread(entered.wait, 1)
        second = asyncio.create_task(router.classify_async('Synthetic input', history=history,
                    audio_state={'eot_probability': .8, 'span_s': 2}))
        await asyncio.sleep(0)
        history[0]['content'], audio['span_s'] = 'Changed', 8
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        release.set()
        self.assertEqual((await asyncio.wait_for(second, 1)).level, FAST)
        self.assertEqual(calls, [([{'role': 'user', 'content': 'Synthetic context'}],
                                 {'span_s': 2, 'eot_probability': .8})])

    async def test_last_waiter_cancellation_stops_late_native_publication(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        published = []

        def classify(*args, **kwargs):
            from voicemem.engine.scheduling.cancellation import check_cancelled
            try:
                entered.set()
                if not release.wait(2):
                    raise TimeoutError('Synthetic worker not released')
                check_cancelled()
                published.append(True)
                return ThinkingDecision(SLOW, 'synthetic')
            finally:
                finished.set()

        router = self.router(classify)
        task = asyncio.create_task(router.classify_async('Synthetic input'))
        await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        release.set()
        self.assertTrue(await asyncio.to_thread(finished.wait, 1))
        self.assertFalse(published)
        self.assertFalse(router._jobs)

    def test_classifier_warmup_does_not_require_a_filler_api(self):
        router = self.router(Mock(return_value=ThinkingDecision(FAST)))
        with patch('studio.core.utils.reply_modes.independent_filler.configuration',
                   side_effect=AssertionError('No auxiliary API on default cached speech')):
            self.assertEqual(router.warmup().level, FAST)


if __name__ == '__main__':
    unittest.main()
