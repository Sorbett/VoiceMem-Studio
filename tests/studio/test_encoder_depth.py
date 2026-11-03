"""Published JEV delegation, host caching, cancellation and adapter compatibility."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

from studio.core.utils.reply_modes.component import FAST, SLOW, ThinkingDecision
from studio.core.utils.reply_modes.encoder_depth import EncoderDepthRouter, _runtime_api


class AdapterTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name).resolve() / 'onnx'
        self.runtime = types.SimpleNamespace(
            meta={'prototype': False, 'report': {'trained_audio_state': True}},
            temperature=.6, model=None, tokenizer=None)

        def load():
            self.runtime.model, self.runtime.tokenizer = object(), object()
        self.runtime._load = Mock(side_effect=load)
        self.runtime.predict = Mock(return_value=types.SimpleNamespace(
            label='deep', p_ordinary=.2, p_deep=.8))
        self.api = types.SimpleNamespace(OnnxRouter=Mock(return_value=self.runtime),
                                         Router=Mock(return_value=self.runtime))
        loader = patch('studio.core.utils.reply_modes.encoder_depth._runtime_api', return_value=self.api)
        loader.start()
        self.addCleanup(loader.stop)

    def test_sdk_owns_prediction_and_cpu_inference_never_acquires_a_torch_device(self):
        router = EncoderDepthRouter(self.path, backend='onnx')
        history = [{'role': 'user', 'content': 'Synthetic context'}]
        with patch('voicemem.engine.device_lock', side_effect=AssertionError('CPU must not wait on Torch')):
            first = router.classify('Synthetic request', history=history, audio_state={'span_s': 1})
            self.assertIs(router.classify('Synthetic request', True, history, audio_state={'span_s': 1}), first)
            router.classify('Synthetic request', history=history, audio_state={'span_s': 2})
        self.api.OnnxRouter.assert_called_once_with(self.path)
        self.api.Router.assert_not_called()
        self.assertEqual(first.level, SLOW)
        self.assertEqual(self.runtime.predict.call_count, 2)
        self.runtime._load.assert_called_once()
        self.runtime.predict.assert_called_with('Synthetic request', history=history, observations={'span_s': 2})

    def test_eager_uses_the_published_runtime_on_the_selected_device(self):
        router = EncoderDepthRouter(self.path, backend='eager', device='cuda:2')
        self.api.Router.assert_called_once_with(self.path, backend='eager', device='cuda:2')
        self.api.OnnxRouter.assert_not_called()
        self.assertEqual(router.device, 'cuda:2')
        with self.assertRaisesRegex(ValueError, 'device=cpu'):
            EncoderDepthRouter(self.path, backend='onnx', device='cuda:2')

    def test_prototype_and_audio_opt_in_remain_host_boundaries(self):
        self.runtime.meta['prototype'] = True
        with self.assertRaisesRegex(ValueError, 'allow_prototype'):
            EncoderDepthRouter(self.path, backend='onnx')
        self.runtime.meta['report']['trained_audio_state'] = False
        router = EncoderDepthRouter(self.path, backend='onnx', allow_prototype=True)
        router.classify('Synthetic request', audio_state={'span_s': 2})
        self.runtime.predict.assert_called_once_with('Synthetic request', history=None, observations=None)
        self.runtime.meta['report']['trained_audio_state'] = 1
        with self.assertRaisesRegex(ValueError, 'audio-state'):
            EncoderDepthRouter(self.path, backend='onnx', allow_prototype=True)

    def test_empty_text_and_invalid_decisions_never_enter_native_work_or_cache(self):
        router = EncoderDepthRouter(self.path, backend='onnx')
        self.assertEqual(router.classify(' ').level, FAST)
        self.runtime._load.assert_not_called()
        for label, ordinary, deep in [('unknown', .2, .8), ('deep', float('nan'), .8),
                                      ('deep', .2, 1.2), ('deep', .8, .8)]:
            self.runtime.predict.return_value = types.SimpleNamespace(label=label, p_ordinary=ordinary, p_deep=deep)
            with self.subTest(label=label, ordinary=ordinary, deep=deep):
                with self.assertRaisesRegex(ValueError, 'decision'):
                    router.classify('Synthetic request')
                self.assertFalse(router._cache)

    def test_runtime_failures_propagate_without_silently_changing_models(self):
        router = EncoderDepthRouter(self.path, backend='onnx')
        self.runtime.predict.side_effect = RuntimeError('synthetic runtime failure')
        with self.assertRaisesRegex(RuntimeError, 'synthetic runtime failure'):
            router.classify('Synthetic request')
        self.assertFalse(router._cache)

    def test_published_subclasses_can_keep_their_existing_load_hook(self):
        external_model, external_tokenizer = object(), object()

        class ExternalAdapter(EncoderDepthRouter):
            def _load(self):
                self._model, self._tokenizer = external_model, external_tokenizer
        adapter = ExternalAdapter(self.path, backend='onnx')
        self.assertEqual(adapter.classify('Synthetic request').level, SLOW)
        self.runtime._load.assert_not_called()
        self.assertIs(self.runtime.model, external_model)
        self.assertIs(self.runtime.tokenizer, external_tokenizer)


class RuntimeImportTests(unittest.TestCase):
    def test_downloaded_package_is_imported_without_persisting_a_python_path_override(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules):
            model = Path(root) / 'onnx'
            package = Path(root) / 'python' / 'voicemem_jev'
            package.mkdir(parents=True)
            (package/'__init__.py').write_text('synthetic_marker = True\n')
            sys.modules.pop('voicemem_jev', None)
            before = list(sys.path)
            self.assertTrue(_runtime_api(model).synthetic_marker)
            self.assertEqual(sys.path, before)

    def test_different_loaded_runtime_is_not_mixed_with_another_bundle(self):
        with tempfile.TemporaryDirectory() as root:
            package = Path(root) / 'python' / 'voicemem_jev'
            package.mkdir(parents=True)
            loaded = types.ModuleType('voicemem_jev')
            loaded.__file__ = str(Path(root)/'other'/'__init__.py')
            with patch.dict(sys.modules, {'voicemem_jev': loaded}):
                with self.assertRaisesRegex(RuntimeError, 'restart Studio'):
                    _runtime_api(Path(root)/'onnx')


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
