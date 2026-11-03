"""Direct repository acquisition and routing startup without real weights or services."""
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from studio.core.utils.cli.component import parse_args
from studio.core.utils.models.component import Model, _digest
from studio.core.utils.models.initialize import models
from studio.core.utils.reply_modes import initialize


class DeviceBindingTests(unittest.TestCase):
    def test_formats_follow_the_router_device(self):
        cases = [('cuda', 'cuda', 'eager', 'cuda'),
                 ('cuda', 'cuda:2', 'eager', 'cuda:2'),
                 ('mlx', 'cpu', 'onnx', 'cpu'),
                 ('mlx', 'mps', 'onnx', 'cpu'),
                 ('cuda', 'cpu', 'onnx', 'cpu')]
        with patch.dict(os.environ, {}, clear=True):
            for backend, device, expected, target in cases:
                with self.subTest(backend=backend, device=device):
                    self.assertEqual(initialize.router_runtime(backend, device), (expected, target))
        with patch.dict(os.environ, {'STUDIO_BACKEND': 'cuda', 'STUDIO_DEVICE': 'cuda:3'}, clear=True):
            self.assertEqual(initialize.router_runtime(), ('eager', 'cuda:3'))
            self.assertEqual(initialize.router_runtime('mlx', 'cpu'), ('onnx', 'cpu'))

    def test_manifest_downloads_one_format_and_the_published_runtime(self):
        for backend, device, selected, other in [('mlx', 'cpu', 'onnx', 'eager'),
                                                ('cuda', 'cuda:1', 'eager', 'onnx')]:
            with self.subTest(backend=backend), patch.dict(os.environ, {}, clear=True):
                args = parse_args(['--backend', backend, '--device', device])
                manifest = models(args)
                model = next(item for item in manifest if item.name == 'VoiceMem-Jev')
                self.assertEqual(model.repository, 'Sorbett/VoiceMem-Jev')
                self.assertIsNone(model.revision)
                self.assertEqual(model.checksums, ())
                self.assertIn(f'{selected}/*', model.patterns)
                self.assertIn('python/*', model.patterns)
                self.assertIn(f'{selected}/tokenizer/tokenizer.json', model.required)
                self.assertIn('python/voicemem_jev/router.py', model.required)
                self.assertFalse(any(path.startswith(f'{other}/') for path in model.required))
                self.assertFalse(any(name in model.required for name in ('TRAINING.json', 'EVALUATION.json')))
                self.assertFalse(any('Qwen3-0.6B' in item.repository for item in manifest))

    def test_download_acquires_runtime_and_reuses_a_complete_cache_offline(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {}, clear=True):
            model = next(item for item in models(parse_args(['--backend', 'mlx'])) if item.name == 'VoiceMem-Jev')

            def download(**kwargs):
                for name in model.required:
                    path = Path(kwargs['local_dir']) / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'{}' if name.endswith('.json') else b'synthetic artifact' * 100)
            with patch('huggingface_hub.snapshot_download', side_effect=download) as fetch:
                model.acquire(Path(root)/'models', Path(root)/'legacy')
                self.assertEqual(fetch.call_args.kwargs['repo_id'], 'Sorbett/VoiceMem-Jev')
                self.assertNotIn('revision', fetch.call_args.kwargs)
                fetch.reset_mock()
                model.acquire(Path(root)/'models', Path(root)/'legacy')
                fetch.assert_not_called()
                runtime = Path(root)/'models'/model.directory/'python/voicemem_jev/router.py'
                runtime.unlink()
                self.assertFalse(model.ready(Path(root)/'models'))
                model.acquire(Path(root)/'models', Path(root)/'legacy')
                fetch.assert_called_once()
                self.assertTrue(runtime.is_file())

    def test_memory_realtime_and_custom_router_do_not_acquire_default(self):
        with patch.dict(os.environ, {}, clear=True):
            args = parse_args(['--backend', 'mlx'])
            for manifest in (models(args, 'memory'), models(parse_args(['--mode', 'realtime']))):
                self.assertNotIn('VoiceMem-Jev', {item.name for item in manifest})
        value = {'router': {'provider': 'adapter', 'factory': 'fixture:create'}}
        with patch.dict(os.environ, {'STUDIO_COMPONENTS': json.dumps(value)}, clear=True):
            self.assertNotIn('VoiceMem-Jev', {item.name for item in models(args)})

    def test_factory_matches_acquisition_and_never_loads_qwen(self):
        self.addCleanup(initialize.thinking_router.cache_clear)
        for backend, device, expected in [('mlx', 'cpu', 'onnx'), ('cuda', 'cuda:1', 'eager')]:
            initialize.thinking_router.cache_clear()
            with patch.dict(os.environ, {'STUDIO_BACKEND': backend, 'STUDIO_DEVICE': device,
                                        'STUDIO_ROUTER_BACKEND': 'mlx'}, clear=True), \
                    patch('studio.core.utils.reply_modes.encoder_depth.EncoderDepthRouter') as factory, \
                    patch.object(initialize, 'QwenThinkingRouter') as qwen, \
                    patch('huggingface_hub.snapshot_download') as download:
                self.assertIs(initialize.thinking_router(), factory.return_value)
                self.assertIs(initialize.thinking_router(), factory.return_value)
                from studio.paths import MODELS
                factory.assert_called_once_with(MODELS / initialize.DIRECTORY / expected,
                                                backend=expected, device=device, allow_prototype=True)
                qwen.assert_not_called()
                download.assert_not_called()

    def test_custom_factory_keeps_its_binding(self):
        initialize.thinking_router.cache_clear()
        self.addCleanup(initialize.thinking_router.cache_clear)
        value = {'router': {'provider': 'adapter', 'factory': 'fixture:create'}}
        with patch.dict(os.environ, {'STUDIO_COMPONENTS': json.dumps(value)}, clear=True), \
                patch('studio.core.utils.component_config.component.custom') as custom, \
                patch('studio.core.utils.reply_modes.encoder_depth.EncoderDepthRouter') as default:
            self.assertIs(initialize.thinking_router(), custom.return_value)
            self.assertEqual(custom.call_args.args[0], 'router')
            default.assert_not_called()

    def test_listing_models_does_not_import_inference_runtimes(self):
        script = """
import sys
from studio.core.utils.cli.component import parse_args
from studio.core.utils.models.initialize import models
models(parse_args(['--backend', 'mlx', '--llm', 'deepseek']))
assert not {'torch', 'onnxruntime', 'transformers', 'mlx.core', 'mlx_lm'} & sys.modules.keys()
"""
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_check_entry_does_not_download_or_start_service(self):
        from studio.core.core import main
        with patch('studio.core.utils.environment.component.load_environment'), \
                patch('studio.core.utils.environment.component.prepare'), \
                patch('studio.core.utils.startup.initialize.inspect') as inspect, \
                patch('studio.core.utils.models.initialize.acquire_all') as acquire, \
                patch('studio.core.core.build_app') as build, \
                patch.dict(os.environ, {}, clear=True):
            main(['--check', '--backend', 'mlx', '--llm', 'deepseek'])
            inspect.assert_called_once()
            acquire.assert_not_called()
            build.assert_not_called()


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / 'current'
        self.legacy = Path(directory.name) / 'legacy'
        self.files = {'onnx/model.onnx': b'synthetic graph' * 100,
                      'onnx/tokenizer/tokenizer.json': b'{"fixture":true}',
                      'onnx/depth_encoder.json': b'{"version":1}'}
        sums = tuple((name, hashlib.sha256(data).hexdigest()) for name, data in self.files.items())
        self.model = Model('Synthetic router', 'router', 'fixture/router', tuple(self.files),
                           tuple(self.files), revision='fixture-revision', checksums=sums)
        _digest.cache_clear()
        self.addCleanup(_digest.cache_clear)
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def download(self, **kwargs):
        target = Path(kwargs['local_dir'])
        for name, data in self.files.items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def test_first_download_is_pinned_and_ready_cache_is_offline(self):
        with patch('huggingface_hub.snapshot_download', side_effect=self.download) as download:
            target = self.model.acquire(self.root, self.legacy)
            self.assertEqual(target, self.root / 'router')
            self.assertTrue(self.model.ready(self.root))
            kwargs = download.call_args.kwargs
            self.assertEqual(kwargs['revision'], 'fixture-revision')
            self.assertEqual(kwargs['allow_patterns'], list(self.files))
            self.assertNotIn('force_download', kwargs)
            download.reset_mock()
            self.model.acquire(self.root, self.legacy)
            download.assert_not_called()
            self.assertFalse((target / 'eager').exists())

    def test_check_is_read_only_and_missing_tokenizer_is_not_ready(self):
        with patch('huggingface_hub.snapshot_download') as download:
            self.assertTrue(self.model.problems(self.root))
            self.assertFalse(self.root.exists())
            self.download(local_dir=self.root / 'router')
            (self.root / 'router/onnx/tokenizer/tokenizer.json').unlink()
            self.assertFalse(self.model.ready(self.root))
            self.assertIn('缺少文件：onnx/tokenizer/tokenizer.json', self.model.problems(self.root))
            download.assert_not_called()

    def test_changed_bytes_invalidate_hash_cache_and_force_repair(self):
        self.download(local_dir=self.root / 'router')
        self.assertTrue(self.model.ready(self.root))
        weight = self.root / 'router/onnx/model.onnx'
        weight.write_bytes(b'x' * weight.stat().st_size)
        self.assertIn('校验失败：onnx/model.onnx', self.model.problems(self.root))
        with patch('huggingface_hub.snapshot_download', side_effect=self.download) as download:
            self.model.acquire(self.root, self.legacy)
            self.assertTrue(download.call_args.kwargs['force_download'])
        self.assertTrue(self.model.ready(self.root))

    def test_unverified_download_never_reports_ready(self):
        def corrupt(**kwargs):
            self.download(**kwargs)
            (Path(kwargs['local_dir']) / 'onnx/model.onnx').write_bytes(b'x' * 1500)
        with patch('huggingface_hub.snapshot_download', side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, '校验失败'):
                self.model.acquire(self.root, self.legacy)

    def test_failed_partial_download_resumes_without_deleting_existing_files(self):
        def partial(**kwargs):
            target = Path(kwargs['local_dir']) / 'onnx'
            target.mkdir(parents=True)
            (target / 'model.onnx').write_bytes(self.files['onnx/model.onnx'])
            raise OSError('synthetic connection failure')
        with patch('huggingface_hub.snapshot_download', side_effect=partial):
            with self.assertRaisesRegex(RuntimeError, '下载失败'):
                self.model.acquire(self.root, self.legacy)
        self.assertEqual((self.root / 'router/onnx/model.onnx').read_bytes(), self.files['onnx/model.onnx'])
        with patch('huggingface_hub.snapshot_download', side_effect=self.download) as download:
            self.model.acquire(self.root, self.legacy)
            self.assertNotIn('force_download', download.call_args.kwargs)
        self.assertTrue(self.model.ready(self.root))

    def test_cancelled_download_remains_cancelled(self):
        with patch('huggingface_hub.snapshot_download', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.model.acquire(self.root, self.legacy)
        self.assertFalse(self.model.ready(self.root))

    def test_legacy_reuse_preserves_checksums(self):
        model = replace(self.model, legacy=('old-router',))
        self.download(local_dir=self.legacy / 'old-router')
        (self.legacy / 'old-router/onnx/model.onnx').write_bytes(b'x' * 1500)
        with patch('huggingface_hub.snapshot_download', side_effect=self.download) as download:
            model.acquire(self.root, self.legacy)
            download.assert_called_once()
        self.assertFalse((self.root / 'router').is_symlink())
        self.assertTrue(model.ready(self.root))

    def test_complete_legacy_cache_can_be_reused_offline(self):
        model = replace(self.model, legacy=('old-router',))
        self.download(local_dir=self.legacy / 'old-router')
        with patch('huggingface_hub.snapshot_download') as download:
            model.acquire(self.root, self.legacy)
            download.assert_not_called()
        self.assertTrue((self.root / 'router').is_symlink())
        self.assertTrue(model.ready(self.root))


if __name__ == '__main__':
    unittest.main()
