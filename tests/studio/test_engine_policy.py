"""Studio startup chooses explicit engine policy without loading models."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from studio.core.utils.environment.component import prepare
from voicemem.engine.scheduling.capacity import BackgroundCapacity


class EnginePolicyTests(unittest.TestCase):
    def prepare(self, backend):
        prepare(SimpleNamespace(backend=backend, device='cpu', asr_device='auto',
                                tts_device='cpu', lang='zh', verbose=False))

    def test_studio_enables_capacity_even_if_engine_was_imported_before_startup(self):
        for backend in ('mlx', 'cuda'):
            capacity = BackgroundCapacity()
            with self.subTest(backend=backend), patch.dict(os.environ, {}, clear=True), \
                    patch('voicemem.engine.scheduling.capacity.background_capacity', capacity):
                self.prepare(backend)
                self.assertTrue(capacity.snapshot()['enabled'])
                self.assertEqual(os.environ['VOICEMEM_ENGINE_CPU_THREADS'],
                                 '2' if backend == 'mlx' else '4')

    def test_explicit_disable_is_respected(self):
        capacity = BackgroundCapacity(enabled=True)
        with patch.dict(os.environ, {'VOICEMEM_ENGINE_BACKGROUND_CAPACITY': '0'}, clear=True), \
                patch('voicemem.engine.scheduling.capacity.background_capacity', capacity):
            self.prepare('mlx')
        self.assertFalse(capacity.snapshot()['enabled'])

    def test_explicit_cpu_budget_is_preserved_and_invalid_values_fail_before_enabling(self):
        capacity = BackgroundCapacity()
        with patch.dict(os.environ, {'VOICEMEM_ENGINE_CPU_THREADS': '3'}, clear=True), \
                patch('voicemem.engine.scheduling.capacity.background_capacity', capacity):
            self.prepare('mlx')
            self.assertEqual(os.environ['VOICEMEM_ENGINE_CPU_THREADS'], '3')
        capacity.configure(False)
        with patch.dict(os.environ, {'VOICEMEM_ENGINE_CPU_THREADS': 'invalid'}, clear=True), \
                patch('voicemem.engine.scheduling.capacity.background_capacity', capacity):
            with self.assertRaisesRegex(ValueError, 'CPU_THREADS'):
                self.prepare('mlx')
        self.assertFalse(capacity.snapshot()['enabled'])

    def test_invalid_policy_fails_startup_without_silently_replacing_it(self):
        capacity = BackgroundCapacity()
        with patch.dict(os.environ, {'VOICEMEM_ENGINE_BACKGROUND_CAPACITY': 'invalid'}, clear=True), \
                patch('voicemem.engine.scheduling.capacity.background_capacity', capacity):
            with self.assertRaisesRegex(ValueError, 'BACKGROUND_CAPACITY'):
                self.prepare('mlx')
        self.assertFalse(capacity.snapshot()['enabled'])
