"""Decode compilation retains eager output and random-state ownership."""
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from voicemem.engine import VerifiedDecode


class ReadOnlyState:
    def __init__(self, random):
        self.random = random

    def __len__(self):
        return 1

    def __iter__(self):
        return iter((self.random.key,))

    def __getitem__(self, index):
        return (self.random.key,)[index]


class DecodeCompilationTests(unittest.TestCase):
    def setUp(self):
        self.compiling = False
        self.calls = []
        self.mx = ModuleType('mlx.core')
        self.mx.random = SimpleNamespace(key=np.array([7], dtype=np.int32))
        self.mx.random.state = ReadOnlyState(self.mx.random)
        self.mx.random.seed = lambda seed: setattr(self.mx.random, 'key', np.array([seed], dtype=np.int32))
        self.mx.eval = lambda *args: None
        self.mx.array_equal = np.array_equal
        self.mx.compile = self.compile
        self.modules = patch.dict(sys.modules, {'mlx.core': self.mx, 'mlx': ModuleType('mlx')})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def compile(self, fn, **kwargs):
        self.assertIs(kwargs['inputs'], self.mx.random.state)
        self.assertIs(kwargs['outputs'], self.mx.random.state)

        def call(*args):
            self.compiling = True
            try:
                return fn(*args)
            finally:
                self.compiling = False
        return call

    def step(self, x):
        self.calls.append('compiled' if self.compiling else 'eager')
        self.mx.random.key = self.mx.random.key + 1
        return x + self.mx.random.state[0]

    def test_verified_step_advances_rng_once_and_reuses_compilation(self):
        step = VerifiedDecode(self.step, label='synthetic_decode')
        x = np.array([2], dtype=np.int32)
        self.assertTrue(step.verify(x, seed=7))
        np.testing.assert_array_equal(self.mx.random.state[0], [7])
        np.testing.assert_array_equal(step(x), [10])
        self.assertTrue(step.enabled)
        np.testing.assert_array_equal(self.mx.random.state[0], [8])
        np.testing.assert_array_equal(step(x), [11])
        self.assertEqual(self.calls, ['eager', 'compiled', 'compiled', 'compiled'])
        np.testing.assert_array_equal(self.mx.random.state[0], [9])
        self.assertEqual(step.snapshot(), dict(verified=True, enabled=True,
                                             compiled_calls=2, eager_calls=0))

    def test_code_mismatch_disables_compilation_and_retains_eager_sampling(self):
        def fn(x):
            result = self.step(x)
            return result + int(self.compiling)

        step = VerifiedDecode(fn, label='synthetic_decode')
        with self.assertLogs('voicemem.engine.backends.mlx.compilation', level='WARNING') as logs:
            self.assertFalse(step.verify(np.array([2]), seed=7))
        self.assertIn('output_mismatch', logs.output[0])
        self.assertFalse(step.enabled)
        np.testing.assert_array_equal(self.mx.random.state[0], [7])
        np.testing.assert_array_equal(step(np.array([2])), [10])
        self.assertEqual(self.calls, ['eager', 'compiled', 'eager'])

    def test_equal_codes_with_different_rng_transition_are_rejected(self):
        def fn(x):
            self.mx.random.key = self.mx.random.key + (2 if self.compiling else 1)
            return x.copy()

        step = VerifiedDecode(fn, label='synthetic_decode')
        with self.assertLogs('voicemem.engine.backends.mlx.compilation', level='WARNING') as logs:
            self.assertFalse(step.verify(np.array([2]), seed=7))
        self.assertIn('random_state_mismatch', logs.output[0])
        np.testing.assert_array_equal(self.mx.random.state[0], [7])
        self.assertFalse(step.enabled)

    def test_compile_error_restores_eager_rng_and_reports_only_error_type(self):
        def fn(x):
            if self.compiling:
                self.mx.random.key = np.array([999])
                raise RuntimeError('private diagnostic detail')
            return self.step(x)

        step = VerifiedDecode(fn, label='synthetic_decode')
        with self.assertLogs('voicemem.engine.backends.mlx.compilation', level='WARNING') as logs:
            self.assertFalse(step.verify(np.array([2]), seed=7))
        self.assertIn('compile_error:RuntimeError', logs.output[0])
        self.assertNotIn('private diagnostic detail', logs.output[0])
        np.testing.assert_array_equal(self.mx.random.state[0], [7])
        self.assertFalse(step.enabled)

    def test_eager_failure_remains_visible(self):
        def fail(x):
            raise ValueError('synthetic eager failure')
        step = VerifiedDecode(fail, label='synthetic_decode')
        with self.assertRaisesRegex(ValueError, 'synthetic eager failure'):
            step.verify(np.array([2]), seed=7)
        self.assertFalse(step.verified)
        np.testing.assert_array_equal(self.mx.random.state[0], [7])

    def test_failure_after_verification_remains_visible(self):
        step = VerifiedDecode(self.step, label='synthetic_decode')
        x = np.array([2])
        step.verify(x, seed=7)
        def fail(*args):
            raise RuntimeError('synthetic native failure')
        step._compiled = fail
        with self.assertRaisesRegex(RuntimeError, 'synthetic native failure'):
            step(x)
        self.assertEqual(self.calls, ['eager', 'compiled'])

    def test_unverified_shape_or_dtype_uses_eager_without_new_trace(self):
        step = VerifiedDecode(self.step, label='synthetic_decode')
        step.verify(np.array([2], dtype=np.int32), seed=7)
        step(np.array([2, 3], dtype=np.int32))
        step(np.array([2], dtype=np.float32))
        self.assertEqual(self.calls, ['eager', 'compiled', 'eager', 'eager'])

    def test_unverified_call_is_eager_and_does_not_compile_or_reset_rng(self):
        step = VerifiedDecode(self.step, label='synthetic_decode')
        step(np.array([2]))
        step(np.array([2]))
        self.assertEqual(self.calls, ['eager', 'eager'])
        np.testing.assert_array_equal(self.mx.random.state[0], [9])
        self.assertFalse(step.verified)

    def test_repeated_verification_does_not_reset_runtime_rng(self):
        step = VerifiedDecode(self.step, label='synthetic_decode')
        x = np.array([2], dtype=np.int32)
        step.verify(x, seed=7)
        step(x)
        self.assertTrue(step.verify(x, seed=99))
        self.assertFalse(step.verify(x.astype(np.float32), seed=99))
        np.testing.assert_array_equal(self.mx.random.state[0], [8])


if __name__ == '__main__':
    unittest.main()
