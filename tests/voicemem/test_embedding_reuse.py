"""Exact local embedding batches reuse CPU results without new native admission."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from voicemem.engine import TORCH_LOCK, Service, background_scope, inference_scope
from voicemem.leftbrain.local_embedder import _LockedEncoder


class EmbeddingReuseTests(unittest.TestCase):
    def test_interactive_encoder_request_precedes_queued_background_request(self):
        entered, release = threading.Event(), threading.Event()
        order = []

        def native(texts, **kwargs):
            if texts == ['owner']:
                entered.set()
                release.wait(2)
            order.append(texts[0])
            return np.zeros((1, 3), dtype=np.float32)

        encoder = _LockedEncoder(SimpleNamespace(encode=native, device='cpu'))

        def encode(text, service):
            with inference_scope(service):
                return encoder.encode([text])

        with ThreadPoolExecutor(max_workers=3) as workers:
            owner = workers.submit(encode, 'owner', Service.INTERACTIVE)
            self.assertTrue(entered.wait(1))
            background = workers.submit(encode, 'background', Service.BACKGROUND)
            interactive = workers.submit(encode, 'interactive', Service.INTERACTIVE)
            try:
                deadline = time.monotonic() + 1
                while len(encoder._lock._waiting) < 2 and time.monotonic() < deadline:
                    time.sleep(.001)
                self.assertEqual(len(encoder._lock._waiting), 2)
            finally:
                release.set()
            for future in (owner, background, interactive):
                future.result(timeout=1)
        self.assertEqual(order, ['owner', 'interactive', 'background'])

    def setUp(self):
        self.enterContext(patch("voicemem.engine.telemetry.metrics._metrics", {}))

    def encoder(self, *, value=None):
        parameter = SimpleNamespace(_version=0, device="cpu", dtype="float32")
        model = SimpleNamespace(training=False, parameters=lambda: [parameter], device="cpu",
                                prompts={}, max_seq_length=512)
        model.encode = Mock(side_effect=lambda *a, **k: (
            np.arange(6, dtype=np.float32).reshape(2, 3) if value is None else value.copy()))
        return _LockedEncoder(model), parameter

    def test_identical_batch_bypasses_admission_and_returns_independent_exact_arrays(self):
        encoder, _ = self.encoder()
        with background_scope(None), patch("voicemem.engine.telemetry.metrics.resources") as resources:
            resources.wait.return_value = (0., False)
            first = encoder.encode(["query: synthetic item", "query: second item"], normalize_embeddings=True)
            expected = first.copy()
            first[:] = -1
            second = encoder.encode(["query: synthetic item", "query: second item"], normalize_embeddings=True)
            second[:] = -2
            third = encoder.encode(["query: synthetic item", "query: second item"], normalize_embeddings=True)
            encoder._m.encode.assert_called_once()
            resources.wait.assert_called_once()
            resources.observe.assert_called_once()
        np.testing.assert_array_equal(third, expected)
        self.assertEqual(third.dtype, expected.dtype)
        self.assertEqual(third.shape, expected.shape)
        self.assertFalse(np.shares_memory(second, third))

    def test_input_order_prefix_shape_and_options_remain_distinct(self):
        encoder, _ = self.encoder()
        with background_scope(None):
            for texts, options in ((["query: a", "query: b"], {}),
                                   (["query: b", "query: a"], {}),
                                   (["passage: a", "passage: b"], {}),
                                   ("query: a", {}),
                                   (["query: a"], {}),
                                   (["query: a"], {"normalize_embeddings": True})):
                encoder.encode(texts, **options)
                encoder.encode(texts, **options)
        self.assertEqual(encoder._m.encode.call_count, 6)

    def test_model_prompt_configuration_and_parameter_versions_invalidate_reuse(self):
        encoder, parameter = self.encoder()
        other, _ = self.encoder()
        with background_scope(None):
            encoder.encode(["fixture"])
            other.encode(["fixture"])
            parameter._version += 1
            encoder.encode(["fixture"])
            encoder._m.prompts["query"] = "query: "
            encoder.encode(["fixture"])
            encoder._m.max_seq_length = 128
            encoder.encode(["fixture"])
            parameter.device = encoder._m.device = "mps"
            encoder.encode(["fixture"])
        self.assertEqual(encoder._m.encode.call_count, 5)
        other._m.encode.assert_called_once()

    def test_device_options_tensor_outputs_training_and_unknown_options_bypass(self):
        encoder, _ = self.encoder()
        with background_scope(None):
            for options in ({"device": "cpu"}, {"convert_to_tensor": True},
                            {"convert_to_numpy": False}, {"output_value": "token_embeddings"},
                            {"custom": True}, {"batch_size": object()}):
                encoder.encode(["fixture"], **options)
                encoder.encode(["fixture"], **options)
            encoder._m.training = True
            encoder.encode(["fixture"])
            encoder.encode(["fixture"])
        self.assertEqual(encoder._m.encode.call_count, 14)

    def test_job_boundaries_and_direct_calls_do_not_share_vectors(self):
        encoder, _ = self.encoder()
        encoder.encode(["fixture"])
        encoder.encode(["fixture"])
        for _ in range(2):
            with background_scope(None):
                encoder.encode(["fixture"])
                encoder.encode(["fixture"])
        self.assertEqual(encoder._m.encode.call_count, 4)

    def test_captured_worker_reuses_after_preparation_scope_has_exited(self):
        encoder, _ = self.encoder()
        with background_scope(None):
            first = encoder.encode(["fixture"])
            captured = copy_context()
        with ThreadPoolExecutor(max_workers=1) as workers:
            second = workers.submit(captured.run, encoder.encode, ["fixture"]).result(timeout=1)
        encoder._m.encode.assert_called_once()
        np.testing.assert_array_equal(first, second)

    def test_device_lock_owner_never_waits_for_coalesced_inference(self):
        encoder, _ = self.encoder()
        with background_scope(None):
            encoder.encode(["fixture"])
            with TORCH_LOCK:
                encoder.encode(["fixture"])
        self.assertEqual(encoder._m.encode.call_count, 2)

    def test_model_lock_owner_uses_reentrant_native_path(self):
        encoder, _ = self.encoder()
        with background_scope(None):
            encoder.encode(["fixture"])
            with encoder._lock:
                encoder.encode(["fixture"])
        self.assertEqual(encoder._m.encode.call_count, 2)

    def test_simultaneous_captured_calls_share_one_native_execution(self):
        encoder, _ = self.encoder()
        entered, release = threading.Event(), threading.Event()
        def native(*a, **k):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("fixture not released")
            return np.ones((1, 3), dtype=np.float32)
        encoder._m.encode.side_effect = native
        with background_scope(None):
            first_context, second_context = copy_context(), copy_context()
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(first_context.run, encoder.encode, ["fixture"])
            try:
                self.assertTrue(entered.wait(1))
                second = workers.submit(second_context.run, encoder.encode, ["fixture"])
            finally:
                release.set()
            np.testing.assert_array_equal(first.result(timeout=1), second.result(timeout=1))
        encoder._m.encode.assert_called_once()

    def test_oversized_or_non_numpy_values_keep_native_behavior(self):
        for value in (np.zeros(600_000, dtype=np.float32), [1, 2, 3],
                      np.zeros(1, dtype=[("field", np.float32)])):
            with self.subTest(type=type(value).__name__), background_scope(None):
                encoder, _ = self.encoder(value=value)
                encoder.encode(["fixture"])
                encoder.encode(["fixture"])
                self.assertEqual(encoder._m.encode.call_count, 2)


if __name__ == "__main__":
    unittest.main()
