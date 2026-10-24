"""Stable canonical audio avoids repeat EOT inference without changing decisions."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from voicemem.utils.audio.eot import EndOfTurn


class EotCacheTests(unittest.TestCase):
    def model(self):
        model = EndOfTurn.__new__(EndOfTurn)
        model._score_cache = None
        model.threshold = .5
        model._fe = Mock(return_value=SimpleNamespace(input_features=np.ones((1, 80, 800), np.float32)))
        model._sess = SimpleNamespace(run=Mock(return_value=[np.array([[.75]], np.float32)]))
        return model

    def test_extra_silence_reuses_identical_framed_input_without_retaining_audio(self):
        model = self.model()
        speech = np.ones(16000, np.float32)*.1
        a = np.concatenate((speech, np.zeros(4800, np.float32)))
        b = np.concatenate((speech, np.zeros(9600, np.float32)))
        self.assertEqual(model.score(a), model.score(b))
        model._sess.run.assert_called_once()
        model._fe.assert_called_once()
        key, score = model._score_cache
        self.assertIsInstance(key[-1], bytes)
        self.assertEqual(len(key[-1]), 32)
        self.assertIsInstance(score, float)
        self.assertFalse(any(isinstance(item, np.ndarray) for item in key))

    def test_changed_audio_rate_and_injected_components_each_invalidate(self):
        model = self.model()
        speech = np.ones(16000, np.float32)*.1
        self.assertEqual(model.score(speech), .75)
        speech[100] = .2
        model.score(speech)
        model.score(speech, 8000)
        self.assertEqual(model._sess.run.call_count, 3)
        model._sess = SimpleNamespace(run=Mock(return_value=[np.array([[.25]], np.float32)]))
        self.assertEqual(model.score(speech, 8000), .25)
        self.assertFalse(model.done(speech, 8000))
        model._sess.run.assert_called_once()
        model._fe = Mock(return_value=SimpleNamespace(input_features=np.zeros((1, 80, 800), np.float32)))
        model.score(speech, 8000)
        self.assertEqual(model._sess.run.call_count, 2)

    def test_provider_failure_does_not_publish_a_cached_result(self):
        model = self.model()
        model._sess.run.side_effect = RuntimeError("synthetic model failure")
        with self.assertRaises(RuntimeError):
            model.score(np.ones(16000, np.float32))
        self.assertIsNone(model._score_cache)
        model._sess.run.side_effect = None
        self.assertTrue(model.done(np.ones(16000, np.float32)))

    def test_strided_input_and_changed_threshold_keep_the_existing_contract(self):
        model = self.model()
        strided = np.ones(300000, np.float32)[::2]
        self.assertTrue(model.done(strided))
        model.threshold = .9
        self.assertFalse(model.done(strided.copy()))
        model._sess.run.assert_called_once()
