"""Reference encoding reuse must preserve codec results and waveform identity."""
import unittest

import numpy as np

from studio.core.utils.tts.reference_cache import ReferenceAudioCache


class ReferenceCacheTests(unittest.TestCase):
    def setUp(self):
        class Codec:
            calls = 0
            sample_rate = 24000

            def encode(self, samples, *, sr):
                self.calls += 1
                return {'audio_codes': [np.asarray(samples).copy()], 'rate': sr}

        self.codec = Codec()
        self.cache = ReferenceAudioCache(self.codec)

    def test_equal_waveforms_reuse_encoding_and_return_independent_values(self):
        samples = np.arange(8, dtype=np.float32)
        first = self.cache.encode(samples, sr=24000)
        first['audio_codes'][0][:] = -1
        second = self.cache.encode(samples.copy(), sr=24000)
        np.testing.assert_array_equal(second['audio_codes'][0], samples)
        second['audio_codes'].clear()
        third = self.cache.encode(samples, sr=24000)
        np.testing.assert_array_equal(third['audio_codes'][0], samples)
        self.assertEqual(self.codec.calls, 1)
        self.assertEqual((self.cache.hits, self.cache.misses), (2, 1))

    def test_waveform_rate_shape_and_dtype_changes_invalidate_the_entry(self):
        original = np.arange(8, dtype=np.float32)
        variants = [(original, 24000), (original + 1, 24000),
                    (original + 1, 16000), ((original + 1).reshape(2, 4), 16000),
                    ((original + 1).reshape(2, 4).astype(np.float64), 16000)]
        for samples, rate in variants:
            self.cache.encode(samples, sr=rate)
        self.assertEqual(self.codec.calls, len(variants))
        self.cache.encode(original, sr=24000)
        self.assertEqual(self.codec.calls, len(variants) + 1)

    def test_input_mutation_does_not_reuse_an_old_encoding(self):
        samples = np.zeros(8, dtype=np.float32)
        self.cache.encode(samples, sr=24000)
        samples[0] = 1
        result = self.cache.encode(samples, sr=24000)
        self.assertEqual(result['audio_codes'][0][0], 1)
        self.assertEqual(self.codec.calls, 2)

    def test_clear_releases_entry_and_codec_attributes_remain_available(self):
        samples = np.zeros(8, dtype=np.float32)
        self.cache.encode(samples, sr=24000)
        self.assertEqual(self.cache.sample_rate, 24000)
        self.cache.clear()
        self.cache.encode(samples, sr=24000)
        self.assertEqual(self.codec.calls, 2)


if __name__ == '__main__':
    unittest.main()
