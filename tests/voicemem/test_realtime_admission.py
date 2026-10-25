"""Periodic native cost paces background work without delaying ASR input."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from voicemem.engine import Service, inference_call
from voicemem.engine.scheduling.admission import ResourceAdmission
from voicemem.utils.audio.asr import FunASRStreamingASR


class RealtimeAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.now = [20.]
        self.admission = ResourceAdmission(clock=lambda: self.now[0])

    def observe(self, cost=.36, *, device='mps'):
        self.admission.realtime('streaming_asr', cost, .6, device=device)

    def heat(self, *, device='mps'):
        for _ in range(3):
            self.observe(device=device)

    def test_three_recent_costs_enable_pacing_without_os_thermal_support(self):
        self.admission.observe('background', .8, device='cpu')
        for _ in range(2):
            self.observe()
            self.assertFalse(self.admission._blocked('next', self.now[0], 'cpu'))
        self.observe()
        state = self.admission.snapshot()
        self.assertEqual(state['thermal_state'], 'unavailable')
        self.assertEqual(state['realtime_pressure'], 1)
        self.assertAlmostEqual(state['cooling_ms'], 800)
        self.assertTrue(self.admission._blocked('next', self.now[0], 'cpu'))
        self.now[0] += .81
        self.assertFalse(self.admission._blocked('next', self.now[0], 'cpu'))

    def test_slow_startup_or_isolated_spike_does_not_establish_pressure(self):
        self.observe(4.)
        self.assertEqual(self.admission.snapshot()['pacing_level'], 0)
        for _ in range(4):
            self.observe(.1)
            self.assertEqual(self.admission.snapshot()['pacing_level'], 0)
        self.now[0] += 3
        for _ in range(3):
            self.observe(.1)
        self.observe(.5)
        self.assertEqual(self.admission.snapshot()['pacing_level'], 0)

    def test_recovery_has_hysteresis_and_stale_costs_expire(self):
        self.heat()
        self.observe(.24)
        self.assertEqual(self.admission.snapshot()['realtime_pressure'], 1)
        for _ in range(8):
            self.observe(.1)
        self.assertEqual(self.admission.snapshot()['realtime_pressure'], 0)
        self.heat(device='cpu')
        self.now[0] += 3
        self.assertEqual(self.admission.snapshot()['realtime_pressure'], 0)
        self.observe(device='cpu')
        self.assertEqual(self.admission.snapshot()['realtime_pressure'], 0)

    def test_invalid_observations_cannot_poison_load(self):
        for cost, cadence in ((float('nan'), .6), (float('inf'), .6), (-1., .6),
                              (.1, 0.), (.1, -1.), (.1, float('nan'))):
            self.admission.realtime('invalid', cost, cadence)
        self.assertFalse(self.admission._realtime)

    def test_distinct_cuda_gpus_keep_independent_load_and_recovery_gaps(self):
        self.heat(device='cuda:0')
        self.admission.observe('first', .5, device='cuda:0')
        self.assertTrue(self.admission._blocked('next', self.now[0], 'cuda:0'))
        self.assertFalse(self.admission._blocked('next', self.now[0], 'cuda:1'))
        self.now[0] += .51
        self.admission.observe('second', 2., device='cuda:1')
        self.assertFalse(self.admission._blocked('next', self.now[0], 'cuda:0'))
        self.assertTrue(self.admission._blocked('next', self.now[0]))

    def test_foreground_never_waits_and_background_admission_remains_bounded(self):
        admission = ResourceAdmission(max_defer_s=.01)
        for _ in range(3):
            admission.realtime('streaming_asr', .5, .6)
        admission.observe('background', 2.)
        with patch('voicemem.engine.telemetry.metrics.resources', admission):
            for service in (Service.AUDIO, Service.INTERACTIVE):
                with inference_call('foreground', 'torch', service=service):
                    pass
            self.assertEqual(admission.deferred, 0)
            with inference_call('durable', 'torch', service=Service.BACKGROUND):
                result = 'preserved'
        self.assertEqual(result, 'preserved')
        self.assertEqual(admission.forced, 1)

    def test_native_runtime_excludes_lock_wait_and_failure_is_not_observed(self):
        clock = Mock(side_effect=[10., 12., 12.2, 20., 22., 22.3])
        admission = Mock()
        lock = SimpleNamespace(acquire=Mock(), release=Mock())
        with patch('voicemem.engine.telemetry.metrics.resources', admission), \
                patch('voicemem.engine.telemetry.metrics.time.perf_counter', clock):
            with inference_call('periodic', 'torch', lock=lock, service=Service.AUDIO,
                                device='mps', cadence_s=.6):
                pass
            with self.assertRaises(ValueError):
                with inference_call('failed', 'torch', service=Service.AUDIO, cadence_s=.6):
                    raise ValueError('fixture')
        admission.realtime.assert_called_once()
        args, kwargs = admission.realtime.call_args
        self.assertEqual(args[0], 'periodic')
        self.assertAlmostEqual(args[1], .2)
        self.assertEqual(args[2], .6)
        self.assertEqual(kwargs, {'device': 'mps'})

    def test_final_asr_tails_do_not_publish_periodic_load_or_change_decoding(self):
        asr = object.__new__(FunASRStreamingASR)
        asr.device = 'mps'
        asr.model = SimpleNamespace(generate=Mock(side_effect=[[{'text': 'partial'}],
                                                               [{'text': ' tail'}]]))
        asr.reset()
        pcm = np.ones(asr.STRIDE + 100, dtype=np.float32)
        with patch('voicemem.engine.device_lock', return_value=None), \
                patch('voicemem.engine.inference_call', return_value=nullcontext()) as call:
            self.assertEqual(asr.feed(pcm), 'partial')
            self.assertEqual(asr.flush(), 'partial tail')
        self.assertEqual([entry.kwargs['cadence_s'] for entry in call.call_args_list], [.6, None])
        requests = asr.model.generate.call_args_list
        np.testing.assert_array_equal(requests[0].kwargs['input'], pcm[:asr.STRIDE])
        np.testing.assert_array_equal(requests[1].kwargs['input'][:100], pcm[asr.STRIDE:])
        self.assertEqual([entry.kwargs['is_final'] for entry in requests], [False, True])
        for entry in requests:
            self.assertEqual(entry.kwargs['chunk_size'], [0, 10, 5])
            self.assertEqual(entry.kwargs['encoder_chunk_look_back'], 4)
            self.assertEqual(entry.kwargs['decoder_chunk_look_back'], 1)
