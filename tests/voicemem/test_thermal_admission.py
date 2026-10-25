"""Thermal pacing changes background admission, never model results."""
import unittest
from unittest.mock import patch

from voicemem.engine import Service, inference_call
from voicemem.engine.scheduling.admission import ResourceAdmission


class ThermalAdmissionTests(unittest.TestCase):
    def test_fair_starts_light_pacing_before_serious_pressure(self):
        now = [0.0]
        admission = ResourceAdmission(clock=lambda: now[0])
        admission.thermal('fair')
        admission.observe('synthetic', .8, device='cpu')
        self.assertAlmostEqual(admission.snapshot()['cooling_ms'], 400)
        self.assertEqual(admission.snapshot()['pacing_level'], .5)
        self.assertTrue(admission._blocked('next', now[0], 'mps'))
        now[0] = .41
        self.assertFalse(admission._blocked('next', now[0], 'mps'))

    def test_hot_background_gets_a_bounded_gap_even_without_active_speech(self):
        now = [0.0]
        admission = ResourceAdmission(clock=lambda: now[0])
        admission.thermal('serious')
        admission.observe('synthetic', .5)
        self.assertTrue(admission._blocked('next', now[0]))
        self.assertAlmostEqual(admission.snapshot()['cooling_ms'], 500)
        now[0] = .6
        self.assertFalse(admission._blocked('next', now[0]))
        admission.thermal('critical')
        admission.observe('synthetic', 10)
        self.assertAlmostEqual(admission.snapshot()['cooling_ms'], 2000)
        now[0] += 2.1
        self.assertFalse(admission._blocked('next', now[0]))

    def test_cool_recovery_requires_five_seconds_of_fresh_samples(self):
        now = [0.0]
        admission = ResourceAdmission(clock=lambda: now[0])
        admission.thermal('serious')
        for second in range(1, 7):
            now[0] = float(second)
            admission.thermal('nominal')
            admission.observe('synthetic', .5)
            self.assertEqual(admission._blocked('next', now[0]), second < 6)

    def test_unknown_unavailable_and_stale_samples_cannot_keep_work_blocked(self):
        now = [0.0]
        admission = ResourceAdmission(clock=lambda: now[0])
        for state in ('unknown', 'unavailable'):
            admission.thermal('critical')
            admission.observe('synthetic', 5)
            admission.thermal(state)
            self.assertFalse(admission._blocked('next', now[0]))
            self.assertEqual(admission.snapshot()['thermal_state'], state)
        admission.thermal('critical')
        now[0] += 3.1
        admission.observe('synthetic', 5)
        self.assertFalse(admission._blocked('next', now[0]))

    def test_recovery_to_fair_keeps_light_pacing_and_heating_escalates_immediately(self):
        now = [0.0]
        admission = ResourceAdmission(clock=lambda: now[0])
        admission.thermal('serious')
        for second in range(1, 7):
            now[0] = float(second)
            admission.thermal('fair')
            self.assertEqual(admission.snapshot()['pacing_level'], 1 if second < 6 else .5)
        admission.thermal('critical')
        self.assertEqual(admission.snapshot()['pacing_level'], 2)

    def test_thermal_pacing_does_not_couple_background_work_on_separate_cuda_devices(self):
        admission = ResourceAdmission()
        admission.thermal('serious')
        admission.observe('synthetic', 1., device='cuda:0')
        self.assertTrue(admission._blocked('next', admission.clock(), 'cuda:0'))
        self.assertFalse(admission._blocked('next', admission.clock(), 'cuda:1'))
        self.assertTrue(admission._blocked('next', admission.clock()))

    def test_foreground_bypasses_cooling_and_durable_background_still_progresses(self):
        admission = ResourceAdmission(max_defer_s=.01)
        admission.thermal('critical')
        admission.observe('synthetic', 1)
        with patch('voicemem.engine.telemetry.metrics.resources', admission):
            for service in (Service.AUDIO, Service.INTERACTIVE):
                with inference_call('foreground', 'torch', service=service):
                    pass
            self.assertEqual(admission.deferred, 0)
            with inference_call('durable', 'torch', service=Service.BACKGROUND):
                result = 'unchanged result'
            self.assertEqual(result, 'unchanged result')
            self.assertEqual(admission.forced, 1)
