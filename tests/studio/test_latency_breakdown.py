"""Confirmed playback timing must retain speculation overlap and omit content."""
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest

from studio.core.utils.latency import log_playback_timing, playback_timing


class LatencyBreakdownTests(unittest.TestCase):
    def fixture(self):
        pending = SimpleNamespace(
            speech_end=10.0, input_turn_id='input-1', text='private test text',
            timing={'endpoint_ms': 200, 'asr_tail_ms': 80, 'confirm_ms': 40,
                    'gate_ms': 10, 'retrieval_ms': 30, 'capture_ms': 5,
                    'captured_at': 10.325, 'route_started_at': 10.330,
                    'route_finished_at': 10.380, 'route_ms': 50})
        timeline = SimpleNamespace(output_id='output-1', latency={
            'started_at': 10.390, 'first_audio_at': 11.572,
            'llm_ms': 862, 'segment_ms': 120, 'tts_ms': 200})
        return pending, timeline

    def test_normal_stages_match_playback_without_double_counting_substages(self):
        pending, timeline = self.fixture()
        row = playback_timing(pending, timeline, 11.730)
        self.assertAlmostEqual(row['effective_head_ms'], 390)
        self.assertAlmostEqual(row['before_route_ms'], 5)
        self.assertAlmostEqual(row['handoff_ms'], 10)
        self.assertAlmostEqual(row['delivery_ms'], 158)
        self.assertAlmostEqual(sum(row[key] for key in (
            'endpoint_ms', 'asr_tail_ms', 'confirm_ms', 'capture_ms',
            'before_route_ms', 'route_ms', 'handoff_ms', 'llm_ms',
            'segment_ms', 'tts_ms', 'delivery_ms')), row['playback_ms'])

    def test_adopted_early_output_uses_confirmed_input_and_reports_overlap(self):
        pending, timeline = self.fixture()
        timeline.latency['started_at'] = 9.9
        row = playback_timing(pending, timeline, 11.730)
        self.assertAlmostEqual(row['effective_head_ms'], -100)
        self.assertAlmostEqual(row['overlap_ms'], 480)
        self.assertEqual(row['handoff_ms'], 0)

    def test_logs_only_identifiers_and_timing_metadata(self):
        pending, timeline = self.fixture()
        with redirect_stdout(io.StringIO()) as captured:
            log_playback_timing(pending, timeline, 11.730)
        line = captured.getvalue()
        self.assertIn('turn=input-1 output=output-1', line)
        self.assertIn('playback_ms=1730', line)
        self.assertNotIn(pending.text, line)

    def test_text_opening_and_legacy_timeline_need_no_extra_state(self):
        self.assertEqual(playback_timing(SimpleNamespace(speech_end=0), object(), 1), {})
        row = playback_timing(SimpleNamespace(speech_end=1), object(), 2)
        self.assertEqual(row, {'playback_ms': 1000})


if __name__ == '__main__':
    unittest.main()
