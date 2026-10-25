"""Thermal diagnostics remain optional and separate from model quality."""
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from voicemem.engine.telemetry.system import EngineObserver
from voicemem.engine.telemetry.thermal import DarwinThermalState, thermal_state


class ThermalTelemetryTests(unittest.TestCase):
    def test_os_categories_and_unknown_future_values(self):
        reader = DarwinThermalState.__new__(DarwinThermalState)
        reader._process, reader._selector = object(), object()
        for value, expected in ((0, 'nominal'), (1, 'fair'), (2, 'serious'),
                                (3, 'critical'), (20, 'unknown')):
            reader._send_state = Mock(return_value=value)
            self.assertEqual(reader.sample(), expected)
            reader._send_state.assert_called_once_with(reader._process, reader._selector)

    def test_unsupported_platform_and_failed_reader_are_not_reported_as_cool(self):
        with patch('voicemem.engine.telemetry.thermal.sys.platform', 'linux'), \
                patch('voicemem.engine.telemetry.thermal._reader') as reader:
            self.assertEqual(thermal_state(), 'unavailable')
            reader.assert_not_called()
        for reader in (None, SimpleNamespace(sample=Mock(side_effect=OSError('fixture')))):
            with patch('voicemem.engine.telemetry.thermal.sys.platform', 'darwin'), \
                    patch('voicemem.engine.telemetry.thermal._reader', return_value=reader):
                self.assertEqual(thermal_state(), 'unavailable')

    def test_idle_observer_reports_heat_without_model_work_or_paging_changes(self):
        observer, output = EngineObserver(), io.StringIO()
        with patch('voicemem.engine.telemetry.system._counters', return_value={}), \
                patch('voicemem.engine.telemetry.system.thermal_state', return_value='serious'), \
                patch('voicemem.engine.telemetry.system.time.monotonic', side_effect=[0, 6]), \
                patch('voicemem.engine.telemetry.system.resources.paging') as paging, \
                patch('voicemem.engine.telemetry.system.resources.thermal') as thermal, \
                patch('voicemem.engine.telemetry.system.model_metrics', return_value=[]), \
                patch.object(observer._stop, 'wait', side_effect=lambda _: observer._stop.set()), \
                redirect_stdout(output):
            observer._run()
        self.assertEqual(observer.snapshot()['thermal_state'], 'serious')
        self.assertIn('thermal_state=serious', output.getvalue())
        paging.assert_called_once_with(0)
        thermal.assert_called_once_with('serious')
