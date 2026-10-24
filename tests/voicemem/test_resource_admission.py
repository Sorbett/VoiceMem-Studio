"""Priority, aging and ownership at native inference boundaries."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from voicemem.engine import InferenceLock, Service, cpu_session_options, inference_scope


class ResourceAdmissionTests(unittest.TestCase):
    def test_unavailable_mps_statistics_cannot_orphan_native_lock(self):
        from unittest.mock import patch
        lock = InferenceLock()
        torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
                                mps=SimpleNamespace(current_allocated_memory=Mock(side_effect=NotImplementedError)))
        with patch.dict("sys.modules", {"torch": torch}):
            with lock:
                pass
            self.assertTrue(lock.acquire(False))
            lock.release()
        self.assertFalse(lock._is_owned())
        self.assertTrue(lock.snapshot()["sampling_unavailable"])

    def test_cpu_budget_bounds_threads_without_changing_model_graph_options(self):
        options = SimpleNamespace(add_session_config_entry=Mock(), graph_optimization_level='runtime-default')
        runtime = SimpleNamespace(SessionOptions=lambda: options,
                                  ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL='sequential'))
        self.assertIs(cpu_session_options(runtime), options)
        self.assertEqual(options.intra_op_num_threads, 2)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertEqual(options.graph_optimization_level, 'runtime-default')
        self.assertEqual(options.execution_mode, 'sequential')
        self.assertEqual(options.add_session_config_entry.call_count, 2)
        with self.assertRaises(ValueError):
            cpu_session_options(runtime, threads=0)

    def wait_for_tickets(self, lock, count):
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with lock._condition:
                if len(lock._waiting) == count:
                    return
            time.sleep(.001)
        self.fail('native waiters did not arrive')

    def test_audio_precedes_background_without_preempting_active_call(self):
        lock, order = InferenceLock(max_defer_s=10), []
        def work(service):
            with inference_scope(service), lock:
                order.append(service)
        with ThreadPoolExecutor(max_workers=2) as pool:
            with lock:
                background = pool.submit(work, Service.BACKGROUND)
                self.wait_for_tickets(lock, 1)
                audio = pool.submit(work, Service.AUDIO)
                self.wait_for_tickets(lock, 2)
                self.assertEqual(order, [])
            background.result(timeout=1)
            audio.result(timeout=1)
        self.assertEqual(order, [Service.AUDIO, Service.BACKGROUND])

    def test_aged_background_gets_one_call_before_new_audio(self):
        lock, order = InferenceLock(max_defer_s=0), []
        def work(service):
            with inference_scope(service), lock:
                order.append(service)
        with ThreadPoolExecutor(max_workers=2) as pool:
            with lock:
                background = pool.submit(work, Service.BACKGROUND)
                self.wait_for_tickets(lock, 1)
                audio = pool.submit(work, Service.AUDIO)
                self.wait_for_tickets(lock, 2)
            background.result(timeout=1)
            audio.result(timeout=1)
        self.assertEqual(order, [Service.BACKGROUND, Service.AUDIO])

    def test_timeout_removes_ticket_and_recursion_preserves_ownership(self):
        lock = InferenceLock()
        with ThreadPoolExecutor(max_workers=1) as pool:
            with lock:
                self.assertTrue(lock.acquire(False))
                self.assertTrue(lock._is_owned())
                lock.release()
                self.assertFalse(pool.submit(lock.acquire, True, .01).result(timeout=1))
                self.assertFalse(pool.submit(lock.acquire, False).result(timeout=1))
                self.assertEqual(lock._waiting, [])
                with self.assertRaises(RuntimeError):
                    pool.submit(lock.release).result(timeout=1)
            self.assertFalse(lock._is_owned())
            self.assertTrue(lock.acquire(timeout=.01))
            lock.release()
        with self.assertRaises(ValueError):
            lock.acquire(False, .01)


if __name__ == '__main__':
    unittest.main()
