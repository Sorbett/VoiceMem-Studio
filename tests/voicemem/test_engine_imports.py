"""Engine import orders preserve one lightweight process resource owner."""
import subprocess
import sys
from pathlib import Path
import unittest


class EngineImportTests(unittest.TestCase):
    def test_new_and_legacy_entrypoints_share_resources_without_loading_runtimes(self):
        for first in ('voicemem.engine', 'voicemem.utils.engine',
                      'voicemem.utils.gpu_loop', 'voicemem.utils.torch_lock',
                      'voicemem.utils.background',
                      'voicemem.engine.backends.mlx.scheduler',
                      'voicemem.engine.backends.cuda',
                      'voicemem.engine.scheduling.admission',
                      'voicemem.engine.telemetry.system'):
            with self.subTest(first=first):
                script = f'''
import importlib
import sys

importlib.import_module({first!r})
engine = importlib.import_module('voicemem.engine')
legacy = importlib.import_module('voicemem.utils.engine')
gpu = importlib.import_module('voicemem.utils.gpu_loop')
torch = importlib.import_module('voicemem.utils.torch_lock')
background = importlib.import_module('voicemem.utils.background')
for name in engine.__all__:
    assert getattr(legacy, name) is getattr(engine, name), name
assert gpu.GpuLoop is engine.GpuLoop
assert gpu.Job is engine.Job
assert torch.TORCH_LOCK is engine.TORCH_LOCK
loop = engine.gpu_loop()
assert gpu.gpu_loop() is loop
assert legacy.gpu_loop() is loop
assert loop._thread is None
assert engine.device_lock('cuda:0') is engine.device_lock('cuda:00')
assert engine.device_lock('cuda:1') is not engine.device_lock('cuda:0')
assert 'cuda:0' in engine.torch_snapshot()['devices']
assert background.background_scope is engine.background_scope
assert importlib.import_module('voicemem.engine.backends.mlx.scheduler').GpuLoop is engine.GpuLoop
assert importlib.import_module('voicemem.engine.backends.cuda').CudaWorker is engine.CudaWorker
assert importlib.import_module('voicemem.engine.scheduling.admission').resources is engine.resources
assert importlib.import_module('voicemem.engine.telemetry.system').observer is engine.observer
calls = []
with background.background_scope(lambda: calls.append('admitted')):
    engine.background_checkpoint()
assert calls == ['admitted']
engine.background_checkpoint()
assert calls == ['admitted']
for name in ('mlx', 'torch', 'onnxruntime', 'studio'):
    assert name not in sys.modules, name
'''
                result = subprocess.run(
                    [sys.executable, '-c', script],
                    cwd=Path(__file__).resolve().parents[2],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_package_exposes_engine_lazily(self):
        script = '''
import sys
import voicemem
assert 'voicemem.engine' not in sys.modules
assert 'engine' in dir(voicemem)
engine = voicemem.engine
assert engine is sys.modules['voicemem.engine']
assert not any(name in sys.modules for name in ('mlx', 'torch', 'onnxruntime', 'studio'))
'''
        result = subprocess.run(
            [sys.executable, '-c', script],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
