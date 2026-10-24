"""Run the real GPU scheduler with a CPU-only MLX surface."""
from contextlib import contextmanager
import types
from unittest.mock import patch

import numpy as np

from voicemem.engine import GpuLoop


@contextmanager
def cpu_gpu_loop():
    core = types.ModuleType('mlx.core')
    core.array, core.argmax = np.array, np.argmax
    core.eval = lambda _: None
    core.gpu = object()
    core.set_default_device = lambda _: None
    core.set_cache_limit = lambda _: None
    root = types.ModuleType('mlx')
    root.core = core
    with patch.dict('sys.modules', {'mlx': root, 'mlx.core': core}):
        yield GpuLoop()
