"""CPU inference budgets supplied to optional provider runtimes."""
from __future__ import annotations

import os


def cpu_thread_budget():
    """Return the shared FunASR intra-op budget without importing a runtime.

    Studio configures the process before models are created. The standalone
    SDK retains FunASR's four-thread default. Invalid values raise ValueError.
    """
    value = os.environ.get("VOICEMEM_ENGINE_CPU_THREADS", "4")
    try:
        threads = int(value)
    except ValueError:
        raise ValueError("VOICEMEM_ENGINE_CPU_THREADS must be a positive integer") from None
    if threads < 1:
        raise ValueError("VOICEMEM_ENGINE_CPU_THREADS must be a positive integer")
    return threads


def cpu_session_options(runtime, *, threads=2):
    """Build bounded ONNX CPU execution without idle worker spinning.

    The caller supplies the optional runtime, keeping engine import lightweight.
    Model inputs, graph optimization and output contracts remain provider-owned.
    """
    if threads < 1:
        raise ValueError("CPU inference threads must be positive")
    options = runtime.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = threads
    options.execution_mode = runtime.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options
