"""Process-wide serialization for shared Torch accelerator work.

Streaming ASR, embedding and GPU perception must not submit MPS work from
different threads concurrently. Independent CPU AST and emotion models use
their own instance locks instead. Reentrancy supports nested inference on the
same thread; never hold this lock across search coordination, background idle
waits, or a future whose worker may need the lock. MLX uses GpuLoop separately.
"""
from voicemem.engine import TORCH_LOCK

__all__ = ["TORCH_LOCK"]
