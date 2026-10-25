"""Shared inference execution and resource scheduling for VoiceMem and Studio.

Importing the engine does not import MLX, Torch or ONNX Runtime. Execution
lanes, priority contexts and background lifetimes each have one process owner;
provider models, dialogue policy and persistent memory retain their own owners.
"""
from .scheduling.background import BackgroundLane, BackgroundPipeline, background_checkpoint, background_scope
from .scheduling.context import Service, inference_scope
from .scheduling.probes import LatestProbe
from .scheduling.cancellation import InferenceCancelled
from .scheduling.capacity import BackgroundCapacity, background_capacity, configure_background_capacity
from .scheduling.reuse import ReusableResult, reuse_call, reuse_scope
from .backends.cpu import cpu_session_options, cpu_thread_budget
from .backends.mlx.scheduler import GpuLoop, Job, gpu_loop
from .backends.mlx.prefill import prefill_steps
from .backends.mlx.compilation import VerifiedDecode
from .backends.torch import TORCH_LOCK, InferenceLock, cuda_initialization, device_lock, torch_snapshot
from .backends.cuda import CudaWorker
from .scheduling.admission import SpeechFeedback, observe_speech, resources, speech_scope
from .telemetry.metrics import inference_call, model_metrics
from .telemetry.system import observer

__all__ = [
    "BackgroundCapacity", "BackgroundLane", "BackgroundPipeline", "CudaWorker", "GpuLoop", "InferenceLock", "Job", "Service", "TORCH_LOCK",
    "background_capacity", "configure_background_capacity",
    "InferenceCancelled", "LatestProbe",
    "ReusableResult", "reuse_call", "reuse_scope",
    "background_checkpoint", "background_scope", "cpu_session_options", "cpu_thread_budget",
    "gpu_loop", "inference_scope", "prefill_steps", "VerifiedDecode",
    "SpeechFeedback", "inference_call", "model_metrics", "observe_speech",
    "observer", "resources", "speech_scope",
    "cuda_initialization", "device_lock", "torch_snapshot",
]
