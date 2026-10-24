"""Shared inference execution and resource scheduling for VoiceMem and Studio.

Importing the engine does not import MLX, Torch or ONNX Runtime. Execution
lanes, priority contexts and background lifetimes each have one process owner;
provider models, dialogue policy and persistent memory retain their own owners.
"""
from .background import BackgroundLane, background_checkpoint, background_scope
from .context import Service, inference_scope
from .cpu import cpu_session_options
from .mlx import GpuLoop, Job, gpu_loop
from .torch import TORCH_LOCK, InferenceLock, cuda_initialization, device_lock, torch_snapshot
from .cuda import CudaWorker
from .admission import SpeechFeedback, observe_speech, resources, speech_scope
from .metrics import inference_call, model_metrics
from .system import observer

__all__ = [
    "BackgroundLane", "CudaWorker", "GpuLoop", "InferenceLock", "Job", "Service", "TORCH_LOCK",
    "background_checkpoint", "background_scope", "cpu_session_options",
    "gpu_loop", "inference_scope",
    "SpeechFeedback", "inference_call", "model_metrics", "observe_speech",
    "observer", "resources", "speech_scope",
    "cuda_initialization", "device_lock", "torch_snapshot",
]
