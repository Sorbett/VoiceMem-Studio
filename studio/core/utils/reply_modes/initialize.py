"""Initialize the depth router once per process after startup acquisition."""
from functools import lru_cache
import os
import platform
from studio.harness.reply_modes.policy import AVAILABLE_MODES, DIRECT, MEMORY, MEMORY_COT
from .component import FAST, MEDIUM, SLOW, THINKING_LEVELS, QwenThinkingRouter, ThinkingDecision, parse_level

REPOSITORY = 'Sorbett/VoiceMem-Jev'
DIRECTORY = 'reply-router/VoiceMem-Jev'


def router_runtime(backend=None, device=None):
    """Select the repository's CUDA weights or its portable CPU ONNX export."""
    backend = backend or os.environ.get('STUDIO_BACKEND') or (
        'mlx' if platform.system() == 'Darwin' else 'cuda')
    device = device or os.environ.get('STUDIO_DEVICE') or ('cuda:0' if backend == 'cuda' else 'cpu')
    if device == 'cuda' or (device.startswith('cuda:') and device[5:].isdigit()):
        return 'eager', device
    return 'onnx', 'cpu'


@lru_cache(maxsize=1)
def thinking_router():
    """Return the recommended router or an adapter retaining filler generation."""
    from studio.core.utils.component_config.component import selection, custom
    value = selection('router')
    if value['provider'] == 'adapter':
        return custom('router', value)
    from studio.paths import MODELS
    from .encoder_depth import EncoderDepthRouter
    backend, device = router_runtime()
    return EncoderDepthRouter(MODELS / DIRECTORY / backend, backend=backend,
                              device=device, allow_prototype=True)
