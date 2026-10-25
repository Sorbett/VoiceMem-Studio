"""Initialize the local three-way router once per process."""
from functools import lru_cache
import os
from studio.harness.reply_modes.policy import AVAILABLE_MODES, DIRECT, MEMORY, MEMORY_COT
from .component import FAST, MEDIUM, SLOW, THINKING_LEVELS, QwenThinkingRouter, ThinkingDecision, parse_level

@lru_cache(maxsize=1)
def thinking_router():
    """Return the recommended router or an adapter retaining filler generation."""
    from studio.core.utils.component_config.component import selection, custom
    value = selection('router')
    if value['provider'] == 'adapter':
        return custom('router', value)
    backend = os.environ.get('STUDIO_ROUTER_BACKEND') or (
        'mlx' if os.environ.get('STUDIO_BACKEND') == 'mlx' else 'torch')
    if backend == 'mlx':
        from .mlx import MLXThinkingRouter
        return MLXThinkingRouter()
    if backend != 'torch':
        raise ValueError('STUDIO_ROUTER_BACKEND must be mlx or torch')
    return QwenThinkingRouter(device=os.environ.get('STUDIO_DEVICE') or None)
