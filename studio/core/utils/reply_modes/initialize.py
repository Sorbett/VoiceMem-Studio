"""Initialize the depth router once per process after startup acquisition."""
from functools import lru_cache
from studio.harness.reply_modes.policy import AVAILABLE_MODES, DIRECT, MEMORY, MEMORY_COT
from .component import FAST, MEDIUM, SLOW, THINKING_LEVELS, QwenThinkingRouter, ThinkingDecision, parse_level

@lru_cache(maxsize=1)
def thinking_router():
    """Return the recommended router or an adapter retaining filler generation."""
    from studio.core.utils.component_config.component import selection, custom
    value = selection('router')
    if value['provider'] == 'adapter':
        return custom('router', value)
    from studio.paths import MODELS
    from .checkpoint import runtime
    from .encoder_depth import EncoderDepthRouter
    selected = runtime()
    return EncoderDepthRouter(MODELS / selected.directory, backend=selected.backend,
                              device=selected.device, allow_prototype=True)
