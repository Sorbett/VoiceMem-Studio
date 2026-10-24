"""Token-bounded MLX prefill steps for the process-owned execution thread."""
from __future__ import annotations


def prefill_steps(model, tokens, cache, *, step_size=256, keep_last=False,
                  cancelled=None):
    """Yield processed-token counts after materializing each private cache update.

    Consume on GpuLoop only. ``cancelled`` is an optional zero-argument predicate;
    cancellation leaves the caller-owned cache partial and must not publish it.
    ``keep_last`` reserves the final prompt token for the generation sampler.
    """
    if step_size < 1:
        raise ValueError("Prefill step size must be positive")
    import mlx.core as mx

    end = max(0, len(tokens) - int(keep_last))
    for start in range(0, end, step_size):
        if cancelled is not None and cancelled():
            return
        stop = min(end, start + step_size)
        model(mx.array(tokens[start:stop])[None], cache=cache)
        mx.eval([layer.state for layer in cache])
        yield stop
