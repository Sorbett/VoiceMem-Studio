"""Bounded auxiliary generation on already-loaded router weights.

Each request owns its KV cache. Torch releases the shared inference lock between
steps; MLX yields each step to the existing GPU scheduler. Cancellation cannot
preempt a native forward pass, but prevents any subsequent decode step.
"""
from __future__ import annotations

import asyncio
import queue
import secrets
import threading
import time


def first_sentence(text: str) -> str:
    """Return a completed first sentence, or empty text while it is incomplete."""
    for index, char in enumerate(text):
        if char in "。！？!?\n.":
            return text[:index + 1].strip()
    return ""


async def generate_short_text(router, system: str, prompt: str, *,
                              max_tokens: int = 40, timeout_s: float = 1.2) -> str:
    """Skip cold models and bound queueing plus generation by one deadline."""
    if router._model is None or max_tokens <= 0 or timeout_s <= 0:
        return ""
    cancelled = threading.Event()
    deadline = time.monotonic() + timeout_s
    try:
        return await asyncio.wait_for(asyncio.to_thread(
            router._short_text, system, prompt, max_tokens, cancelled, deadline),
            timeout=timeout_s)
    except asyncio.TimeoutError:
        return ""
    finally:
        cancelled.set()


def _expired(cancelled, deadline):
    return cancelled.is_set() or time.monotonic() >= deadline


def _messages(system, prompt):
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt}]


def torch_short_text(router, system, prompt, max_tokens, cancelled, deadline):
    """Decode off-loop, releasing device admission between native steps."""
    import torch
    from voicemem.engine import device_lock

    if _expired(cancelled, deadline):
        return ""
    tokenizer, model, device = router._tokenizer, router._model, router._device
    lock = device_lock(device)
    rendered = tokenizer.apply_chat_template(
        _messages(system, prompt), tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    inputs = tokenizer(rendered, return_tensors='pt')
    # Keep prefill bounded without cutting the current utterance mid-sentence.
    # Oversized auxiliary requests are optional and can simply be skipped.
    if inputs['input_ids'].shape[1] > 1024:
        return ""
    pending = inputs['input_ids']
    cache = None
    generated = []
    rng = None
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    eos.add(tokenizer.eos_token_id)
    for _ in range(max_tokens):
        while not _expired(cancelled, deadline):
            if lock.acquire(timeout=0.01):
                break
        else:
            return ""
        try:
            if _expired(cancelled, deadline):
                return ""
            with torch.inference_mode():
                if rng is None:
                    rng = torch.Generator(device=device).manual_seed(secrets.randbits(63))
                result = model(input_ids=pending.to(device), past_key_values=cache,
                               use_cache=True)
                cache = result.past_key_values
                probabilities = torch.softmax(result.logits[0, -1].float() / 0.7, dim=-1)
                token = torch.multinomial(probabilities, 1, generator=rng).item()
                pending = torch.tensor([[token]], device=device)
        finally:
            lock.release()
        if _expired(cancelled, deadline):
            return ""
        if token in eos:
            return tokenizer.decode(generated, skip_special_tokens=True).strip()
        generated.append(token)
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        sentence = first_sentence(text)
        if sentence:
            return sentence
        # Let queued ASR/router calls acquire the lock before the next token.
        if cancelled.wait(0.001):
            return ""
    return ""


def mlx_short_text(router, system, prompt, max_tokens, cancelled, deadline):
    """Consume a cancellable, non-exclusive job on the process MLX thread."""
    from voicemem.engine import gpu_loop

    def tokens():
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        if _expired(cancelled, deadline):
            return
        tokenizer, model = router._tokenizer, router._model
        ids = tokenizer.apply_chat_template(
            _messages(system, prompt), tokenize=True, add_generation_prompt=True,
            enable_thinking=False)
        if len(ids) > 1024:
            return
        # This cache never replaces the classifier's immutable prefix cache.
        cache = make_prompt_cache(model)
        pending = mx.array(ids)
        while len(pending) > 128:
            if _expired(cancelled, deadline):
                return
            model(pending[:128][None], cache=cache)
            mx.eval([layer.state for layer in cache])
            pending = pending[128:]
            yield ('', False)
        generated = []
        key = mx.random.key(secrets.randbits(32))
        for _ in range(max_tokens):
            if _expired(cancelled, deadline):
                return
            logits = model(pending[None], cache=cache)
            key, sample_key = mx.random.split(key)
            token = mx.random.categorical(logits[0, -1, :] / 0.7, key=sample_key).item()
            if token in tokenizer.eos_token_ids:
                yield (tokenizer.decode(generated).strip(), True)
                return
            generated.append(token)
            yield (tokenizer.decode(generated).strip(), False)
            pending = mx.array([token])

    if _expired(cancelled, deadline):
        return ""
    job = gpu_loop().iter(tokens, weight=1)
    try:
        while not _expired(cancelled, deadline):
            try:
                item = job.out.get(timeout=0.02)
            except queue.Empty:
                continue
            if item is None:
                return ""
            kind, value = item
            if kind == 'err':
                raise value
            text, done = value
            if _expired(cancelled, deadline):
                return ""
            sentence = first_sentence(text)
            if sentence or done:
                return sentence or text
        return ""
    finally:
        job.cancel()
