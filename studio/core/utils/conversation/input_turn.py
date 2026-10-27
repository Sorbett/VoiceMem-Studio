"""Retain accepted speech when a continuation cancels an unheard reply."""
import asyncio
import math
from dataclasses import dataclass, replace

from studio.core.utils.reply_modes.initialize import DIRECT, MEMORY
from studio.harness.turn_taking.policy import CONTROLS
from voicemem import gate
from voicemem.lang import detect_language
from voicemem.stream import empty_result


def can_continue_input(base, pending):
    """Require a short voiced gap on one capture clock; unknown timing stays separate."""
    epoch = getattr(base, 'input_audio_epoch', None)
    if epoch is None or epoch != getattr(pending, 'input_audio_epoch', None):
        return False
    try:
        start, end = getattr(base, 'input_audio_span', None)
        next_start, next_end = getattr(pending, 'input_audio_span', None)
        return (all(math.isfinite(value) for value in (start, end, next_start, next_end))
                and 0 <= start <= end <= next_start <= next_end
                and next_start - end <= CONTROLS['continuation_gap_ms'] / 1000 + 1e-9)
    except (TypeError, ValueError):
        return False


def join_input(base, pending):
    """Combine validated speech spans while preserving repeated words and audio paths."""
    joiner = ("" if base.text and pending.text and all(
        "\u3400" <= ch <= "\u9fff" for ch in (base.text[-1], pending.text[0])) else " ")
    text = f"{base.text}{joiner}{pending.text}".strip()
    span = (base.input_audio_span[0], pending.input_audio_span[1])
    observed = getattr(pending, 'audio_state', None)
    if observed is not None:
        observed = {**observed, 'span_s': span[1] - span[0]}
    return replace(pending, text=text, continuation_prompt=False, early_ok=False,
                   replace_input_turn_id=base.input_turn_id, transcript_managed=False,
                   input_audio_span=span, audio_state=observed,
                   language=detect_language(text, base.language or "zh"))


async def refresh_continuation(agent, pending, memory_vm):
    """Revalidate memory against complete speech instead of the last fragment."""
    pending.result = empty_result()
    pending.memory_context = pending.replay = pending.memory_query = ""
    pending.route = await asyncio.to_thread(
        gate.route, pending.text, language=pending.language)
    pending.reply_mode = MEMORY if gate.needs_memory(pending.route) else DIRECT
    if gate.needs_memory(pending.route):
        try:
            await agent._ensure_pending_memory(pending, memory_vm)
        except Exception as exc:
            pending.result = empty_result()
            pending.memory_context = pending.replay = ""
            print(f"[route] continuation memory retrieval failed: {type(exc).__name__}: {exc}",
                  flush=True)


@dataclass
class DeferredInput:
    pending: object
    timeline: object
    memory_vm: object
    space: str
    owner: dict


class InputContinuation:
    """Own one unpersisted input until its next reply or session cleanup adopts it."""

    def __init__(self):
        self.held = None

    def defer(self, pending, timeline, memory_vm, space, owner, *, started=False):
        """Hold only identified owner speech whose main output has not started."""
        if (self.held is not None or pending is None or timeline is None
                or timeline.context_saved or started
                or timeline.rendered_cutoff_samples() > 0
                or not pending.input_turn_id or not pending.spoken
                or pending.opening or pending.stranger):
            return False
        self.held = DeferredInput(pending, timeline, memory_vm, space, dict(owner))
        return True

    def owns(self, timeline):
        return self.held is not None and self.held.timeline is timeline

    def in_scope(self, memory_vm, space, owner):
        held = self.held
        return bool(held and held.memory_vm is memory_vm and held.space == space
                    and all(held.owner.get(key, "") == owner.get(key, "")
                            for key in ("id", "last")))

    def merge(self, pending, memory_vm, space, owner, *, explicit=False):
        """Join only speech with the same captured memory and speaker ownership."""
        held = self.held
        if (held is None or not pending.spoken or pending.stranger or pending.opening
                or not pending.input_turn_id or explicit
                or not can_continue_input(held.pending, pending)
                or not self.in_scope(memory_vm, space, owner)):
            return pending
        held.pending = join_input(held.pending, pending)
        return held.pending

    def adopt(self, pending):
        """Transfer finalization to the replacement output, retiring the old one."""
        if self.held and self.held.pending.input_turn_id == pending.input_turn_id:
            self.held.timeline.context_saved = True
            self.held = None

    def take(self):
        """Release a deferred input for exactly-once fallback persistence."""
        held, self.held = self.held, None
        return held
