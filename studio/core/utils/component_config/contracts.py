"""Check custom adapters at the existing Studio input/output boundaries."""
from contextlib import aclosing
import inspect
import math


def _sync(value, name):
    if inspect.isawaitable(value):
        close = getattr(value, 'close', None)
        if close:
            close()
        raise TypeError(f'{name} must be synchronous on its existing worker')
    return value


def _text(value, name):
    if not isinstance(_sync(value, name), str):
        raise TypeError(f'{name} must return text')
    return value


class _Checked:
    def __init__(self, adapter):
        self.adapter = adapter

    def __getattr__(self, name):
        return getattr(self.adapter, name)


class _Streaming(_Checked):
    def new_stream(self):
        child = _sync(self.adapter.new_stream(), 'new_stream')
        if child is self.adapter:
            raise ValueError('new_stream must return an independent decoder')
        return checked(self.role, child)


class _ASR(_Streaming):
    role = 'asr'

    def feed(self, samples):
        return _text(self.adapter.feed(samples), 'ASR.feed')

    def flush(self):
        return _text(self.adapter.flush(), 'ASR.flush')

    def reset(self):
        _sync(self.adapter.reset(), 'ASR.reset')


class _FinalASR(_Checked):
    def transcribe(self, audio):
        return _text(self.adapter.transcribe(audio), 'ASR.transcribe')


class _VAD(_Streaming):
    role = 'vad'

    def is_speech(self, samples):
        result = _sync(self.adapter.is_speech(samples), 'VAD.is_speech')
        if not isinstance(result, bool):
            raise TypeError('VAD.is_speech must return bool')
        return result


class _EOT(_Checked):
    def score(self, audio):
        result = float(_sync(self.adapter.score(audio), 'EOT.score'))
        if not math.isfinite(result) or not 0 <= result <= 1:
            raise ValueError('EOT.score must return a probability in [0, 1]')
        return result


class _Router(_Checked):
    async def classify_async(self, *args, **kwargs):
        from studio.core.utils.reply_modes.component import ThinkingDecision, THINKING_LEVELS
        result = await self.adapter.classify_async(*args, **kwargs)
        if not isinstance(result, ThinkingDecision) or result.level not in THINKING_LEVELS:
            raise TypeError('Router must return ThinkingDecision with fast, medium or slow')
        return result

    async def generate_short_text_async(self, *args, **kwargs):
        result = await self.adapter.generate_short_text_async(*args, **kwargs)
        return _text(result, 'Router.generate_short_text_async')

    def warmup(self):
        _sync(self.adapter.warmup(), 'Router.warmup')


class _Speech(_Checked):
    async def stream(self, text, instruction=None):
        from voicemem.audio_timing import TimedAudioChunk
        async with aclosing(self.adapter.stream(text, instruction)) as chunks:
            async for chunk in chunks:
                pcm = chunk.pcm if isinstance(chunk, TimedAudioChunk) else chunk
                if not isinstance(pcm, bytes) or len(pcm) % 2:
                    raise TypeError('TTS must yield complete PCM16 samples')
                if isinstance(chunk, TimedAudioChunk) and chunk.sample_rate != 24000:
                    raise ValueError('TTS must yield mono audio at 24000 Hz')
                yield chunk


def checked(role, adapter):
    """Validate required methods, then retain cancellation and stream ownership."""
    _sync(adapter, f'{role} factory')
    methods = {
        'asr': ('feed', 'flush', 'reset', 'new_stream'),
        'asr_final': ('transcribe',),
        'vad': ('is_speech', 'new_stream'),
        'eot': ('score',),
        'router': ('classify_async', 'generate_short_text_async', 'warmup'),
        'speech': ('stream',),
    }[role]
    missing = [method for method in methods if not callable(getattr(adapter, method, None))]
    if missing:
        raise ValueError(f'{role} 适配器缺少接口：{", ".join(missing)}')
    if role == 'eot':
        threshold = getattr(adapter, 'threshold', None)
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError('EOT 适配器需要 0–1 之间的 threshold。')
    cls = {'asr':_ASR, 'asr_final':_FinalASR, 'vad':_VAD,
           'eot':_EOT, 'router':_Router, 'speech':_Speech}[role]
    return cls(adapter)
