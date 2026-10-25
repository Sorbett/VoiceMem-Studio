"""Initialize the existing recognizers from verified local weights."""
from .component import FunASRStreamingASR, StreamingASR, OfflineASR
from studio.paths import MODELS
import os
from functools import lru_cache


@lru_cache(maxsize=4)
def _streaming_model(kind, device):
    if kind == "funasr":
        model = FunASRStreamingASR(
            model=str(MODELS / "asr/funasr-paraformer-zh-streaming"), device=device)
        print(f"[asr] 流式识别设备：{device}", flush=True)
        return model
    return StreamingASR(str(MODELS / "asr/sherpa-onnx-streaming-zipformer-en-2023-06-26"))

def streaming(language="auto"):
    """Default to bilingual FunASR; retain the explicit English adapter."""
    from studio.core.utils.component_config.component import selection
    value = selection('asr')
    if value['provider'] == 'deepgram':
        from .deepgram import DeepgramASR
        return DeepgramASR(model=value['model'], base_url=value['baseUrl'],
                           api_key=value['apiKey'], language=value['language'])
    if value['provider'] == 'adapter':
        return _custom_streaming().new_stream()
    from voicemem.utils.audio.asr import pick_device
    from voicemem.engine import device_lock
    kind = "funasr" if language in {"auto", "zh"} else "sherpa"
    configured = (os.environ.get('STUDIO_ASR_DEVICE')
                  or os.environ.get('VOICEMEM_ASR_DEVICE')
                  or (os.environ.get('STUDIO_DEVICE')
                      if os.environ.get('STUDIO_BACKEND') == 'cuda' else None)
                  or 'auto')
    device = (pick_device() if configured == 'auto' else configured) if kind == "funasr" else "cpu"
    lock = device_lock(device)
    device = lock.device or device
    with lock:
        return _streaming_model(kind, device).new_stream()


@lru_cache(maxsize=1)
def _final_model():
    return OfflineASR(str(MODELS / "asr/sensevoice-int8"),
                      allowed_languages=("zh", "en", "yue"))


def final():
    """Reuse stateless refinement weights without fixing one input language."""
    from studio.core.utils.component_config.component import selection
    if selection('asr_final')['provider'] == 'adapter':
        return _custom_final()
    from voicemem.engine import TORCH_LOCK
    with TORCH_LOCK:
        return _final_model()


@lru_cache(maxsize=1)
def _custom_streaming():
    from studio.core.utils.component_config.component import custom
    return custom('asr')


@lru_cache(maxsize=1)
def _custom_final():
    from studio.core.utils.component_config.component import custom
    return custom('asr_final')
