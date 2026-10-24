"""Studio's bridge for VoiceMem initialization, streaming, and shared models."""
from voicemem import VoiceMem

def open_memory(config):
    """Create memory with Studio reply/TTS/ASR injected through existing contracts."""
    from voicemem.config import build_kwargs
    from studio.core.utils.llm.initialize import create as reply_model, credential
    from studio.core.utils.tts.initialize import create as speech_model
    from studio.core.utils.asr.initialize import streaming, final
    segment = config["reply"]["llm"]
    memory_config = {k: v for k, v in config.items() if k not in {"reply", "tts"}}
    kwargs = build_kwargs(memory_config)
    kwargs.update(reply=reply_model(segment["config"]["system"], segment["provider"],
                                   credential(segment["provider"], "reply")),
                  tts=speech_model, asr=streaming, asr_final=final,
                  follow_input_language=True)
    return VoiceMem(**kwargs)

def open_stream(memory, **options):
    """Return the native stream without wrapping workers or cancellation guards."""
    return memory.stream(**options)


def greeting_memories(memory, *, limit: int = 3) -> list[str]:
    """Read a few low-sensitivity facts from this VoiceMem user's space."""
    repo = memory._o._get_repo()
    user_id = memory._o._user_id
    cognitive = repo._cognitive_store
    if cognitive is None:
        return []
    allowed_slots = {"daily_life", "knowledge", "goals", "work"}
    entries = repo._vector_store.list_entries(user_id=user_id, limit=80)
    entries.sort(key=lambda item: item.get("date", ""), reverse=True)
    selected = []
    for entry in entries:
        if entry.get("role") == "assistant":
            continue
        record = cognitive.get_memory_record(entry["id"])
        if (record is None or record.user_id != user_id
                or record.slot not in allowed_slots
                or record.sensitivity > 0.2):
            continue
        fact = " ".join(entry.get("text", "").split())[:160]
        if fact and fact not in selected:
            selected.append(fact)
        if len(selected) >= limit:
            break
    return selected


def shared_embed_model():
    """Return VoiceMem's process-shared embedding model for Studio perception."""
    from voicemem.leftbrain.local_embedder import resolve, resolve_path, shared_model

    return shared_model(resolve_path(resolve()))


def memory_warmups(memory, *, audio_path=None):
    """Return read-only inference checks using a supplied or disposable probe WAV."""
    import numpy as np
    from voicemem import gate
    from studio.core.utils.startup.audio import warmup_audio

    def audio_check(action):
        if audio_path is not None:
            return action(audio_path)
        with warmup_audio() as (_, path):
            return action(path)

    def require_vector(value, name):
        if value is None:
            raise RuntimeError(f'{name} 未输出向量')
        vector = np.asarray(value)
        if (vector.ndim != 1 or not vector.size or not np.isfinite(vector).all()
                or not np.any(vector)):
            raise RuntimeError(f'{name} 输出无效向量')

    def asr():
        model = memory.utils.get('asr')
        try:
            model.feed(np.zeros(9600, dtype=np.float32))
            model.flush()
        finally:
            model.reset()

    def scene():
        result = audio_check(memory._o._get_env_detector().detect_full)
        require_vector(result.get('embedding'), 'AST 场景')

    def speaker():
        vector = audio_check(memory._o._get_speaker_encoder().embed)
        require_vector(vector, '声纹')

    def emotion():
        detector = memory._o._get_emotion_detector()
        detector.warmup()
        audio_check(detector.detect)

    steps = [
        ('Embedding / 槽分类', lambda: memory.classify('你好')),
        ('流式 ASR', asr),
        ('VAD', lambda: memory.utils.get('vad').is_speech(np.zeros(512, dtype=np.float32))),
        ('AST 场景', scene),
        ('声纹', speaker),
        ('SenseVoice 情绪', emotion),
    ]
    if gate.ON:
        steps.insert(1, ('Turn Gate 语义路由', lambda: gate.semantic_margin(
            '这是一条启动检查消息。', memory.memory_language)))
    return tuple(steps)
