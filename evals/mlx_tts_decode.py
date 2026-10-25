"""Offline ABBA probe for verified Breeze depth compilation on Apple Silicon.

Run with Python 3.12 and already installed MLX/FunASR weights. No API, personal
audio, microphone, playback or Memory Space is used. Generated speech remains
in memory; PCM digests compare decoding outputs. Optional streaming ASR uses
the same generated fixture at its existing 600 ms cadence. Results measure
production under contention, not browser latency, temperature or recognition
quality on real speech. Run without other model benchmarks or active Studio.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEXTS = (
    '请帮我设置一个十分钟的计时器，结束之后提醒我休息。',
    '这是一条语音延迟测试，用来检查连续播放是否平稳。',
    '我们先检查缓存，再分析 CPU 和 GPU 的资源调度。',
)


def summary(values):
    if not values:
        return {'count': 0}
    ordered = sorted(values)
    return dict(count=len(values), mean_ms=statistics.mean(values)*1000,
                p95_ms=ordered[min(len(ordered)-1, int(len(ordered)*.95))]*1000,
                max_ms=max(values)*1000, deadline_misses=sum(x > .6 for x in values))


async def run(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('VOICEMEM_ENGINE_CPU_THREADS', '2')
    from studio.paths import MODELS
    os.environ['VOICEMEM_MODELS_DIR'] = str(MODELS)
    import numpy as np
    from studio.core.utils.tts.component import BreezeMLXTTS
    from voicemem.engine import cpu_thread_budget, gpu_loop, model_metrics
    from voicemem.engine.telemetry.thermal import thermal_state
    from voicemem.utils.audio.stream_io import resample

    tts = BreezeMLXTTS(model=str(MODELS / 'tts/Breeze-TTS-2-mlx-4bit'), max_tokens=200)
    print('ENVIRONMENT', json.dumps(dict(python=platform.python_version(),
        machine=platform.machine(), cpu_threads=cpu_thread_budget(), order=args.order,
        utterances=args.utterances, with_asr=args.with_asr, asr_device=args.asr_device,
        packages={name: version(name) for name in ('mlx', 'mlx-audio', 'torch', 'funasr')})), flush=True)

    with patch('voicemem.utils.common.prompt_trace.record_request'):
        fixture = b''.join([chunk async for chunk in tts.stream(TEXTS[0])])
        pcm = resample(np.frombuffer(fixture, dtype='<i2').astype(np.float32)/32767,
                       src=24000, dst=16000)
        pcm = np.concatenate((pcm, np.zeros(8000, dtype=np.float32)))
        asr = None
        if args.with_asr:
            def load():
                from voicemem.utils.audio.asr import FunASRStreamingASR
                model = FunASRStreamingASR(device=args.asr_device)
                for offset in range(0, len(pcm), model.STRIDE):
                    model.feed(pcm[offset:offset+model.STRIDE])
                model.flush()
                model.reset()
                return model
            asr = await asyncio.to_thread(load)

        for phase, arm in enumerate(args.order):
            mode = 'cached' if arm == 'A' else 'compiled'
            def configure():
                tts._depth.mode = mode
                tts._depth.functions.clear()
                tts._depth.warmup(2, tts.cfg_scale, tts.instruction)
                return [dict(verified=getattr(fn, 'verified', False),
                             enabled=getattr(fn, 'enabled', False))
                        for fn in tts._depth.functions.values()]
            begin = time.perf_counter()
            verification = await asyncio.to_thread(gpu_loop().call, configure)
            warmup_s = time.perf_counter()-begin
            stop = threading.Event()
            costs, transcripts = [], []

            def listen():
                offset = 0
                while not stop.is_set():
                    started = time.perf_counter()
                    asr.feed(pcm[offset:offset+asr.STRIDE])
                    costs.append(time.perf_counter()-started)
                    offset += asr.STRIDE
                    if offset >= len(pcm):
                        transcripts.append(hashlib.sha256(asr.flush().encode()).hexdigest())
                        asr.reset()
                        offset = 0
                    stop.wait(max(0., .6-(time.perf_counter()-started)))

            model_metrics(reset=True)
            listener = asyncio.create_task(asyncio.to_thread(listen)) if asr else None
            started, cpu_started = time.perf_counter(), time.process_time()
            results = []
            try:
                for index in range(args.utterances):
                    text = TEXTS[index % len(TEXTS)]
                    begin = time.perf_counter()
                    digest, audio, arrivals = hashlib.sha256(), 0., []
                    async for chunk in tts.stream(text):
                        arrivals.append(time.perf_counter()-begin)
                        digest.update(chunk)
                        audio += len(chunk)/48000
                    elapsed = time.perf_counter()-begin
                    results.append(dict(fixture=index % len(TEXTS), elapsed_s=elapsed,
                        first_pcm_ms=arrivals[0]*1000, rtf=elapsed/audio, audio_s=audio,
                        max_gap_ms=max((b-a for a,b in zip(arrivals, arrivals[1:])), default=0.)*1000,
                        pcm_sha256=digest.hexdigest()))
            finally:
                stop.set()
                if listener:
                    await listener
                    asr.reset()
            def decode_snapshot():
                import mlx.core as mx
                return dict(profiles=[fn.snapshot() for fn in tts._depth.functions.values()
                                      if hasattr(fn, 'snapshot')],
                            allocated_bytes=mx.get_active_memory(), cache_bytes=mx.get_cache_memory(),
                            peak_bytes=mx.get_peak_memory())
            decoding = await asyncio.to_thread(gpu_loop().call, decode_snapshot)
            print('RESULT', json.dumps(dict(phase=phase, arm=arm, depth=mode,
                warmup_s=warmup_s, verification=verification,
                decoding=decoding,
                elapsed_s=time.perf_counter()-started, cpu_s=time.process_time()-cpu_started,
                asr=summary(costs), asr_transcript_digests=transcripts, tts=results,
                models=model_metrics(reset=True),
                thermal=await asyncio.to_thread(thermal_state))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order', default='ABBA')
    parser.add_argument('--utterances', type=int, default=3)
    parser.add_argument('--with-asr', action='store_true')
    parser.add_argument('--asr-device', default='mps', choices=('cpu', 'mps'))
    args = parser.parse_args()
    if not args.order or set(args.order) - {'A', 'B'} or args.utterances < 1:
        parser.error('order must contain A/B and utterances must be positive')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
