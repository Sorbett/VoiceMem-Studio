"""Native Apple Silicon crossover probe for background admission.

Run with Python 3.12 from the repository root. Models must already be installed.
ABBA uses the same resident weights, synthetic input and decoding settings in
one process. PCM is discarded; no microphone, API, Memory Space or browser is
used. This measures production under contention, not perceived playback latency.
Host temperature, paging by other applications and phase order still matter.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from importlib.metadata import version
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summary(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {"count": len(values), "mean_ms": round(statistics.mean(values)*1000, 2),
            "p95_ms": round(ordered[min(len(ordered)-1, int(len(ordered)*.95))]*1000, 2),
            "max_ms": round(max(values)*1000, 2)}


async def main(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from studio.paths import MODELS
    os.environ["VOICEMEM_MODELS_DIR"] = str(MODELS)
    from studio.core.utils.startup.audio import warmup_audio
    from studio.core.utils.tts.component import BreezeMLXTTS
    from voicemem.engine import (SpeechFeedback, background_scope, model_metrics,
                                observer, resources, speech_scope)
    from voicemem.stream import _EOT_EXECUTOR
    from voicemem.utils.audio.eot import EndOfTurn
    if args.studio_voice:
        from studio.core.utils.tts.initialize import REFERENCE
        args.reference_audio = str(REFERENCE)

    tts = BreezeMLXTTS(model=str(MODELS / "tts/Breeze-TTS-2-mlx-4bit"),
                      ref_audio=args.reference_audio, max_tokens=160)
    eot = EndOfTurn(model_path=str(MODELS / "eot/smart-turn-v3.2-cpu.onnx"))
    with warmup_audio() as (pcm, path):
        def load():
            from voicemem.utils.audio.asr import FunASRStreamingASR
            from voicemem.utils.audio.environment.environment_detector_ast import ASTEnvironmentDetector
            from voicemem.utils.audio.emotion.detector import SmallEmotionDetector
            from voicemem.leftbrain.local_embedder import shared_model
            from studio.core.utils.reply_modes.mlx import MLXThinkingRouter
            from studio.core.utils.perception.component import Perception
            from voicemem.utils.audio.voiceprint.speaker_encoder import SpeakerEncoder
            asr = FunASRStreamingASR(device="mps")
            asr.feed(pcm[:9600]); asr.reset()
            scene = ASTEnvironmentDetector(); scene.detect_full(path)
            emotion = SmallEmotionDetector(); emotion.detect(path)
            embed = shared_model(str(MODELS / "embedding")); embed.encode(["synthetic scheduling fixture"])
            router = MLXThinkingRouter(); router.warmup()
            speaker = SpeakerEncoder(device="cpu")
            if speaker.embed(path) is None:
                raise RuntimeError("speaker probe produced no embedding")
            perception = Perception()
            perception._E2V = {}
            perception._E2V_MAP = {}
            perception._acoustic_emotion(str(path))
            eot.score(pcm)
            return asr, scene, emotion, embed, router, speaker, perception

        models = await asyncio.to_thread(load)
        async for _ in tts.stream("你好，语音测试。"):
            pass
        print("ENVIRONMENT", json.dumps(dict(python=platform.python_version(), machine=platform.machine(),
                                            system=platform.system(), release=platform.release(),
                                            packages={name: version(name) for name in (
                                                "mlx", "mlx-lm", "mlx-audio", "torch", "onnxruntime", "funasr")},
                                            order=args.order, utterances=args.utterances,
                                            background_jobs=args.background_jobs,
                                            studio_voice=args.studio_voice,
                                            threads_a=args.threads_a, threads_b=args.threads_b,
                                            depth_a=args.depth_a, depth_b=args.depth_b,
                                            fixed_policy=args.fixed_policy)), flush=True)
        observer.start()
        stop, done = threading.Event(), asyncio.Event()
        asr_times, loop_lags = [], []

        def listen():
            count = 0
            while not stop.is_set():
                begin = time.perf_counter()
                models[0].feed(pcm[:9600])
                count += 1
                if count % 5 == 0:
                    models[0].flush(); models[0].reset()
                asr_times.append(time.perf_counter()-begin)
                stop.wait(max(0, .6-(time.perf_counter()-begin)))

        async def heartbeat():
            while not done.is_set():
                begin = time.perf_counter()
                await asyncio.sleep(.005)
                loop_lags.append(max(0, time.perf_counter()-begin-.005))

        async def scores():
            index = 0
            while not done.is_set():
                begin = time.perf_counter()
                # Vary the waveform so continuous-input stress cannot benefit
                # from the stable-input EOT cache used during real pauses.
                import numpy as np
                index += 1
                await asyncio.get_running_loop().run_in_executor(
                    _EOT_EXECUTOR, eot.score, np.roll(pcm, index*317))
                await asyncio.sleep(max(.001, .04-(time.perf_counter()-begin)))

        def background():
            begin = time.perf_counter()
            with background_scope(None):
                models[1].detect_full(path)
                models[5].embed(path)
                models[2].detect(path)
                models[3].encode(["synthetic persistence fixture"])
                models[6]._acoustic_emotion(str(path))
            return time.perf_counter()-begin

        workers = []
        try:
            for phase, policy in enumerate(args.order):
                depth = args.depth_a if policy == "A" else args.depth_b
                if depth is not None:
                    def configure_depth():
                        tts._depth.mode = depth
                        tts._depth.functions.clear()
                        tts._depth.warmup(2, tts.cfg_scale, tts.instruction)
                    from voicemem.engine import gpu_loop
                    await asyncio.to_thread(gpu_loop().call, configure_depth)
                threads = args.threads_a if policy == "A" else args.threads_b
                if threads is not None:
                    def configure_threads():
                        import torch
                        for model in (models[0].model, models[2]._transcriber.model, models[6]._E2V["m"]):
                            model.kwargs["ncpu"] = threads
                            model._base_kwargs_map["kwargs"]["ncpu"] = threads
                        torch.set_num_threads(threads)
                    await asyncio.to_thread(configure_threads)
                stop, done = threading.Event(), asyncio.Event()
                workers = [asyncio.create_task(asyncio.to_thread(listen)),
                           asyncio.create_task(heartbeat()), asyncio.create_task(scores())]
                resources.enabled = (args.fixed_policy or policy) == "B"
                model_metrics(reset=True)
                first, gaps, utterances = [], [], []
                asr_start, lag_start = len(asr_times), len(loop_lags)

                async def background_job():
                    submitted = time.perf_counter()
                    durations = []
                    for index in range(args.background_jobs):
                        await asyncio.sleep(max(0, submitted+.5+index*4-time.perf_counter()))
                        durations.append(await asyncio.to_thread(background))
                    return durations

                job = asyncio.create_task(background_job())
                try:
                    with speech_scope(SpeechFeedback(), producing=False):
                        for _ in range(args.utterances):
                            begin, arrivals, audio = time.perf_counter(), [], 0
                            async for chunk in tts.stream("这是一条语音延迟测试，用来检查连续播放是否平稳。"):
                                arrivals.append(time.perf_counter()-begin)
                                audio += len(chunk)/48000
                            first.append(arrivals[0])
                            local_gaps = [b-a for a, b in zip(arrivals, arrivals[1:])]
                            gaps.extend(local_gaps)
                            utterances.append(dict(rtf=round((time.perf_counter()-begin)/audio, 3),
                                                   audio_s=round(audio, 3), gaps=summary(local_gaps)))
                    asr_result, loop_result = summary(asr_times[asr_start:]), summary(loop_lags[lag_start:])
                    background_s = await job
                    print("PHASE", json.dumps(dict(phase=phase, policy=policy, first=summary(first),
                                                  threads=threads, admission_enabled=resources.enabled,
                                                  depth=depth,
                                                  gaps=summary(gaps), utterances=utterances,
                                                  asr=asr_result, event_loop=loop_result,
                                                  background_completed=len(background_s),
                                                  background_s=[round(s, 3) for s in background_s],
                                                  admission=resources.snapshot(), paging=observer.snapshot())), flush=True)
                finally:
                    await asyncio.gather(job, return_exceptions=True)
                    stop.set(); done.set()
                    await asyncio.gather(*workers, return_exceptions=True)
                    models[0].reset()
                    workers = []
        finally:
            stop.set(); done.set()
            resources.enabled = True
            resources.wake()
            await asyncio.gather(*workers, return_exceptions=True)
            observer.stop()
            process = models[5]._proc
            if process is not None and process.poll() is None:
                process.stdin.write("__exit__\n"); process.stdin.flush()
                await asyncio.to_thread(process.wait, 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", default="ABBA", help="A disables dynamic admission; B enables it")
    parser.add_argument("--utterances", type=int, default=2)
    parser.add_argument("--background-jobs", type=int, default=1)
    parser.add_argument("--reference-audio", help="Optional explicit reference; generated PCM is never saved")
    parser.add_argument("--studio-voice", action="store_true", help="Read the configured bundled Studio reference")
    parser.add_argument("--threads-a", type=int, help="Optional CPU thread-budget crossover, phase A")
    parser.add_argument("--threads-b", type=int, help="Optional CPU thread-budget crossover, phase B")
    parser.add_argument("--fixed-policy", choices=("A", "B"), help="Hold admission fixed for a thread-budget comparison")
    parser.add_argument("--depth-a", choices=("cached", "compiled"), help="Optional depth-decoder crossover, phase A")
    parser.add_argument("--depth-b", choices=("cached", "compiled"), help="Optional depth-decoder crossover, phase B")
    args = parser.parse_args()
    if args.utterances < 1 or args.background_jobs < 1 or not args.order or set(args.order)-set("AB"):
        parser.error("use a nonempty A/B order and at least one utterance")
    if any(value is not None and value < 1 for value in (args.threads_a, args.threads_b)):
        parser.error("thread budgets must be positive")
    asyncio.run(main(args))
