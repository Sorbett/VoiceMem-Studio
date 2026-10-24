"""Measure exact-input EOT reuse with synthetic speech-like audio and extra silence.

Run from the repository root with Python 3.12 and an installed EOT model. No
network, microphone, audio files, Memory Space or GPU is used. ABBA compares
native recomputation with digest/score reuse after a shared model warmup.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main(repeats):
    import numpy as np
    from studio.paths import MODELS
    from voicemem.engine import model_metrics
    from voicemem.utils.audio.eot import EndOfTurn

    model = EndOfTurn(model_path=str(MODELS / "eot/smart-turn-v3.2-cpu.onnx"))
    times = np.arange(48000, dtype=np.float32)/16000
    speech = (.05*np.sin(2*np.pi*220*times)+.025*np.sin(2*np.pi*440*times)).astype(np.float32)
    waveforms = [np.concatenate((speech, np.zeros(4800+index*320, np.float32)))
                 for index in range(repeats)]
    for _ in range(5):
        model._score_cache = None
        model.score(waveforms[0])
    print("ENVIRONMENT", json.dumps(dict(python=platform.python_version(),
                                        machine=platform.machine(), onnxruntime=version("onnxruntime"),
                                        repeats=repeats)), flush=True)
    expected = None
    for policy in "ABBA":
        model_metrics(reset=True)
        samples, scores = [], []
        for waveform in waveforms:
            if policy == "A":
                model._score_cache = None
            started = time.perf_counter()
            scores.append(model.score(waveform))
            samples.append((time.perf_counter()-started)*1000)
        if expected is None:
            expected = scores[0]
        if any(score != expected for score in scores):
            raise AssertionError("stable-input EOT scores changed")
        rows = model_metrics()
        print("PHASE", json.dumps(dict(policy=policy, mean_ms=statistics.mean(samples),
                                      p95_ms=sorted(samples)[min(len(samples)-1, int(.95*len(samples)))],
                                      max_ms=max(samples), scores_identical=True,
                                      native_calls=sum(r["calls"] for r in rows if r["backend"] == "onnx"),
                                      cache_hits=sum(r["calls"] for r in rows if r["backend"] == "cache"))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    main(args.repeats)
