"""Offline native MLX prefill fairness and numerical-equivalence probe.

Uses a seeded small Qwen3.5 architecture with both recurrent and attention
caches, not pretrained weights. A is one native prefix call; B yields between
token blocks. The competing job is a materialized arithmetic probe, not speech.
No API, Memory Space, audio, model download or persistent output is used.
These timings do not measure production-model speed or perceived audio latency.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from importlib.metadata import version
import platform
import statistics
import threading
import time


def main(args):
    from voicemem.engine import gpu_loop
    from voicemem.engine.prefill import prefill_steps

    gpu = gpu_loop()

    def load():
        import mlx.core as mx
        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
        mx.random.seed(7)
        model = TextModel(TextModelArgs(
            hidden_size=256, intermediate_size=512, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=64,
            vocab_size=128, linear_num_value_heads=4, linear_num_key_heads=4,
            linear_key_head_dim=64, linear_value_head_dim=64,
            full_attention_interval=2))
        mx.eval(model.parameters())
        return model

    model = gpu.call(load)
    tokens = [index % 128 for index in range(args.tokens)]

    def native(stepped, entered, timings):
        import mlx.core as mx
        cache = model.make_cache()
        entered.set()
        if stepped:
            steps = prefill_steps(model, tokens, cache, step_size=args.step_size, keep_last=True)
            while True:
                started = time.perf_counter()
                try:
                    count = next(steps)
                except StopIteration:
                    break
                timings.append(time.perf_counter()-started)
                yield count
        else:
            started = time.perf_counter()
            model(mx.array(tokens[:-1])[None], cache=cache)
            mx.eval([item.state for item in cache])
            timings.append(time.perf_counter()-started)
            yield len(tokens)-1
        yield cache

    def verify(left, right):
        import mlx.core as mx
        errors = []
        def compare(a, b):
            if isinstance(a, (list, tuple)):
                for x, y in zip(a, b, strict=True):
                    compare(x, y)
            elif a is not None:
                errors.append(float(mx.max(mx.abs(a-b))))
        for a, b in zip(left, right, strict=True):
            compare(a.state, b.state)
        greedy, token = [], tokens[-1]
        for _ in range(8):
            a = model(mx.array([[token]]), cache=left)[:, -1, :]
            b = model(mx.array([[token]]), cache=right)[:, -1, :]
            x, y = int(mx.argmax(a)), int(mx.argmax(b))
            if x != y:
                raise AssertionError("Chunked prefill changed the fixture's greedy token")
            compare(a, b)
            greedy.append(x)
            token = x
        maximum = max(errors, default=0.)
        if maximum > 1e-3:
            raise AssertionError(f"Fixture numerical difference exceeds tolerance: {maximum}")
        return dict(max_abs_error=maximum, identical_greedy_tokens=len(greedy))

    reference = [gpu.run_steps(lambda: native(mode, threading.Event(), [])) for mode in (False, True)]
    equivalence = gpu.call(lambda: verify(*reference))
    del reference
    rows = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for policy in args.order:
            timings, entered = [], threading.Event()
            started = time.perf_counter()
            work = pool.submit(gpu.run_steps, lambda: native(policy == "B", entered, timings))
            if not entered.wait(10):
                work.result(timeout=10)
                raise TimeoutError("Native prefix did not start")
            submitted = time.perf_counter()
            def interactive():
                import mlx.core as mx
                value = float(mx.sum(mx.sin(mx.arange(1024, dtype=mx.float32))))
                yield value, time.perf_counter()
            job = gpu.iter(interactive, solo_first=True, label="prefill_probe")
            kind, payload = job.out.get(timeout=30)
            if kind != "ok":
                raise payload
            _, finished = payload
            if job.out.get(timeout=30) is not None:
                raise RuntimeError("Native probe did not finish")
            result = work.result(timeout=30)
            del result
            rows.append(dict(policy=policy, prefix_ms=round((time.perf_counter()-started)*1000, 3),
                             max_step_ms=round(max(timings)*1000, 3),
                             probe_wait_ms=round((finished-submitted)*1000, 3)))
    summary = {policy: {key: round(statistics.mean(row[key] for row in rows if row["policy"] == policy), 3)
                        for key in ("prefix_ms", "max_step_ms", "probe_wait_ms")}
               for policy in set(args.order)}
    print(json.dumps(dict(python=platform.python_version(), machine=platform.machine(),
                          mlx=version("mlx"), mlx_lm=version("mlx-lm"), tokens=args.tokens,
                          step_size=args.step_size, equivalence=equivalence,
                          summary=summary, phases=rows), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--step-size", type=int, default=256)
    parser.add_argument("--order", default="ABBAABBA")
    args = parser.parse_args()
    if not 2 <= args.tokens <= 8192 or args.step_size < 1 or not args.order or set(args.order)-set("AB"):
        parser.error("Use 2–8192 tokens, a positive step size and an A/B phase order")
    main(args)
