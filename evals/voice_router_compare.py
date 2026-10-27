"""Compare local ONNX depth routing with MLX Qwen on synthetic bilingual inputs.

Run with the Studio Python environment on native Apple Silicon. No APIs, audio,
Memory Spaces or private traces are used. This is a routing-only benchmark.
"""
from __future__ import annotations

import argparse
import asyncio
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time

from evals.router_quality import CASES, HOLDOUT_CASES


EXTRA_CASES = (
    ('Recite the first thousand digits of pi, without a derivation.', (), 'direct'),
    ('Hello, how are you?', (), 'direct'),
    ('What is the difference between tea and coffee?', (), 'direct'),
    ('What did I say my budget was?', (), 'memory'),
    ('What does "deep reasoning" mean? Do not enable it.', (), 'direct'),
    ('Take your time, I am still thinking.', (), 'direct'),
    ('I feel a little sad and just want to talk.', (), 'direct'),
    ('What is seventeen plus twenty-five?', (), 'direct'),
    ('Just give the final answer from the calculation you already finished.',
     (('user', 'Derive the integral.'), ('assistant', 'The proof is complete; the result is one quarter.')), 'direct'),
    ('Continue the story.', (('user', 'Tell me a bedtime story.'),
                            ('assistant', 'A cat walked into a glowing forest.')), 'direct'),
    ('Please think deeply about this problem before answering.', (), 'memory_cot'),
    ('Rigorously prove convergence and handle all parameter ranges and boundary cases.', (), 'memory_cot'),
    ('Design a database migration with no downtime, no data loss, a limited budget, and a verified rollback strategy.', (), 'memory_cot'),
    ('Diagnose an intermittent deadlock by analyzing lock ordering and design a validation experiment.', (), 'memory_cot'),
    ('Use my workload and budget to weigh scaling costs, peak latency, and recovery guarantees.', (), 'memory_cot'),
    ('Continue the remaining proof.', (('user', 'Prove this distributed protocol is safe.'),
                                     ('assistant', 'Only the single-failure case is complete; the double-failure case remains.')), 'memory_cot'),
    ('Please reason deeply, but keep the reasoning process private and give me the conclusion.', (), 'memory_cot'),
    ('Can you carefully compare the interacting effects of retries, timeouts and idempotency under network faults?', (), 'memory_cot'),
    ('请背诵圆周率小数点后一千位，不用推导。', (), 'direct'),
    ('不要展示思考过程，但请深入分析三个方案的风险和收益。', (), 'memory_cot'),
    ('用一句英文告诉我，刚才已经算出的结果是多少。', (), 'direct'),
    ('我说的是解释 reasoning 这个单词，不是要进入深度思考。', (), 'direct'),
    ('请分析 retry、timeout 和幂等如何共同导致重复扣款，并设计验证修复方案。', (), 'memory_cot'),
    ('帮我解释一下 API 是什么意思，简单说。', (), 'direct'),
)


def cases():
    """Return public synthetic checks; training independence is not established."""
    rows = []
    for suite, items in (('existing', CASES), ('existing_holdout', HOLDOUT_CASES), ('bilingual_edges', EXTRA_CASES)):
        for index, (text, history, expected) in enumerate(items):
            rows.append({'id': f'{suite}-{index}', 'suite': suite, 'text': text,
                         'history': [{'role': role, 'content': content} for role, content in history],
                         'deep': expected == 'memory_cot',
                         'language': 'zh' if any('\u3400' <= char <= '\u9fff' for char in text) else 'en'})
    return rows


def summarize(rows):
    times = sorted(row['ms'] for row in rows)
    ordinary = [row for row in rows if not row['deep']]
    deep = [row for row in rows if row['deep']]
    return {'n': len(rows), 'correct': sum(row['deep'] == row['predicted_deep'] for row in rows),
            'false_deep': sum(row['predicted_deep'] for row in ordinary), 'ordinary_n': len(ordinary),
            'missed_deep': sum(not row['predicted_deep'] for row in deep), 'deep_n': len(deep),
            'median_ms': statistics.median(times), 'p95_ms': times[int(.95 * (len(times) - 1))],
            'mean_ms': statistics.mean(times)}


def memory():
    """Record process RSS in bytes; accelerator allocation is reported separately."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'peak_rss_bytes': int(value if platform.system() == 'Darwin' else value * 1024)}


async def main(args):
    from studio.core.utils.reply_modes.encoder_depth import EncoderDepthRouter
    from studio.core.utils.reply_modes.mlx import MLXThinkingRouter
    from studio.harness.prompts import effective_prompts, prompt_scope
    scope = effective_prompts({'prompt_language': args.prompt_language})
    report = {'scope': 'uncached_depth_only_synthetic_no_audio_no_api',
              'training_independence': 'not established', 'platform': platform.platform(),
              'machine': platform.machine(), 'order': args.order, 'prompt_language': args.prompt_language,
              'versions': {name: version(name) for name in ('onnxruntime', 'transformers', 'mlx', 'mlx-lm')},
              'cold': {}, 'phases': []}
    report['chip'] = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    report['ram_bytes'] = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
    routers = {'A': MLXThinkingRouter(),
               'B': EncoderDepthRouter(args.checkpoint, backend='onnx', allow_prototype=True)}
    for label in ('A', 'B'):
        started = time.perf_counter()
        with prompt_scope(scope):
            await routers[label].classify_async('介绍一下向量数据库')
        report['cold'][label] = {'load_and_first_ms': (time.perf_counter() - started) * 1000, **memory()}
    for index, label in enumerate(args.order):
        router, results = routers[label], []
        cpu_started = time.process_time()
        for row in cases():
            router._cache.clear()
            started = time.perf_counter()
            with prompt_scope(scope):
                decision = await router.classify_async(row['text'], history=row['history'])
            results.append({**row, 'predicted_deep': decision.level == 'slow', 'raw': decision.raw,
                            'ms': (time.perf_counter() - started) * 1000})
        phase = {'index': index, 'router': label, **summarize(results),
                 'process_cpu_s': time.process_time() - cpu_started, **memory(),
                 'by_language': {language: summarize([row for row in results if row['language'] == language])
                                 for language in ('zh', 'en')}, 'cases': results}
        report['phases'].append(phase)
        print(json.dumps({key: value for key, value in phase.items() if key != 'cases'}, ensure_ascii=False), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('studio/models/reply-router/voice-router-v6/onnx'))
    parser.add_argument('--order', default='ABBA')
    parser.add_argument('--prompt-language', choices=('zh', 'en'), default='zh')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.order or set(args.order) - {'A', 'B'}:
        parser.error('--order must contain only A and B')
    asyncio.run(main(args))
