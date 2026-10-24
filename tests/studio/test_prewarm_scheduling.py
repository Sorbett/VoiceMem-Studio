"""CPU-only regressions: background KV warming must yield to speech replies.

No demo import, GPU/model load, network or memory DB writes.
"""
import ast
import asyncio
import contextlib
import io
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.helpers.studio import studio_tree, studio_source, execute
from voicemem.lang import detect_language
from tests.helpers.gpu import cpu_gpu_loop


def session_namespace():
    tree = studio_tree()
    names = {"prewarm_local", "stop_prewarm", "hearing", "close_session",
             "start_early", "drop_early"}
    functions = [n for n in tree.body if getattr(n, "name", "") in names]
    async def route_pending(pending, memory_vm=None, history=None):
        return pending

    ns = dict(
        asyncio=asyncio, threading=threading, time=time, detect_language=detect_language,
        space_language=lambda _: "zh",
        prewarm={"task": None, "cancelled": None, "closed": False},
        turn={"task": None, "timeline": None, "until": 0.0},
        early={"task": None}, acoustic_tasks=set(), candidate_paused=False,
        _SESSION_CONTEXT=types.SimpleNamespace(messages=lambda *a, **kw: []),
        context_session="session", ACTIVE_SPACE="test", HISTORY_TURNS=6,
        gate=types.SimpleNamespace(DEEP="deep", needs_memory=lambda route: route == "deep"),
        route_pending_thinking=route_pending,
        DIRECT="direct", MEMORY="memory",
        build_memory_context=lambda _: "memory",
        build_reply_context=lambda *a, **kw: "context", _replay_id=lambda *a: "",
        BARGE_DEBUG=False, owner={}, speech_rate=None, vm=object(),
        Pending=lambda *a, **kw: types.SimpleNamespace(),
        ReplySink=lambda *a: types.SimpleNamespace(send=None, send_audio=None),
        AudioTimeline=lambda **kw: types.SimpleNamespace(),
        sock=types.SimpleNamespace(send_json=None), send_audio=None)
    execute(functions, ns)
    ns["agent"] = ns["self"]
    return ns


class SchedulingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ns = session_namespace()
        self.calls = []
        def warm(hist, ctx, *, cancelled):
            self.calls.append((hist, ctx, cancelled))
            return 42
        self.ns["_LOCAL_LLM"] = types.SimpleNamespace(prewarm=warm)

    async def test_idle_can_warm_and_does_not_duplicate_pending_work(self):
        self.assertTrue(self.ns["prewarm_local"](object(), "你好"))
        self.assertFalse(self.ns["prewarm_local"]())
        await self.ns["prewarm"]["task"]
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][:2], ([], "context"))
        self.assertFalse(self.calls[0][2].is_set())

    async def test_early_generation_and_finished_uncommitted_audio_block_warming(self):
        task = asyncio.create_task(asyncio.sleep(0))
        self.ns["early"]["task"] = task
        self.assertFalse(self.ns["prewarm_local"]())
        await task
        self.assertFalse(self.ns["prewarm_local"]())
        self.ns["early"]["task"] = None
        self.assertTrue(self.ns["prewarm_local"]())
        await self.ns["prewarm"]["task"]
        self.assertEqual(len(self.calls), 1)

    async def test_formal_generation_and_remaining_browser_audio_block_warming(self):
        task = asyncio.create_task(asyncio.sleep(0))
        self.ns["turn"]["task"] = task
        self.assertFalse(self.ns["prewarm_local"]())
        await task
        self.ns["turn"]["until"] = time.monotonic() + 10
        self.assertFalse(self.ns["prewarm_local"]())
        self.assertEqual(self.calls, [])

    async def test_start_early_invalidates_pending_warm_without_waiting_for_worker(self):
        self.ns["prewarm_local"]()
        warm_task = self.ns["prewarm"]["task"]
        cancelled = self.ns["prewarm"]["cancelled"]
        entered = asyncio.Event()
        pending_text = []
        self.ns["Pending"] = lambda text, *a, **kw: (
            pending_text.append(text) or types.SimpleNamespace())
        async def reply(*a, **kw):
            entered.set()
            await asyncio.Event().wait()
        self.ns["voicemem_llm_tts"] = reply
        refined = asyncio.create_task(asyncio.sleep(0, result="离线复核文本"))
        try:
            await self.ns["start_early"]("你好", types.SimpleNamespace(
                memory=object(), route="deep", eot_score=.99), refined)
            await asyncio.wait_for(entered.wait(), 1)
            self.assertEqual(pending_text, ["离线复核文本"])
            self.assertTrue(cancelled.is_set())
            self.assertTrue(warm_task.cancelled())
            self.assertFalse(self.ns["prewarm_local"]())
        finally:
            await self.ns["close_session"]()

    async def test_close_cancels_warming_early_and_formal_tasks(self):
        self.ns["prewarm_local"]()
        warm_task = self.ns["prewarm"]["task"]
        early_task = asyncio.create_task(asyncio.Event().wait())
        formal_task = asyncio.create_task(asyncio.Event().wait())
        self.ns["early"]["task"] = early_task
        self.ns["turn"]["task"] = formal_task
        await self.ns["close_session"]()
        await asyncio.gather(warm_task, early_task, formal_task, return_exceptions=True)
        self.assertTrue(all(t.cancelled() for t in (warm_task, early_task, formal_task)))
        self.assertTrue(self.ns["prewarm"]["cancelled"].is_set())
        self.assertFalse(self.ns["prewarm_local"]())

    async def test_warm_failure_is_observed_and_does_not_disable_future_attempts(self):
        def fail(*a, **kw):
            raise RuntimeError("test failure")
        self.ns["_LOCAL_LLM"].prewarm = fail
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertTrue(self.ns["prewarm_local"]())
            await self.ns["prewarm"]["task"]
        self.assertIn("后台预热失败", output.getvalue())
        self.assertTrue(self.ns["prewarm_local"]())
        await self.ns["close_session"]()

    def test_normal_reply_invalidates_warm_before_early_commit_or_new_generation(self):
        tree = ast.parse((ROOT / "studio/core/core.py").read_text())
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFor))
        calls = [ast.unparse(n) for n in loop.body]
        stop = calls.index("session.stop_prewarm()")
        commit = next(i for i, text in enumerate(calls) if "session.commit_early(" in text)
        self.assertLess(stop, commit)


class QueuedWarmTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalidated_gpu_queue_entry_never_runs_model(self):
        from studio.core.utils.llm.local import LocalLLM
        llm = LocalLLM(system="synthetic persona")
        calls = []
        llm._prewarm_steps = lambda *a: calls.append(a) or iter([42])
        queued, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        def run_steps(fn, **kwargs):
            queued.set()
            if not release.wait(2):
                raise TimeoutError("test queue was not released")
            return None if kwargs["cancelled"].is_set() else list(fn())[-1]
        with patch("studio.core.utils.llm.local.gpu_loop", return_value=types.SimpleNamespace(run_steps=run_steps)):
            task = asyncio.create_task(asyncio.to_thread(
                llm.prewarm, [], "memory", cancelled=cancelled))
            try:
                self.assertTrue(await asyncio.to_thread(queued.wait, 1))
                cancelled.set()
            finally:
                release.set()
            self.assertEqual(await task, 0)
        self.assertEqual(calls, [])

    def test_startup_and_uncancelled_warming_preserve_arguments_and_result(self):
        from studio.core.utils.llm.local import LocalLLM
        llm = LocalLLM(system="synthetic persona")
        calls = []
        llm._prewarm_steps = lambda *a: calls.append(a) or iter([42])
        with cpu_gpu_loop() as gpu, patch("studio.core.utils.llm.local.gpu_loop", return_value=gpu):
            self.assertEqual(llm.prewarm(), 42)
            self.assertEqual(llm.prewarm(["history"], "memory", cancelled=threading.Event()), 42)
            cancelled = threading.Event()
            cancelled.set()
            self.assertEqual(llm.prewarm(cancelled=cancelled), 0)
        self.assertEqual(calls[0], (None, "", None))
        self.assertEqual(calls[1][:2], (["history"], "memory"))

    async def test_audio_runs_between_materialized_prefix_blocks(self):
        from studio.core.utils.llm.local import LocalLLM
        llm = LocalLLM(system="synthetic persona")
        entered, release = threading.Event(), threading.Event()
        calls = []
        cache = [types.SimpleNamespace(state=[])]

        def forward(tokens, *, cache):
            calls.append(("prefix", tokens.shape[1], threading.get_ident()))
            cache[0].state.extend(tokens[0].tolist())
            if len(calls) == 1:
                entered.set()
                release.wait(2)

        llm.load = lambda: (forward, None)
        llm._prefix_ids = lambda *_: list(range(768))
        llm._pick_base = lambda _: (cache, 0, "")
        with cpu_gpu_loop() as gpu, patch("studio.core.utils.llm.local.gpu_loop", return_value=gpu):
            warm = asyncio.create_task(asyncio.to_thread(llm.prewarm))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                def audio():
                    calls.append(("audio", 0, threading.get_ident()))
                    yield b"pcm"
                job = gpu.iter(audio, solo_first=True)
            finally:
                release.set()
            self.assertEqual(await asyncio.wait_for(warm, 2), 768)
            self.assertEqual(job.out.get(timeout=1), ("ok", b"pcm"))
            self.assertIsNone(job.out.get(timeout=1))
            self.assertTrue(all(row[2] == gpu._thread.ident for row in calls))
        self.assertEqual([row[0] for row in calls], ["prefix", "audio", "prefix", "prefix"])
        self.assertEqual(cache[0].state, list(range(768)))
        self.assertIs(llm._base, cache)

    async def test_cancellation_or_prompt_invalidation_never_publishes_partial_cache(self):
        from studio.core.utils.llm.local import LocalLLM
        for invalidate in ("cancel", "system", "drop"):
            with self.subTest(invalidate=invalidate):
                llm = LocalLLM(system="synthetic persona")
                entered, release, cancelled = (threading.Event() for _ in range(3))
                calls = []
                def forward(tokens, *, cache):
                    calls.append(tokens.shape[1])
                    entered.set()
                    release.wait(2)
                llm.load = lambda: (forward, None)
                llm._prefix_ids = lambda *_: list(range(768))
                llm._pick_base = lambda _: ([types.SimpleNamespace(state=None)], 0, "")
                with cpu_gpu_loop() as gpu, patch("studio.core.utils.llm.local.gpu_loop", return_value=gpu):
                    task = asyncio.create_task(asyncio.to_thread(llm.prewarm, cancelled=cancelled))
                    try:
                        self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                        if invalidate == "cancel":
                            cancelled.set()
                        elif invalidate == "system":
                            llm.set_system("replacement persona")
                        else:
                            llm.drop_cache()
                    finally:
                        release.set()
                    self.assertEqual(await asyncio.wait_for(task, 2), 0)
                self.assertEqual(calls, [256])
                self.assertIsNone(llm._cache)
                self.assertIsNone(llm._base)

    async def test_failed_prefill_keeps_previous_cache_and_allows_retry(self):
        from studio.core.utils.llm.local import LocalLLM
        llm = LocalLLM(system="synthetic persona")
        previous = llm._cache = [types.SimpleNamespace(state="complete")]
        llm._cached_prefix = [1]
        llm._prefix_ids = lambda *_: list(range(512))
        llm._pick_base = lambda _: ([types.SimpleNamespace(state=None)], 0, "")
        def fail(*args, **kwargs):
            raise ValueError("synthetic prefill failure")
        llm.load = lambda: (fail, None)
        with cpu_gpu_loop() as gpu, patch("studio.core.utils.llm.local.gpu_loop", return_value=gpu):
            with self.assertRaisesRegex(ValueError, "synthetic prefill failure"):
                await asyncio.to_thread(llm.prewarm)
            self.assertIs(llm._cache, previous)
            llm.load = lambda: (lambda *args, **kwargs: None, None)
            self.assertEqual(await asyncio.to_thread(llm.prewarm), 512)
            self.assertEqual(llm._cached_prefix, list(range(512)))


class GenerationPrefillTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_hit_and_miss_preserve_tokens_options_and_visible_text(self):
        from studio.core.utils.llm.local import LocalLLM
        for hit in (False, True):
            with self.subTest(hit=hit):
                llm = LocalLLM(system="synthetic persona")
                tokens = list(range(600))
                calls, generated = [], []
                cache_module = types.ModuleType("mlx_lm.models.cache")
                cache_module.make_prompt_cache = lambda _: [types.SimpleNamespace(state=[])]
                prefix = [types.SimpleNamespace(state=tokens[:2])]
                if hit:
                    llm._cache, llm._cached_prefix = prefix, tokens[:2]
                llm._fork = lambda src: [types.SimpleNamespace(state=list(src[0].state))]
                def model(batch, *, cache):
                    calls.append((batch.shape[1], threading.get_ident()))
                    cache[0].state.extend(batch[0].tolist())
                llm.load = lambda: (model, types.SimpleNamespace(encode=lambda _: tokens))
                llm._render, llm._msgs = lambda _: "prompt", lambda *args: []
                language_model = types.ModuleType("mlx_lm")
                def generate(model, tokenizer, remaining, **kwargs):
                    self.assertEqual(remaining, [599])
                    self.assertEqual(kwargs["max_tokens"], 512)
                    import mlx.core as mx
                    model(mx.array(remaining)[None], cache=kwargs["prompt_cache"])
                    generated.append(list(kwargs["prompt_cache"][0].state))
                    yield types.SimpleNamespace(text="visible reply")
                language_model.stream_generate = generate
                with cpu_gpu_loop() as gpu, \
                        patch("studio.core.utils.llm.local.gpu_loop", return_value=gpu), \
                        patch.dict(sys.modules, {"mlx_lm": language_model,
                                                  "mlx_lm.models.cache": cache_module}):
                    self.assertEqual([text async for text in llm("input")], ["visible reply"])
                    self.assertTrue(all(owner == gpu._thread.ident for _, owner in calls))
                self.assertEqual(generated, [tokens])
                self.assertLessEqual(max(size for size, _ in calls), 256)
                self.assertEqual(prefix[0].state, tokens[:2])

    async def test_cancel_during_generation_prefill_skips_sampling_and_later_blocks(self):
        from studio.core.utils.llm.local import LocalLLM
        llm = LocalLLM(system="synthetic persona")
        llm._cache, llm._cached_prefix = [types.SimpleNamespace(state=None)], [0]
        llm._fork = lambda _: [types.SimpleNamespace(state=None)]
        entered, release = threading.Event(), threading.Event()
        calls = []
        def model(batch, *, cache):
            calls.append(batch.shape[1])
            entered.set()
            release.wait(2)
        llm.load = lambda: (model, types.SimpleNamespace(encode=lambda _: list(range(768))))
        llm._render, llm._msgs = lambda _: "prompt", lambda *args: []
        language_model = types.ModuleType("mlx_lm")
        language_model.stream_generate = unittest.mock.Mock(return_value=iter([]))
        with cpu_gpu_loop() as gpu, \
                patch("studio.core.utils.llm.local.gpu_loop", return_value=gpu), \
                patch.dict(sys.modules, {"mlx_lm": language_model}):
            stream = llm("input")
            task = asyncio.create_task(anext(stream))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
            self.assertEqual([v async for v in gpu.aiter(lambda: iter(["next"]))], ["next"])
        self.assertEqual(calls, [256])
        language_model.stream_generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
