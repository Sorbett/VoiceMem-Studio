"""Studio memory implementation."""
import asyncio
import re
import time
import uuid
from .background import background_lane

class Memory:
    BACKGROUND_DEFER_S = 0.5

    def save_turn_audio(self, pcm16k) -> str:
        """Archive turn PCM and return its path, or an empty string on failure."""
        if pcm16k is None or not len(pcm16k):
            return ""
        try:
            import numpy as np
            import soundfile as sf
            self.TURN_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            path = self.TURN_AUDIO_DIR / f"turn_{uuid.uuid4().hex[:12]}.wav"
            sf.write(path, np.asarray(pcm16k, dtype="float32"), 16000)
            return str(path)
        except Exception as e:
            print(f"[web] 存本轮音频失败（不影响对话）：{e}", flush=True)
            return ""

    def remember_turn(self, pending, reply: str, owner: dict, history_turn_id: str = "",
                      memory_vm=None, on_complete=None) -> None:
        """Persist the captured turn and heard reply to the originating memory instance."""

        text = pending.text
        meaningful = re.sub(r"[^\w\u4e00-\u9fff]", "", text or "")
        cjk = len(re.findall(r"[\u4e00-\u9fff]", meaningful))
        if pending.audio_path and cjk < 2 and len(meaningful) < 3:
            from voicemem.stream import SOUND_ONLY_TEXT
            text = SOUND_ONLY_TEXT

        writes = getattr(self, '_MEMORY_WRITES', None)
        if writes is None:
            writes = self._MEMORY_WRITES = set()
        write_id = uuid.uuid4().hex
        writes.add(write_id)

        def complete(result):
            if write_id not in writes:
                return
            writes.discard(write_id)
            try:
                self._finish_history_turn(history_turn_id, result)
            finally:
                if on_complete is not None:
                    on_complete(result)

        try:
            target_vm = memory_vm or self.vm
            r = target_vm.ingest(
                text, agent_reply=reply, async_facts=True,
                audio=pending.audio_path,
                language=pending.language or getattr(target_vm, "memory_language", "en"),
                on_complete=complete,
            ) or {}
        except Exception as e:
            complete({"error": True, "persistent_memory_created": False})
            print(f"[web] 存这一轮失败：{type(e).__name__}: {e}", flush=True)
            return

        affect = r.get("affect")
        if isinstance(affect, dict):
            affect = affect.get("emotion") or affect.get("label") or ""
        owner["emotion"] = str(affect or "").strip()

        if r.get("recognized_tune") and pending.audio_path:
            self._remember_tune(pending.audio_path)
        elif self.BARGE_DEBUG and self._wants_sound(pending.text or ""):
            print(f"  [replay] 这一轮没记住音乐："
                  f"tune={bool(r.get('recognized_tune'))} audio={bool(pending.audio_path)}",
                  flush=True)

        sid = r.get("speaker_id") or ""
        if not sid:

            return
        if not owner["id"]:
            owner["id"] = sid
        owner["last"] = sid
        owner["miss"] = 0 if sid == owner["id"] else owner.get("miss", 0) + 1

    def hot_path_enter(self) -> dict:
        """Mark input or reply work active and return a token for balanced release."""
        token = {"open": True}
        self._HOT["n"] += 1
        self._IDLE.clear()
        native_idle = getattr(self, "_NATIVE_IDLE", None)
        if native_idle is not None:
            native_idle.clear()
        try:
            task = asyncio.current_task()
            if task is not None:
                callback = lambda _: self.hot_path_exit(token)
                token["owner"] = (task, callback)
                task.add_done_callback(callback)
        except Exception:
            pass
        return token

    def hot_path_exit(self, token: dict) -> None:
        if not token.get("open"):
            return
        token["open"] = False
        # Capture tasks live across many turns; do not retain completed leases.
        owner = token.pop("owner", None)
        if owner is not None:
            owner[0].remove_done_callback(owner[1])
        self._HOT["n"] = max(0, self._HOT["n"] - 1)
        if self._HOT["n"] == 0:
            native_idle = getattr(self, "_NATIVE_IDLE", None)
            if native_idle is not None:
                native_idle.set()
            self._IDLE.set()

    async def wait_idle(self, what: str = "") -> None:
        """Prefer idle admission, with bounded deferral for confirmed memory."""
        if self._IDLE.is_set():
            return
        t0 = time.monotonic()
        try:
            await asyncio.wait_for(self._IDLE.wait(), self.BACKGROUND_DEFER_S)
        except asyncio.TimeoutError:
            pass
        if self.BARGE_DEBUG and what:
            print(f"  [idle] {what} 让路 {(time.monotonic()-t0)*1000:.0f}ms"
                  f"（热路径计数 {self._HOT['n']}）", flush=True)

    async def _remember_background(self, pending, reply: str, owner: dict,
                                   history_turn_id: str, memory_vm, on_complete=None) -> None:
        queued_at = time.monotonic()
        async def persist():
            waited = time.monotonic() - queued_at
            if waited > 0.05 and self.BARGE_DEBUG:
                print(f"[memory] 入库排队 {waited:.2f}s", flush=True)
            started = time.monotonic()
            loop = asyncio.get_running_loop()
            completed = loop.create_future()

            def complete(result):
                try:
                    if on_complete is not None:
                        on_complete(result)
                finally:
                    def resolve():
                        if not completed.done():
                            completed.set_result(None)
                    try:
                        loop.call_soon_threadsafe(resolve)
                    except RuntimeError:
                        pass

            from voicemem.engine import background_scope
            try:
                with background_scope(None):
                    await asyncio.to_thread(
                        self.remember_turn, pending, reply, owner, history_turn_id, memory_vm,
                        on_complete=complete)
            finally:
                # Metadata failures cannot release the lane ahead of native writes.
                await completed
            if self.BARGE_DEBUG:
                print(f"[memory] 入库完成 {time.monotonic()-started:.2f}s", flush=True)
        await background_lane(self).run(lambda: self.wait_idle("入库"), persist)

    def queue_remember_turn(self, pending, reply: str, owner: dict,
                            history_turn_id: str = "", memory_vm=None, on_complete=None) -> None:
        memory_vm = memory_vm or self.vm
        task = asyncio.create_task(
            self._remember_background(pending, reply, owner, history_turn_id, memory_vm, on_complete))
        self._REMEMBER_TASKS.add(task)

        def done(t: asyncio.Task) -> None:
            self._REMEMBER_TASKS.discard(t)
            try:
                t.result()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[web] 后台记忆任务失败：{type(e).__name__}: {e}", flush=True)

        task.add_done_callback(done)

    async def drain_memory(self):
        """Finish accepted writes on shutdown while the owning event loop lives.

        This is graceful draining, not a persistent retry journal. Forced process
        termination cannot guarantee completion of queued or in-flight writes.
        """
        idle = getattr(self, "_NATIVE_IDLE", None)
        if idle is not None:
            idle.set()
        self._IDLE.set()
        while self._REMEMBER_TASKS:
            await asyncio.gather(*tuple(self._REMEMBER_TASKS), return_exceptions=True)
        lane = background_lane(self)
        if lane.tasks:
            await asyncio.gather(*tuple(lane.tasks), return_exceptions=True)
