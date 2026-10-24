"""Synthetic audio regressions for decoder restarts inside an unfinished turn."""
import asyncio
import types
import unittest
from unittest.mock import patch

import numpy as np

from studio.core.utils.turn_taking.pause import PauseGate
from studio.core.utils.conversation.component import Conversation
from tests.helpers.capture import anticipate_namespace
from voicemem.stream import VoiceStream


class ContinuationAudio:
    """Run real streaming and pause policy with independently controlled decoders."""

    def __init__(self):
        self.text = ""
        self.speaking = False
        self.flush_ready = False
        self.final_result = ""
        self.resets = 0
        self.decoded_audio = []
        self.pause = PauseGate()
        self.stream = VoiceStream(
            types.SimpleNamespace(), src_rate=16000, gate=lambda _: "shallow",
            eot=None, confirm_s=.2, textless_confirm_s=.2,
            spec_min_chars=999, gamble_s=999, turn_end_guard=self.pause.allow_end)
        self.stream._asr_w = types.SimpleNamespace(
            push=lambda _: self.text, flush=self.flush, reset=self.reset,
            report=lambda: "synthetic", close=lambda: None,
            t=types.SimpleNamespace(join=lambda _: None))
        self.stream._vad = types.SimpleNamespace(is_speech=lambda _: self.speaking)
        self.stream._final_text_async = self.final_text

    def reset(self):
        self.resets += 1
        self.text = ""

    def flush(self):
        future = asyncio.get_running_loop().create_future()
        if self.flush_ready:
            future.set_result(self.text)
        return future

    async def final_text(self, pcm, **kwargs):
        self.decoded_audio.append(pcm.copy())
        if self.final_result is None:
            await asyncio.Event().wait()
        if isinstance(self.final_result, Exception):
            raise self.final_result
        return self.final_result

    async def feed(self, seconds, *, speaking=False, text=None):
        self.speaking = speaking
        if text is not None:
            self.text = text
        raw = np.full(320, 1200 if speaking else 0, np.int16).tobytes()
        return [await self.stream.feed(raw) for _ in range(round(seconds / .02))]

    async def hold_prefix(self, prefix="我准备开一个频道，然后。", *, textless=False):
        self.final_result = prefix
        await self.feed(.16, speaking=True, text="" if textless else "我准备开一个频道")
        states = await self.feed(.22)
        if any(state.turn for state in states):
            raise AssertionError("An unfinished prefix must stay in the current turn")
        return states[-1]


class ASRContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixtures = []
        self.grace = patch("voicemem.stream.ASR_FINAL_GRACE_S", 0)
        self.grace.start()

    async def asyncTearDown(self):
        for fixture in self.fixtures:
            await fixture.stream.aclose()
        self.grace.stop()

    def fixture(self):
        fixture = ContinuationAudio()
        self.fixtures.append(fixture)
        return fixture

    async def test_restart_preserves_prefix_in_partials_and_confirmed_turn(self):
        fixture = self.fixture()
        prefix = "我准备开一个频道，然后。"
        held = await fixture.hold_prefix(prefix)
        self.assertEqual(held.text, prefix)
        self.assertEqual(fixture.resets, 1)
        states = await fixture.feed(.04, speaking=True)
        self.assertEqual(states[-1].text, prefix)
        states = await fixture.feed(.16, speaking=True, text="请列出实施步骤")
        expected = prefix + "请列出实施步骤"
        self.assertTrue(all(state.text == expected for state in states))
        fixture.final_result = None
        fixture.flush_ready = True
        turns = [state.turn for state in await fixture.feed(.22) if state.turn]
        self.assertEqual([turn.text for turn in turns], [expected])
        self.assertEqual(turns[0].raw_text, expected)
        self.assertGreater(len(fixture.decoded_audio[-1]), len(fixture.decoded_audio[0]))

    async def test_repeated_pauses_keep_all_segments_exactly_once(self):
        fixture = self.fixture()
        await fixture.hold_prefix()
        await fixture.feed(.16, speaking=True, text="先做教程")
        prefix = "我准备开一个频道，然后先做教程，然后。"
        fixture.final_result = prefix
        states = await fixture.feed(.22)
        self.assertTrue(all(state.turn is None for state in states))
        self.assertEqual(fixture.resets, 2)
        states = await fixture.feed(.16, speaking=True, text="再做访谈")
        self.assertEqual(states[-1].text, prefix + "再做访谈")
        fixture.final_result = None
        fixture.flush_ready = True
        turns = [state.turn for state in await fixture.feed(.22) if state.turn]
        self.assertEqual([turn.text for turn in turns], [prefix + "再做访谈"])

    async def test_full_refinement_replaces_joined_text_without_duplicate_prefix(self):
        fixture = self.fixture()
        await fixture.hold_prefix()
        await fixture.feed(.16, speaking=True, text="请列出实施步骤")
        fixture.flush_ready = True
        fixture.final_result = "我准备开一个播客频道，然后请列出实施步骤。"
        turns = [state.turn for state in await fixture.feed(.22) if state.turn]
        self.assertEqual([turn.text for turn in turns], [fixture.final_result])
        self.assertEqual(turns[0].raw_text, "我准备开一个频道，然后。请列出实施步骤")

    async def test_matching_full_refinement_keeps_complete_raw_transcript(self):
        fixture = self.fixture()
        await fixture.hold_prefix()
        await fixture.feed(.16, speaking=True, text="请列出实施步骤")
        fixture.flush_ready = True
        fixture.final_result = "我准备开一个频道，然后。请列出实施步骤"
        turns = [state.turn for state in await fixture.feed(.22) if state.turn]
        self.assertEqual([turn.text for turn in turns], [fixture.final_result])
        self.assertEqual(turns[0].raw_text, fixture.final_result)

    async def test_failed_or_empty_refinement_falls_back_to_entire_streamed_turn(self):
        for result in ("", RuntimeError("synthetic decode failure")):
            with self.subTest(result=type(result).__name__):
                fixture = self.fixture()
                await fixture.hold_prefix()
                await fixture.feed(.16, speaking=True, text="请列出实施步骤")
                fixture.flush_ready = True
                fixture.final_result = result
                turns = [state.turn for state in await fixture.feed(.22) if state.turn]
                self.assertEqual([turn.text for turn in turns], [
                    "我准备开一个频道，然后。请列出实施步骤"])

    async def test_textless_probe_can_hold_then_resume_without_losing_its_decode(self):
        fixture = self.fixture()
        await fixture.hold_prefix(textless=True)
        self.assertEqual(fixture.resets, 1)
        await fixture.feed(.16, speaking=True, text="请列出实施步骤")
        fixture.final_result = None
        fixture.flush_ready = True
        turns = [state.turn for state in await fixture.feed(.22) if state.turn]
        self.assertEqual([turn.text for turn in turns], [
            "我准备开一个频道，然后。请列出实施步骤"])

    async def test_timeout_restart_preserves_latest_text_for_resumed_input(self):
        fixture = self.fixture()
        await fixture.feed(.16, speaking=True, text="请解释这个设计")
        fixture.final_result = None
        with patch("voicemem.stream.ASR_FINISH_TIMEOUT_S", .01):
            await fixture.stream._finish_asr(np.ones(160, np.float32))
        self.assertEqual(fixture.resets, 1)
        states = await fixture.feed(.16, speaking=True, text="以及实现步骤")
        self.assertEqual(states[-1].text, "请解释这个设计以及实现步骤")

    async def test_cumulative_decoder_without_restart_is_not_prefixed_twice(self):
        fixture = self.fixture()
        fixture.flush_ready = True
        await fixture.hold_prefix()
        self.assertEqual(fixture.resets, 0)
        full = "我准备开一个频道然后请列出实施步骤"
        states = await fixture.feed(.16, speaking=True, text=full)
        self.assertEqual(states[-1].text, full)
        fixture.final_result = None
        turns = [state.turn for state in await fixture.feed(.22) if state.turn]
        self.assertEqual([turn.text for turn in turns], [full])

    async def test_next_turn_and_other_connections_do_not_inherit_prefix(self):
        fixture = self.fixture()
        await fixture.hold_prefix()
        other = self.fixture()
        states = await other.feed(.16, speaking=True, text="另一个连接")
        self.assertEqual(states[-1].text, "另一个连接")
        fixture.final_result = None
        fixture.flush_ready = True
        await fixture.feed(.16, speaking=True, text="请列出实施步骤")
        turns = [state.turn for state in await fixture.feed(.22) if state.turn]
        self.assertEqual(len(turns), 1)
        states = await fixture.feed(.16, speaking=True, text="下一个问题")
        self.assertEqual(states[-1].text, "下一个问题")

    async def test_snapshot_fallback_freezes_entire_joined_text(self):
        fixture = self.fixture()
        await fixture.hold_prefix()
        await fixture.feed(.16, speaking=True, text="请列出实施步骤")
        fixture.final_result = ""
        snapshot = fixture.stream.refine_current_snapshot()
        await fixture.feed(.04, speaking=True, text="请列出实施步骤和工具")
        self.assertEqual(await snapshot, "我准备开一个频道，然后。请列出实施步骤")
        self.assertEqual(fixture.stream._text, "我准备开一个频道，然后。请列出实施步骤和工具")

    async def test_english_boundaries_and_repeated_words_survive_restart(self):
        for prefix, tail, expected in (
            ("I want a plan and then", "please list the steps",
             "I want a plan and then please list the steps"),
            ("I want a plan, and then.", "Please list the steps",
             "I want a plan, and then. Please list the steps"),
            ("先检查，然后", "然后再检查", "先检查，然后然后再检查"),
        ):
            with self.subTest(prefix=prefix):
                fixture = self.fixture()
                await fixture.hold_prefix(prefix)
                states = await fixture.feed(.16, speaking=True, text=tail)
                self.assertEqual(states[-1].text, expected)
                fixture.final_result = None
                fixture.flush_ready = True
                turns = [state.turn for state in await fixture.feed(.22) if state.turn]
                self.assertEqual([turn.text for turn in turns], [expected])

    async def test_cancelled_finalization_keeps_prefix_and_accepts_further_speech(self):
        fixture = self.fixture()
        await fixture.hold_prefix()
        await fixture.feed(.16, speaking=True, text="请列出实施步骤")
        entered = asyncio.Event()

        async def stalled_decode(pcm, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        fixture.stream._final_text_async = stalled_decode
        task = asyncio.create_task(fixture.feed(.22))
        try:
            await asyncio.wait_for(entered.wait(), 1)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        states = await fixture.feed(.16, speaking=True, text="请列出实施步骤和工具")
        self.assertEqual(states[-1].text, "我准备开一个频道，然后。请列出实施步骤和工具")

    async def test_capture_publishes_entire_resumed_turn_with_one_input_identity(self):
        fixture = self.fixture()
        fixture.stream.src_rate = 24000
        ns = anticipate_namespace()
        prefix = "我准备开一个频道，然后。"
        expected = prefix + "请列出实施步骤"
        frames = iter(
            [(True, "我准备开一个频道", prefix, False)] * 8
            + [(False, None, prefix, False)] * 11
            + [(True, "请列出实施步骤", None, True)] * 8
            + [(False, None, None, True)] * 11)
        sent = []

        class Socket:
            async def receive(self):
                frame = next(frames, None)
                if frame is None:
                    return {"type": "websocket.disconnect"}
                speaking, text, final, ready = frame
                fixture.speaking = speaking
                if text is not None:
                    fixture.text = text
                fixture.final_result, fixture.flush_ready = final, ready
                return {"bytes": np.full(480, 1200 if speaking else 0, np.int16).tobytes()}

            async def send_json(self, message):
                sent.append(message)

        def open_stream(**kwargs):
            fixture.stream.turn_end_guard = kwargs["turn_end_guard"]
            return fixture.stream

        ns["vm"] = types.SimpleNamespace(stream=open_stream)
        socket = Socket()
        with patch("studio.core.utils.turn_taking.backchannel.emitting", return_value=False), \
                patch("studio.core.utils.turn_taking.backchannel.Backchannel.offer", return_value=None):
            async for pending in ns["anticipate"](socket):
                self.assertEqual(pending.text, expected)
                await Conversation.publish_user_input(types.SimpleNamespace(sock=socket), pending)
        finals = [event for event in sent if event["type"] == "user_transcript"]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["text"], expected)
        self.assertTrue(finals[0]["input_turn_id"])
        self.assertTrue(all(event["input_turn_id"] == finals[0]["input_turn_id"]
                            for event in sent if event["type"] == "partial_transcript"))


if __name__ == "__main__":
    unittest.main()
