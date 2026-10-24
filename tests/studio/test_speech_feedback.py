"""Output ownership and native cancellation for speech scheduling feedback."""
import asyncio
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from studio.core.utils.audio_timeline.component import AudioTimeline
from studio.core.utils.conversation.component import Conversation
from studio.core.utils.tts.component import BreezeMLXTTS
from tests.helpers.gpu import cpu_gpu_loop
from voicemem.engine import SpeechFeedback, resources


class SpeechFeedbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_current_output_can_replace_scheduling_inventory(self):
        session = Conversation.__new__(Conversation)
        timeline = AudioTimeline()
        feedback = timeline.inference_feedback = SpeechFeedback()
        session.agent = SimpleNamespace(MIC_RATE=24000, BARGE_DEBUG=False)
        session.turn = dict(timeline=timeline, play_started=True, until=1)
        event = dict(output_id="obsolete", rendered_samples=0,
                     buffered_samples=24000, sample_rate=24000, state="paused")
        await session.playback_checkpoint(event)
        self.assertEqual(feedback.snapshot()["buffer_s"], 0)
        event["output_id"] = timeline.output_id
        await session.playback_checkpoint(event)
        self.assertEqual(feedback.snapshot()["buffer_s"], 1)
        event["buffered_samples"], event["state"] = 0, "stalled"
        await session.playback_checkpoint(event)
        self.assertEqual(feedback.snapshot()["buffer_s"], 0)

    async def test_cancelled_consumer_retains_reservation_until_native_cleanup(self):
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        tts = BreezeMLXTTS(model="synthetic")

        def segments(*_):
            try:
                entered.set()
                release.wait(2)
                yield b"pcm"
            finally:
                closed.set()

        tts._segments = segments
        baseline = resources.snapshot()["speech_jobs"]
        with cpu_gpu_loop() as gpu, patch("voicemem.engine.gpu_loop", return_value=gpu):
            stream = tts.stream("synthetic fixture")
            task = asyncio.create_task(anext(stream))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await stream.aclose()
                self.assertEqual(resources.snapshot()["speech_jobs"], baseline+1)
                self.assertFalse(closed.is_set())
            finally:
                release.set()
                self.assertTrue(await asyncio.to_thread(closed.wait, 1))
                # The cleanup reservation is released after the inner generator.
                for _ in range(100):
                    if resources.snapshot()["speech_jobs"] == baseline:
                        break
                    await asyncio.sleep(.001)
                self.assertEqual(resources.snapshot()["speech_jobs"], baseline)
