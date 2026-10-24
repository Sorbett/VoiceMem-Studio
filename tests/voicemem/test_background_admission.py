"""Opt-in perception admission outside model locks and captured worker contexts."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import unittest

from voicemem.lang import contextualize
from voicemem.utils.audio.perceiver import AudioPerceiver
from voicemem.engine import background_checkpoint, background_scope


class BackgroundAdmissionTests(unittest.TestCase):
    def test_perception_rechecks_admission_before_each_native_stage(self):
        calls = []
        perceiver = object.__new__(AudioPerceiver)
        perceiver._enable_scene = perceiver._enable_voiceprint = perceiver._enable_emotion = True
        perceiver._enable_music = perceiver._enable_abnormal_sound = False
        def scene(_):
            calls.append('scene')
            return '', '', None, [], None, [], {}
        def speaker(*_):
            calls.append('speaker')
            return None, 'Speaker 0', False, None
        def emotion(_):
            calls.append('emotion')
            return 'calm'
        perceiver._detect_scene = scene
        perceiver._detect_speaker = speaker
        perceiver._emotion_detector = lambda: SimpleNamespace(detect=emotion)
        with background_scope(lambda: calls.append('idle')):
            result = perceiver.preprocess('synthetic fixture', audio_path='synthetic.wav')
        self.assertEqual(calls, ['idle', 'scene', 'idle', 'speaker', 'idle', 'emotion'])
        self.assertEqual(result.emotion, 'calm')
        calls.clear()
        perceiver.preprocess('synthetic fixture', audio_path='synthetic.wav')
        self.assertEqual(calls, ['scene', 'speaker', 'emotion'])

    def test_background_context_reaches_fact_worker_without_affecting_direct_callers(self):
        calls = []
        with background_scope(lambda: calls.append('worker idle')):
            worker = contextualize(background_checkpoint)
        background_checkpoint()
        self.assertEqual(calls, [])
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(worker).result(timeout=1)
        self.assertEqual(calls, ['worker idle'])
        background_checkpoint()
        self.assertEqual(calls, ['worker idle'])

    def test_new_foreground_activity_pauses_between_perception_stages(self):
        idle, scene_done, emotion_started = (threading.Event() for _ in range(3))
        idle.set()
        perceiver = object.__new__(AudioPerceiver)
        perceiver._enable_scene = perceiver._enable_emotion = True
        perceiver._enable_voiceprint = perceiver._enable_music = perceiver._enable_abnormal_sound = False
        def scene(_):
            idle.clear()
            scene_done.set()
            return '', '', None, [], None, [], {}
        def emotion(_):
            emotion_started.set()
            return 'calm'
        perceiver._detect_scene = scene
        perceiver._emotion_detector = lambda: SimpleNamespace(detect=emotion)
        with background_scope(idle.wait):
            work = contextualize(lambda: perceiver.preprocess('synthetic fixture', audio_path='synthetic.wav'))
        with ThreadPoolExecutor(max_workers=1) as pool:
            job = pool.submit(work)
            try:
                self.assertTrue(scene_done.wait(1))
                self.assertFalse(emotion_started.wait(.02))
            finally:
                idle.set()
            self.assertEqual(job.result(timeout=1).emotion, 'calm')
        self.assertTrue(emotion_started.is_set())


if __name__ == '__main__':
    unittest.main()
