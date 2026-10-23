"""Committed memory mutations publish advisory revisions without model work."""
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from voicemem.leftbrain.mem0_backend_store import Mem0BackendStore
from voicemem.rightbrain.traits_store import Evidence, TraitStore
from voicemem.utils.common.memory_changes import MemoryChanges, for_space


class MemoryChangeTests(unittest.TestCase):
    def test_shared_roots_isolated_spaces_and_unsubscribe(self):
        with tempfile.TemporaryDirectory() as temp:
            feed = for_space(temp)
            self.assertIs(feed, for_space(Path(temp) / "."))
            self.assertIsNot(feed, for_space(Path(temp) / "other"))
            events = []
            remove = feed.subscribe(events.append)
            feed.publish(["first"])
            remove()
            remove()
            feed.publish(["second", "first"])
            self.assertEqual(events, [1])
            self.assertEqual(feed.state(), (2, ["first", "second"]))

    def test_listener_failure_does_not_lose_other_notifications_or_committed_ids(self):
        feed = MemoryChanges()
        feed.subscribe(Mock(side_effect=RuntimeError("Synthetic listener failure")))
        events = []
        feed.subscribe(events.append)
        with self.assertLogs("voicemem.utils.common.memory_changes", level="ERROR"):
            feed.publish(str(i) for i in range(150))
        self.assertEqual(events, [1])
        self.assertEqual(len(feed.state()[1]), 128)
        self.assertEqual(feed.state()[1][0], "149")

    def test_partial_native_batch_notifies_first_commit_before_later_failure(self):
        store = Mem0BackendStore.__new__(Mem0BackendStore)
        store._changes = MemoryChanges()
        store._mem0 = SimpleNamespace(add=Mock(side_effect=[
            {"results": [{"id": "committed"}]}, RuntimeError("Synthetic second write failure"),
        ]))
        events = []
        store._changes.subscribe(events.append)
        with self.assertRaises(RuntimeError):
            store.add_records_with_ids("fixture-user", [
                ("", "First fact", "user", {}), ("", "Second fact", "user", {}),
            ])
        self.assertEqual(events, [1])
        self.assertEqual(store._changes.state()[1], ["committed"])

    def test_native_updates_deletes_and_archive_notify_only_on_success(self):
        store = Mem0BackendStore.__new__(Mem0BackendStore)
        store._changes = MemoryChanges()
        store._mem0 = SimpleNamespace(update=Mock(), delete=Mock())
        for action in [lambda: store.update_memory("id", "New fact"),
                       lambda: store.delete_memory("id"),
                       lambda: store.archive_memory("id"),
                       lambda: store.unarchive_memory("id")]:
            self.assertTrue(action())
        self.assertEqual(store._changes.state()[0], 4)
        store._mem0.update.side_effect = RuntimeError("Synthetic write failure")
        store._mem0.delete.side_effect = RuntimeError("Synthetic write failure")
        self.assertFalse(store.update_memory("id", "Failed fact"))
        self.assertFalse(store.delete_memory("id"))
        self.assertEqual(store._changes.state()[0], 4)

    def test_trait_and_merged_evidence_are_readable_at_notification(self):
        with tempfile.TemporaryDirectory() as temp:
            store = TraitStore(Path(temp) / "fixture.sqlite", lambda text: [1., 0.])
            snapshots = []
            store._changes.subscribe(lambda revision: snapshots.append(store.counts("fixture-user")))
            first = store.add("fixture-user", "表达风格", "Prefers short replies", Evidence(quote="Brief please"))
            second = store.add("fixture-user", "表达风格", "Prefers concise answers", Evidence(quote="Less detail"))
            self.assertEqual(first, second)
            self.assertEqual(snapshots, [(1, 1), (1, 2)])
            self.assertEqual(len(store.all("fixture-user")[0].evidence), 2)
            self.assertEqual(store._changes.state(), (2, [first]))

    def test_recent_trait_wins_display_cap_even_when_commit_timestamps_tie(self):
        with tempfile.TemporaryDirectory() as temp:
            store = TraitStore(Path(temp) / "fixture.sqlite", Mock(side_effect=ValueError("No embedding")))
            ids = [store.add("fixture-user", "表达风格", f"Synthetic preference {index}", Evidence(quote="Fixture"))
                   for index in range(12)]
            self.assertEqual(store.all("fixture-user", per_slot=2)[0].id, ids[-1])

    def test_background_completion_publishes_after_success_empty_and_partial_failure(self):
        from voicemem.orchestrator import Orchestrator
        with tempfile.TemporaryDirectory() as temp:
            for index, result in enumerate([
                {"persistent_memory_created": True, "memory_ids": ["fact"]},
                {"persistent_memory_created": False}, RuntimeError("Synthetic partial failure"),
            ]):
                memory = Orchestrator.__new__(Orchestrator)
                memory._memory_root = Path(temp) / str(index)
                memory.memory_language = "en"
                memory.follow_input_language = True
                memory.remember_reply = Mock()
                memory.last_agent_reply = lambda **kw: ""
                memory._clap_memory_enabled = lambda: False
                memory.preprocess = lambda *args: SimpleNamespace(
                    speaker="user", emotion="", environment="", environment_hint="",
                    scene_tag="", scene_raw_labels=[], person_id="", tune_result=None,
                    abnormal_hits=[], detection=None)
                started, release, finished = (threading.Event() for _ in range(3))
                feed = for_space(memory._memory_root)
                def persist(context):
                    started.set()
                    if not release.wait(2):
                        raise TimeoutError("Synthetic persistence was not released")
                    if isinstance(result, Exception):
                        raise result
                    return result
                memory._finish_ingest = persist
                try:
                    memory.Ingest("Synthetic input", agent_reply="", async_facts=True,
                                  on_complete=lambda outcome: finished.set())
                    self.assertTrue(started.wait(1))
                    self.assertEqual(feed.state()[0], 0)
                    release.set()
                    self.assertTrue(finished.wait(1))
                    self.assertEqual(feed.state()[0], 1)
                finally:
                    release.set()


if __name__ == "__main__":
    unittest.main()
