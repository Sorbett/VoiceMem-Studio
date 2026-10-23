"""Passive browser observers follow durable Space changes without a session."""
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from studio.web.transport import build_app
from voicemem.utils.common.memory_changes import MemoryChanges


class MemoryObserverTests(unittest.TestCase):
    def client(self, feed, **kwargs):
        session = Mock(side_effect=AssertionError("Memory observers must not start a conversation"))
        app = build_app("llm_tts", session, None,
                        memory_changes=lambda: ("fixture-space", feed), **kwargs)
        client = TestClient(app)
        self.addCleanup(client.close)
        return client

    def test_two_passive_styles_receive_worker_commits_and_disconnect_unsubscribes(self):
        feed = MemoryChanges()
        client = self.client(feed)
        with client.websocket_connect("/ws/memories") as technical, \
                client.websocket_connect("/ws/memories") as digital:
            expected = {"type": "memory_changed", "space": "fixture-space",
                        "epoch": feed.epoch, "revision": 0}
            self.assertEqual(technical.receive_json(), expected)
            self.assertEqual(digital.receive_json(), expected)
            with ThreadPoolExecutor(max_workers=1) as worker:
                worker.submit(feed.publish, ["fixture-fact"]).result(timeout=1)
            expected["revision"] = 1
            self.assertEqual(technical.receive_json(), expected)
            self.assertEqual(digital.receive_json(), expected)
        self.assertFalse(feed._listeners)
        feed.publish(["after-disconnect"])

    def test_reconnect_reads_revision_after_missed_commits(self):
        feed = MemoryChanges()
        client = self.client(feed)
        with client.websocket_connect("/ws/memories") as socket:
            self.assertEqual(socket.receive_json()["revision"], 0)
        feed.publish(["missed-while-offline"])
        feed.publish(["second-missed-write"])
        with client.websocket_connect("/ws/memories") as socket:
            self.assertEqual(socket.receive_json()["revision"], 2)
        self.assertFalse(feed._listeners)

    def test_graph_and_full_list_use_separate_off_loop_callbacks(self):
        import threading
        calls = []
        def read(kind):
            calls.append((kind, threading.current_thread().name))
            return {"kind": kind}
        client = self.client(MemoryChanges(), snapshot=lambda: read("graph"),
                             full_snapshot=lambda: read("all"))
        self.assertEqual(client.get("/api/memories").json(), {"kind": "graph"})
        self.assertEqual(client.get("/api/memories?all=true").json(), {"kind": "all"})
        self.assertEqual([kind for kind, _ in calls], ["graph", "all"])
        self.assertTrue(all("portal" not in thread for _, thread in calls))


if __name__ == "__main__":
    unittest.main()
