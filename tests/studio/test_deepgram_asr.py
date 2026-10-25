"""Deterministic Listen v1 transcripts and socket ownership without network audio."""
import asyncio
from collections import deque
import json
from urllib.parse import parse_qs, urlsplit
import unittest
from unittest.mock import patch

import numpy as np

from studio.core.utils.asr.deepgram import DeepgramASR


class Socket:
    def __init__(self):
        self.messages, self.sent, self.closed = deque(), [], False

    def send(self, value):
        self.sent.append(value)

    def recv(self, timeout=0):
        if not self.messages:
            raise TimeoutError
        return json.dumps(self.messages.popleft())

    def close(self):
        self.closed = True

    def result(self, text, *, final=False, start=0, duration=1, flushed=False):
        self.messages.append({'type':'Results','start':start,'duration':duration,
                              'is_final':final,'from_finalize':flushed,
                              'channel':{'alternatives':[{'transcript':text}]}})


class DeepgramTests(unittest.TestCase):
    def adapter(self):
        adapter = DeepgramASR(model='nova-3',base_url='https://api.example/v1',api_key='key-fixture',language='zh')
        socket = Socket()
        adapter._socket = socket
        return adapter, socket

    def test_interim_replacement_and_final_segments_form_one_cumulative_transcript(self):
        adapter, socket = self.adapter()
        socket.result('你好')
        self.assertEqual(adapter.feed(np.zeros(160)), '你好')
        socket.result('你好世界',final=True)
        socket.result('今天',start=1)
        self.assertEqual(adapter.feed(np.zeros(160)), '你好世界今天')
        socket.result('今天晴天',final=True,start=1)
        socket.result('今天晴天',final=True,start=1)
        self.assertEqual(adapter.feed(np.zeros(160)), '你好世界今天晴天')
        self.assertEqual(len(socket.sent[0]), 320)

    def test_finalize_preserves_tail_and_allows_more_audio_after_a_pause(self):
        adapter, socket = self.adapter()
        adapter.feed(np.zeros(160))
        socket.result('Hello',final=True,flushed=True)
        self.assertEqual(adapter.flush(),'Hello')
        self.assertEqual(json.loads(socket.sent[-1]),{'type':'Finalize'})
        sent = len(socket.sent)
        self.assertEqual(adapter.flush(),'Hello')
        self.assertEqual(len(socket.sent),sent)
        socket.result('again',final=True,start=1)
        self.assertEqual(adapter.feed(np.zeros(160)),'Hello again')

    def test_reset_and_new_stream_do_not_share_connections_or_late_results(self):
        adapter, socket = self.adapter()
        socket.result('old',final=True)
        child = adapter.new_stream()
        self.assertIsNone(child._socket)
        self.assertEqual(child.text,'')
        adapter.reset()
        self.assertTrue(socket.closed)
        self.assertIsNone(adapter._socket)
        self.assertEqual(adapter.text,'')
        self.assertEqual(len(socket.messages),1)

    def test_handshake_keeps_key_in_header_and_preserves_harness_turn_ownership(self):
        adapter = DeepgramASR(model='nova-3',base_url='https://api.example/v1',api_key='key-fixture')
        with patch('websockets.sync.client.connect',return_value=Socket()) as connect:
            adapter.feed(np.zeros(160))
        uri = connect.call_args.args[0]
        self.assertNotIn('key-fixture',uri)
        self.assertEqual(urlsplit(uri).path,'/v1/listen')
        params = parse_qs(urlsplit(uri).query)
        self.assertEqual(params['sample_rate'],['16000'])
        self.assertEqual(params['endpointing'],['false'])
        self.assertEqual(params['interim_results'],['true'])
        self.assertEqual(connect.call_args.kwargs['additional_headers'],{'Authorization':'Token key-fixture'})
        adapter.close()

    def test_service_error_is_explicit_without_returning_vendor_error_payload(self):
        adapter, socket = self.adapter()
        socket.messages.append({'type':'Error','description':'private-service-data'})
        with self.assertRaisesRegex(RuntimeError, 'Deepgram') as error:
            adapter.feed(np.zeros(160))
        self.assertNotIn('private-service-data',str(error.exception))


class DisconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_asr_worker_closes_transport_on_disconnect(self):
        from voicemem.stream import _AsrWorker
        adapter = DeepgramASR(model='fixture',base_url='https://api.example/v1',api_key='key-fixture')
        socket = Socket()
        adapter._socket = socket
        worker = _AsrWorker(adapter)
        worker.close()
        await asyncio.to_thread(worker.t.join,1)
        self.assertFalse(worker.t.is_alive())
        self.assertTrue(socket.closed)


if __name__ == '__main__':
    unittest.main()
