"""Deepgram Listen v1 on the existing connection-owned ASR worker."""
from __future__ import annotations

import json
import time
from urllib.parse import urlencode, urlsplit, urlunsplit

import numpy as np


class DeepgramASR:
    """Expose cumulative transcripts while leaving turn confirmation to Harness.

    ``feed`` and ``flush`` run on VoiceStream's ASR worker, never its event loop.
    Each ``new_stream`` owns a socket; reset and close discard its pending output.
    """

    def __init__(self, *, model, base_url, api_key, language='zh'):
        self.config = dict(model=model, base_url=base_url, api_key=api_key, language=language)
        self._socket = None
        self._parts = []
        self._interim = ''
        self._final_end = -1.0
        self._flushed = True

    def new_stream(self):
        """Allocate a decoder with no socket or transcript shared with its parent."""
        return type(self)(**self.config)

    def _connect(self):
        if self._socket is None:
            from websockets.sync.client import connect
            url = urlsplit(self.config['base_url'])
            query = urlencode(dict(model=self.config['model'], language=self.config['language'],
                                   encoding='linear16', sample_rate=16000, channels=1,
                                   interim_results='true', punctuate='true', endpointing='false'))
            uri = urlunsplit(('wss' if url.scheme == 'https' else 'ws', url.netloc,
                              url.path.rstrip('/') + '/listen', query, ''))
            headers = {'Authorization': 'Token ' + self.config['api_key']} if self.config['api_key'] else {}
            try:
                self._socket = connect(uri, additional_headers=headers, open_timeout=5,
                                       close_timeout=.25, max_size=1024 * 1024)
            except Exception:
                raise RuntimeError('Deepgram 连接失败，请检查服务地址、网络和 API Key。') from None
        return self._socket

    @property
    def text(self):
        parts = [*self._parts, self._interim]
        text = ''
        for part in filter(None, parts):
            space = bool(text and text[-1].isascii() and part[0].isascii()
                         and not text.endswith(' ') and part[0] not in ',.!?;:')
            text += (' ' if space else '') + part
        return text

    def _result(self, message):
        if message.get('type') == 'Error':
            raise RuntimeError('Deepgram 返回识别错误，请检查模型和语言配置。')
        if message.get('type') != 'Results':
            return False
        alternatives = message.get('channel', {}).get('alternatives', [])
        transcript = alternatives[0].get('transcript', '').strip() if alternatives else ''
        if message.get('is_final'):
            end = float(message.get('start', 0)) + float(message.get('duration', 0))
            if end > self._final_end:
                if transcript:
                    self._parts.append(transcript)
                self._final_end = end
                self._interim = ''
        else:
            self._interim = transcript
        return bool(message.get('from_finalize'))

    def _receive(self, timeout=0):
        deadline = time.monotonic() + timeout
        for _ in range(128):
            try:
                raw = self._socket.recv(timeout=max(0, deadline - time.monotonic()))
            except TimeoutError:
                return
            if self._result(json.loads(raw)):
                return

    def feed(self, samples):
        """Send mono float audio and return the latest available cumulative text."""
        pcm = (np.clip(np.asarray(samples), -1, 1) * 32767).astype('<i2').tobytes()
        if pcm:
            self._connect().send(pcm)
            self._flushed = False
            self._receive()
        return self.text

    def flush(self):
        """Request remaining tokens once, with a bounded wait for finalization."""
        if self._socket is not None and not self._flushed:
            self._socket.send(json.dumps({'type': 'Finalize'}))
            self._flushed = True
            self._receive(timeout=1)
        return self.text

    def close(self):
        """Release this decoder's connection; no sibling decoder is affected."""
        socket, self._socket = self._socket, None
        if socket is not None:
            socket.close()

    def reset(self):
        """Discard the previous utterance and reconnect lazily on the next frame."""
        try:
            self.close()
        finally:
            self._parts, self._interim, self._final_end, self._flushed = [], '', -1.0, True
