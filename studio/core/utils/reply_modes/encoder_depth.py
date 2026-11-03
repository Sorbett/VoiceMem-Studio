"""Adapt the published VoiceMem-Jev runtime to Studio's routing lifecycle."""
from __future__ import annotations

from collections import OrderedDict
import importlib
import json
import math
from pathlib import Path
import sys
import threading

from .component import FAST, SLOW, ThinkingDecision
from .decision import LocalDecisionRouter

_RUNTIME_LOCK = threading.Lock()


def _runtime_api(model):
    """Import the runtime acquired with the weights, or an explicitly installed SDK."""
    source = Path(model).parent / 'python'
    package = source / 'voicemem_jev'
    with _RUNTIME_LOCK:
        if package.is_dir():
            loaded = sys.modules.get('voicemem_jev')
            if loaded is not None and Path(loaded.__file__).resolve().parent != package.resolve():
                raise RuntimeError('A different VoiceMem-Jev runtime is loaded; restart Studio')
            sys.path.insert(0, str(source))
            try:
                return importlib.import_module('voicemem_jev')
            finally:
                sys.path.remove(str(source))
        try:
            return importlib.import_module('voicemem_jev')
        except ModuleNotFoundError as error:
            if error.name != 'voicemem_jev':
                raise
            raise RuntimeError('VoiceMem-Jev runtime is missing; download its python/ directory '
                               'with the model, or install voicemem-jev for a custom bundle') from error


class EncoderDepthRouter(LocalDecisionRouter):
    """Keep host cancellation, caching and device admission around SDK inference."""

    def __init__(self, model, *, device='cpu', backend='eager', allow_prototype=False):
        if backend not in {'eager', 'onnx'} or (backend == 'onnx' and device != 'cpu'):
            raise ValueError('Depth encoder ONNX requires device=cpu')
        self.path = Path(model).expanduser().resolve()
        self.device, self.backend = device, backend
        api = _runtime_api(self.path)
        self._runtime = (api.OnnxRouter(self.path) if backend == 'onnx' else
                         api.Router(self.path, backend='eager', device=device))
        self.metadata = self._runtime.meta
        if self.metadata.get('prototype', True) and not allow_prototype:
            raise ValueError('Prototype checkpoint requires allow_prototype=true')
        self.accepts_audio_state = self.metadata.get('report', {}).get('trained_audio_state', False)
        if type(self.accepts_audio_state) is not bool:
            raise ValueError('Invalid audio-state training flag')
        self.temperature = self._runtime.temperature
        self._model = self._tokenizer = None
        self._model_lock = threading.Lock()
        self._lock = threading.Lock()
        self._cache, self._jobs = OrderedDict(), {}

    def _input_key(self, text, history, audio_state):
        return (json.dumps((text, history or [], audio_state if self.accepts_audio_state else None),
                           sort_keys=True, ensure_ascii=False),)

    def _load(self):
        """Retain the load hook used by published Studio adapter subclasses."""
        self._runtime._load()
        self._model, self._tokenizer = self._runtime.model, self._runtime.tokenizer

    def classify(self, text, memory_prefetch_hint=False, history=None, *, audio_state=None):
        """Delegate model semantics to the SDK; cancelled calls cannot enter the cache."""
        from voicemem.engine.scheduling.cancellation import check_cancelled
        from voicemem.engine import device_lock, inference_call
        check_cancelled()
        if not str(text).strip():
            return ThinkingDecision(FAST, 'encoder:empty')
        observed = audio_state if self.accepts_audio_state else None
        key = self._input_key(text, history, observed)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        with self._model_lock:
            check_cancelled()
            with inference_call('reply_router', 'onnx' if self.backend == 'onnx' else 'torch',
                                lock=device_lock(self.device) if self.backend != 'onnx' else None,
                                device=self.device):
                if self._model is None:
                    self._load()
                check_cancelled()
                self._runtime.model, self._runtime.tokenizer = self._model, self._tokenizer
                decision = self._runtime.predict(text, history=history, observations=observed)
        check_cancelled()
        probabilities = (decision.p_ordinary, decision.p_deep)
        if (decision.label not in {'ordinary', 'deep'}
                or any(isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1
                       for value in probabilities)
                or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6)):
            raise ValueError('Invalid VoiceMem-Jev decision')
        result = ThinkingDecision(SLOW if decision.label == 'deep' else FAST,
                                  f'encoder:p_deep={decision.p_deep:.6f}')
        with self._lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > 256:
                self._cache.popitem(last=False)
        return result


def create(*, model, base_url='', api_key='', options=None):
    """Construct the installed router adapter from a complete local model bundle."""
    options = dict(options or {})
    if base_url or api_key or set(options)-{'device', 'backend', 'allow_prototype'}:
        raise ValueError('Depth encoder accepts only local runtime options')
    if 'allow_prototype' in options and type(options['allow_prototype']) is not bool:
        raise ValueError('allow_prototype must be a boolean')
    if 'device' in options and (not isinstance(options['device'], str) or not options['device']):
        raise ValueError('device must be a nonempty device name')
    return EncoderDepthRouter(model, **options)
