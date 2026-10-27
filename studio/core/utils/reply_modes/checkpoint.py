"""Select the published VoiceMem-Jev format without importing model runtimes."""
from __future__ import annotations

from dataclasses import dataclass
import os
import platform

from studio.core.utils.models.component import Model

REPOSITORY = 'Sorbett/VoiceMem-Jev'
REVISION = 'f2ede1f81a39c785fe191a7f00327a122d799f26'
DIRECTORY = 'reply-router/VoiceMem-Jev'

_SHARED = {
    'encoder/config.json': '109b4b1013b6dafabbfcb2af4ddc3e7b9c4a7d475ea8128d3008fcd419249174',
    'tokenizer/special_tokens_map.json': '5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a',
    'tokenizer/tokenizer.json': '870a5478a34f0c22ec65735350f131c094aa1a1783bbe60aa78defc6b4378b25',
    'tokenizer/tokenizer_config.json': 'bc3788b020afe8d772684fc7a5f64cbb562f1b816d2585858558d3df8b95bbf7',
    'tokenizer/vocab.txt': '45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c',
}
_ROOT = {
    'LICENSE': 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
    'NOTICE.txt': 'f771be8daf9bcbadd07a0546999d50c5cce75366ab54670c4a194b1b1e9aa44b',
    'SOURCE.json': '95c292a32c807104a2d04228e90106e8f204c1f8787e8c72247a9c29adaf0a44',
    'UPSTREAM_REVISION': '3f0af6a5a166ac735427a7f1c9894c122c05111b0a9160b548288bd674c6f0c4',
    'EVALUATION.json': 'cbc0b804804e66ae70d4c60069742133273c67a36deb4236b6c28d1bb4c0b2a1',
}
_FORMATS = {
    'eager': {
        'depth_encoder.json': '4093dc0e9f23bae8e54e4201f1810dd2016082a373108e969302122cd502dc89',
        'model.safetensors': '925758be661b3322808eadf967d7e2c174c83ad7bc5949367bf906f88386efa5',
    },
    'onnx': {
        'depth_encoder.json': '509b6583fce4f35ffbbe7306bc9b17032f138205bce71339c3cdd9d4e68e0d54',
        'model.onnx': '35ef159d8a14aa4b81eb959663af01508e76cc80609860e94956abcd3a939290',
    },
}


@dataclass(frozen=True)
class RouterRuntime:
    """Bind one checkpoint format to its supported inference device."""

    backend: str
    device: str

    @property
    def directory(self):
        return f'{DIRECTORY}/{self.backend}'


def runtime(backend=None, device=None):
    """Use Torch weights for CUDA and CPU ONNX for all other router devices."""
    backend = backend or os.environ.get('STUDIO_BACKEND') or (
        'mlx' if platform.system() == 'Darwin' else 'cuda')
    device = device or os.environ.get('STUDIO_DEVICE') or ('cuda:0' if backend == 'cuda' else 'cpu')
    if device == 'cuda' or (device.startswith('cuda:') and device[5:].isdigit()):
        return RouterRuntime('eager', device)
    return RouterRuntime('onnx', 'cpu')


def model_spec(backend=None, device=None):
    """Return only the selected format and pinned runtime-file checksums."""
    selected = runtime(backend, device)
    checksums = dict(_ROOT)
    checksums.update({f'{selected.backend}/{name}': digest
                      for name, digest in {**_SHARED, **_FORMATS[selected.backend]}.items()})
    required = tuple(checksums)
    return Model('VoiceMem-Jev', DIRECTORY, REPOSITORY, required,
                 patterns=required, revision=REVISION, checksums=tuple(checksums.items()))
