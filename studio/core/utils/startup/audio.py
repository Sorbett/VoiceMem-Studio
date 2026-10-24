"""Disposable, non-silent input for local model startup checks."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def warmup_audio():
    """Yield mono 16 kHz PCM and a WAV path, removing the file on exit."""
    import numpy as np
    import soundfile as sf

    sample_rate = 16000
    time = np.arange(3 * sample_rate, dtype=np.float32) / sample_rate
    pcm = (.05 * np.sin(2 * np.pi * 220 * time)
           + .025 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    # The speaker worker skips silence and inputs shorter than one second.
    with TemporaryDirectory(prefix='studio-warmup-') as temporary:
        path = Path(temporary) / 'probe.wav'
        sf.write(path, pcm, sample_rate)
        yield pcm, path
