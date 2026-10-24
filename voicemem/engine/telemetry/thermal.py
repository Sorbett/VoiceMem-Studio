"""Optional OS thermal-state sampling without model imports or subprocesses."""
from __future__ import annotations

from functools import lru_cache
import sys


class DarwinThermalState:
    """Read NSProcessInfo's thermal category on the observer thread."""

    STATES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}

    def __init__(self):
        import ctypes
        self._foundation = ctypes.CDLL(
            "/System/Library/Frameworks/Foundation.framework/Foundation")
        self._runtime = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        get_class = self._runtime.objc_getClass
        get_class.argtypes, get_class.restype = [ctypes.c_char_p], ctypes.c_void_p
        selector = self._runtime.sel_registerName
        selector.argtypes, selector.restype = [ctypes.c_char_p], ctypes.c_void_p
        send_object = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", self._runtime))
        self._send_state = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", self._runtime))
        process_class = get_class(b"NSProcessInfo")
        if not process_class:
            raise RuntimeError("NSProcessInfo is unavailable")
        self._process = send_object(process_class, selector(b"processInfo"))
        if not self._process:
            raise RuntimeError("NSProcessInfo initialization failed")
        self._selector = selector(b"thermalState")

    def sample(self):
        """Return the OS category, or unknown for an unrecognized future value."""
        return self.STATES.get(self._send_state(self._process, self._selector), "unknown")


@lru_cache(maxsize=1)
def _reader():
    try:
        return DarwinThermalState()
    except (OSError, AttributeError, RuntimeError):
        return None


def thermal_state():
    """Return an optional macOS thermal category; unavailable is never nominal."""
    if sys.platform != "darwin":
        return "unavailable"
    reader = _reader()
    if reader is None:
        return "unavailable"
    try:
        return reader.sample()
    except (OSError, AttributeError, RuntimeError):
        return "unavailable"
