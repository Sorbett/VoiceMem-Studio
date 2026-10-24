"""Darwin VM counters through the stable Mach rev1 statistics ABI."""
from __future__ import annotations

import ctypes
from functools import lru_cache
import os


class VMStatistics64(ctypes.Structure):
    """HOST_VM_INFO64_REV1 prefix; subsequent ABI revisions append fields."""

    _fields_ = ([(name, ctypes.c_uint32) for name in (
        "free_count", "active_count", "inactive_count", "wire_count")]
        + [(name, ctypes.c_uint64) for name in (
            "zero_fill_count", "reactivations", "pageins", "pageouts", "faults",
            "cow_faults", "lookups", "hits", "purges")]
        + [(name, ctypes.c_uint32) for name in ("purgeable_count", "speculative_count")]
        + [(name, ctypes.c_uint64) for name in (
            "decompressions", "compressions", "swapins", "swapouts")]
        + [(name, ctypes.c_uint32) for name in (
            "compressor_page_count", "throttled_count", "external_page_count", "internal_page_count")]
        + [("total_uncompressed_pages_in_compressor", ctypes.c_uint64)])


class DarwinCounters:
    """Keep one host port; sampling never forks the model-bearing process."""

    def __init__(self, library=None):
        self.library = library or ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        self.library.mach_host_self.restype = ctypes.c_uint32
        self.library.mach_host_self.argtypes = []
        self.library.host_statistics64.restype = ctypes.c_int
        self.library.host_statistics64.argtypes = [ctypes.c_uint32, ctypes.c_int,
                                                 ctypes.POINTER(ctypes.c_int),
                                                 ctypes.POINTER(ctypes.c_uint32)]
        self.host = self.library.mach_host_self()
        self.page_size = os.sysconf("SC_PAGE_SIZE")

    def sample(self):
        """Return cumulative byte counters and current memory gauges; raise on failure."""
        stats = VMStatistics64()
        expected = ctypes.sizeof(stats)//ctypes.sizeof(ctypes.c_int)
        count = ctypes.c_uint32(expected)
        result = self.library.host_statistics64(
            self.host, 4, ctypes.cast(ctypes.byref(stats), ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(count))
        if result or count.value < expected:
            raise OSError(f"host_statistics64 failed: status={result}, count={count.value}")
        page = self.page_size
        counters = {name: getattr(stats, field)*page for name, field in (
            ("swap_in", "swapins"), ("swap_out", "swapouts"),
            ("page_in", "pageins"), ("page_out", "pageouts"),
            ("compress", "compressions"), ("decompress", "decompressions"))}
        counters.update({name+"_mib": getattr(stats, field)*page/1048576
                         for name, field in (("free", "free_count"), ("wired", "wire_count"),
                                             ("compressor", "compressor_page_count"))})
        return counters


@lru_cache(maxsize=1)
def reader():
    """Create the Mach reader only when the sampler runs on Darwin."""
    return DarwinCounters()
