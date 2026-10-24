"""Low-frequency OS paging rates and engine summaries, off the asyncio loop."""
from __future__ import annotations

import sys
import threading
import time

from ..scheduling.admission import resources
from .metrics import model_metrics


def paging_rates(previous, current, elapsed):
    """Return MiB/s between samples; first samples and resets have no rate."""
    if previous is None or elapsed <= 0:
        return {}
    return {name + "_mib_s": max(0, value-previous.get(name, value)) / elapsed / 1048576
            for name, value in current.items() if not name.endswith("_mib")}


def _counters():
    if sys.platform == "darwin":
        from .darwin import reader
        return reader().sample()
    if sys.platform.startswith("linux"):
        import os
        with open("/proc/vmstat", encoding="ascii") as source:
            values = dict(line.split() for line in source)
        page = os.sysconf("SC_PAGE_SIZE")
        return {"swap_in": int(values["pswpin"]) * page,
                "swap_out": int(values["pswpout"]) * page,
                "page_in": int(values["pgpgin"]) * 1024,
                "page_out": int(values["pgpgout"]) * 1024}
    raise NotImplementedError("OS paging counters are unavailable")


class EngineObserver:
    """Own one optional daemon sampler; importing the engine starts no threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._latest = {}

    def snapshot(self):
        with self._lock:
            return dict(self._latest)

    def start(self):
        """Start OS sampling and five-second metadata summaries once per process."""
        with self._lock:
            if self._thread is not None:
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="engine-observer", daemon=True)
            self._thread.start()

    def stop(self):
        """Request nonblocking shutdown; no model or background work is cancelled."""
        self._stop.set()

    def _run(self):
        previous = None
        sampled = reported = time.monotonic()
        cpu_at, cpu_time = sampled, time.process_time()
        warned = False
        try:
            while not self._stop.is_set():
                gauges = {}
                try:
                    counters = _counters()
                    now = time.monotonic()
                    rates = paging_rates(previous, counters, now-sampled)
                    previous, sampled = counters, now
                    gauges = {k: v for k, v in counters.items() if k.endswith("_mib")}
                    resources.paging(rates.get("swap_in_mib_s", 0) + rates.get("swap_out_mib_s", 0)
                                     + rates.get("decompress_mib_s", 0))
                except Exception as exc:
                    rates = {}
                    now = time.monotonic()
                    if not warned:
                        print(f"[engine-system] sampling_unavailable={type(exc).__name__}", flush=True)
                        warned = True
                current_cpu = time.process_time()
                if now > cpu_at:
                    rates["process_cpu_cores"] = max(0, current_cpu-cpu_time) / (now-cpu_at)
                cpu_at, cpu_time = now, current_cpu
                with self._lock:
                    self._latest = dict(sampled_at=now, **rates, **gauges)
                if now-reported >= 5:
                    rows = model_metrics(reset=True)
                    if rows:
                        from ..backends.mlx.scheduler import gpu_loop
                        from ..backends.torch import torch_snapshot
                        print("[engine-system] " + " ".join(f"{k}={v:.2f}" for k, v in rates.items())
                              + " " + " ".join(f"{k}={v:.2f}" for k, v in gauges.items())
                              + f" admission={resources.snapshot()} mlx={gpu_loop().snapshot()}"
                              + f" torch={torch_snapshot()}", flush=True)
                    for row in rows:
                        count = row["calls"]
                        print(f'[engine-model] model={row["model"]} backend={row["backend"]}'
                              f' service={row["service"]} calls={count}'
                              f' device={row.get("device") or "unspecified"}'
                              f' run_mean_ms={row["run_ms"]/count:.2f}'
                              f' run_max_ms={row["max_run_ms"]:.2f}'
                              f' lock_ms={row["lock_ms"]:.2f}'
                              f' admission_ms={row["admission_ms"]:.2f}'
                              f' queue_ms={row["queue_ms"]:.2f}'
                              f' forced={row["forced"]} errors={row["errors"]}', flush=True)
                    reported = now
                self._stop.wait(1)
        finally:
            with self._lock:
                self._thread = None


observer = EngineObserver()
