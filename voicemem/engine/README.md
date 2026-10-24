# Harness inference infrastructure

`voicemem.engine` is shared runtime infrastructure used by Studio's Harness and
the VoiceMem SDK. Importing it starts no threads and imports no optional model
runtimes. Dialogue policy, prompt protocols, weights, model factories and memory
semantics remain in their existing owners.

## Layout and ownership

```text
engine/
├── __init__.py              # Public facade; no optional runtime imports
├── backends/
│   ├── mlx/
│   │   ├── scheduler.py     # One MLX owner thread and bounded delivery
│   │   └── prefill.py       # Cooperative prompt-cache materialization
│   ├── cuda.py             # CUDA speech worker and native cleanup
│   ├── torch.py            # Per-device locks and graph initialization
│   ├── cpu.py              # CPU runtime thread budgets
│   └── devices.py          # Runtime-independent placement labels
├── scheduling/
│   ├── context.py          # Audio, interactive and background priority
│   ├── admission.py        # Speech inventory, stage cost and resource pressure
│   └── background.py       # Captured admission and native task lifetime
└── telemetry/
    ├── metrics.py          # Native execution, lock, admission and queue time
    ├── system.py           # Lifecycle-owned OS/device observer
    └── darwin.py           # Native macOS memory counters
```

Adapters use `from voicemem.engine import ...`. Implementation modules are
internal boundaries; the public facade and historical `voicemem.utils` entry
points expose the same classes, contexts and process singletons. Package imports
start no worker or observer threads. Adding a backend must retain cancellation,
bounded delivery and owner-thread cleanup without importing Studio policy.

## Execution contracts

| Execution path | Owner and boundary |
| --- | --- |
| Apple Silicon MLX | `backends/mlx/scheduler.py`: one process execution thread, stepped fairness and bounded output |
| NVIDIA CUDA speech | `backends/cuda.py`: one model owner thread, device admission between host-PCM steps and bounded async delivery |
| Torch CUDA | `backends/torch.py`: one priority lock per resolved device, shared by speech, ASR, routing and GPU perception |
| Torch MPS / legacy | `backends/torch.py`: the existing reentrant `TORCH_LOCK` singleton |
| CPU ONNX | `backends/cpu.py`: bounded session thread budgets; existing model-private serialization remains in adapters |

`backends/mlx/prefill.py` materializes bounded token blocks on that same MLX thread. Local
reply generation reserves the last prompt token for the existing sampler;
prefix warmup publishes only complete caches. `GpuLoop.run_steps` lets native
warmup callers yield between blocks and drains owner-thread cancellation.
Prompt changes and cache resets invalidate an unfinished warmup before publication.
Studio depth routing uses this helper for both static and request prefixes;
synchronous warmup also yields through `run_steps`. Providers own tokenization,
cache validity, decoding and prompt semantics.

`scheduling/context.py` carries audio, interactive and background priority through explicitly
captured workers. `scheduling/admission.py` uses output-owned audio inventory, observed native
cost and fresh resource pressure to delay background stages before model locks.
It preserves bounded persistence progress and does not preempt native inference.
Known work on another CUDA GPU can proceed independently; unknown placement is
treated conservatively. Browser inventory is accepted only after output ownership
validation and is never evidence that the user heard audio.

One `background_scope` shares accumulated admission wait across captured child
workers. After sustained waiting, a stage can use a smaller safe inventory
reserve that covers its own cost and the next observed synthesis step, once per
newly produced speech chunk. This does not bypass stalled or
unsustainable production, the stage-cost estimate, or fresh memory pressure.
The existing per-stage forced-progress limit remains the fallback; accumulated
waiting is not an end-to-end persistence deadline. Snapshot `aged` counts these
headroom-checked admissions separately from forced progress.

CUDA adapters must resolve placement before crossing threads and materialize
their host results before releasing native admission. `CudaWorker` performs
factory creation, iterator steps and cleanup on one owner, releases device
admission during downstream backpressure and retains speech ownership until
native cleanup finishes after cancellation. Callers must close async iterators
and drain the worker at provider shutdown.

CUDA graph initialization runs under `cuda_initialization()` before acquiring
device admission. It excludes registered CUDA execution across all devices.
An attempted read-to-write upgrade raises `RuntimeError`; holding a device lock
while waiting on another worker remains unsupported.

## Observation and verification

`telemetry/metrics.py` separates adapter time, device-lock wait, background admission and
initial worker queue time by model/backend/service/device. Timings include the
adapter's host materialization boundary, not isolated kernel time. `telemetry/system.py`
samples OS paging outside the event loop. Device owners cache CUDA allocator
allocated/reserved/peak and global free/total bytes, or MPS tensor/process-wide
Metal driver allocation. Reading snapshots never initializes CUDA. Sampling
failure is visible and must not leak an admission permit.

Deterministic checks run without NVIDIA hardware:

```bash
python -m unittest tests.voicemem.test_engine_imports tests.voicemem.test_cuda_engine tests.studio.test_cuda_backend tests.studio.test_mlx_scheduling
```

Native CUDA delivery and ASR contention are measured separately with
`python -m evals.breeze_cuda_latency --with-asr --segmented` on the configured
NVIDIA environment. These results require hardware validation and do not imply
browser-perceived latency. See [evaluation entry points](../../evals/README.md).

`python -m evals.mlx_prefill` checks recurrent/attention cache equivalence and
native scheduling on a seeded small architecture without downloading weights.
Use `--architecture qwen3 --step-size 128` to probe the router architecture.
Its competing arithmetic probe is not an audio or production-model benchmark.
