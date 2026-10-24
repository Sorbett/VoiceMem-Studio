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
│   ├── capacity.py         # Optional cross-provider background native capacity
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

`scheduling/capacity.py` provides background native capacity before speech
admission and model locks. Studio enables it during startup through
`configure_background_capacity(default=True)`; the standalone SDK keeps its
disabled default. Set `VOICEMEM_ENGINE_BACKGROUND_CAPACITY=0` to explicitly
disable it or `1` to enable it; invalid values fail explicitly. CPU and MPS share
one host reservation. Known CUDA devices can run independently; unknown placement
conflicts conservatively with every domain. Competing background waiters are
FIFO. Audio and interactive calls bypass this budget. Reservations release on
failure and retain ownership through already-started native work after caller
cancellation. Capacity waiting does not consume the speech-admission timeout.

The policy changes neither provider settings nor playback pacing. Keep baseline
and candidate measurements on the same models, prompts and native environment.
`capacity_ms` reports native-capacity queueing separately from model lock,
speech admission and execution; system summaries include capacity owner/waiter
counts. Configure programmatic `background_capacity` only at a quiescent boundary.

`backends/cpu.py` also supplies `cpu_thread_budget()` for FunASR model creation.
Studio defaults to two CPU intra-op threads on MLX and four on CUDA; the SDK
retains four. `VOICEMEM_ENGINE_CPU_THREADS` overrides the positive integer.
Streaming ASR, SenseVoice and emotion2vec receive the same budget, preserving
their model and decoding settings while avoiding competing thread-pool defaults.

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

`telemetry/thermal.py` reads the optional macOS thermal category on the observer
thread without subprocesses or model imports. Five-second system summaries
continue while idle so cooling remains observable. Unsupported platforms and
failed or unknown reads remain explicit; a category is neither a temperature
nor a measured clock rate and does not alter model quality or system power.

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

`python -m evals.engine_contention --capacity-crossover --background-workers 2`
compares baseline speech-aware admission with the added capacity budget in ABBA
order. It waits for every background pipeline to finish before switching policy.
The probe does not measure browser stalls or Studio end-to-end reply latency.
