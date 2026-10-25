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
│   │   ├── prefill.py       # Cooperative prompt-cache materialization
│   │   └── compilation.py   # Decode output and RNG validation before compilation
│   ├── cuda.py             # CUDA speech worker and native cleanup
│   ├── torch.py            # Per-device locks and graph initialization
│   ├── cpu.py              # CPU runtime thread budgets
│   └── devices.py          # Runtime-independent placement labels
├── scheduling/
│   ├── context.py          # Audio, interactive and background priority
│   ├── admission.py        # Speech inventory, stage cost and resource pressure
│   ├── capacity.py         # Optional cross-provider background native capacity
│   ├── probes.py           # Nonblocking latest-snapshot CPU inference
│   ├── reuse.py            # Bounded exact-request reuse within a captured job
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

`VerifiedDecode` in `backends/mlx/compilation.py` accepts an array-valued step
whose only captured mutable state is `mx.random.state`. On the MLX owner it
replays eager and compiled results and RNG transitions from the same seed during
explicit startup verification, resetting that seed afterward. Live generation
does not run verification or consume extra sampling draws.
Mismatch or compilation failure logs a metadata-only warning and keeps eager
execution. Later runtime failures still propagate. Changed input shapes or dtypes
and unverified profiles use eager execution without additional traces. The
supplied sample is a startup acceptance check, not proof of numerical identity
for all inputs. `snapshot()` reports actual compiled/eager invocation counts
without inputs or RNG keys.
Studio warms this wrapper for Breeze depth decoding before serving; CUDA's
existing graph path and the shared provider's cached default remain independent.
Warmup uses isolated prompt-prefilled KV and actual decode activations rather
than quantized head-weight dtype. Studio startup reports `verified_depth` profile
counts, while failed checks emit a reason without model inputs or private data.

`scheduling/context.py` carries audio, interactive and background priority through explicitly
captured workers. `scheduling/admission.py` uses output-owned audio inventory, observed native
cost and fresh resource pressure to delay background stages before model locks.
It preserves bounded persistence progress and does not preempt native inference.
Known work on another CUDA GPU can proceed independently; unknown placement is
treated conservatively. Browser inventory is accepted only after output ownership
validation and is never evidence that the user heard audio.

`background_scope(wait, cancelled=predicate)` optionally captures a thread-safe
predicate for disposable work. Native capacity, resource admission and model
lock waits raise `InferenceCancelled` before executing obsolete work and release
all acquired reservations. A model call already executing retains its ownership
until completion. Confirmed memory leaves this option unset.

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

`scheduling/probes.py` supplies the serial CPU EOT executor and `LatestProbe`.
Each input connection owns at most one queued or running snapshot; the input
owner polls results without awaiting inference and retains current audio while
busy. Native work above its 40 ms budget gets an equal recovery gap, excluding
executor queue wait. Reset cancels unstarted work and rejects started results
without freeing an active native slot. The input owner validates turn, detector,
speech coverage and silence before confirmation. All ASR frames remain intact.
This CPU path is shared by MLX and CUDA deployments and starts no thread at
import. `snapshot()` reports counts, execution, queue and recovery timings
without audio or text.

One `background_scope` shares accumulated admission wait across captured child
workers. After sustained waiting, a stage can use a smaller safe inventory
reserve that covers its own cost and the next observed synthesis step, once per
newly produced speech chunk. This does not bypass stalled or
unsustainable production, the stage-cost estimate, or fresh memory pressure.
The existing per-stage forced-progress limit remains the fallback; accumulated
waiting is not an end-to-end persistence deadline. Snapshot `aged` counts these
headroom-checked admissions separately from forced progress.

`BackgroundPipeline` bounds deferred operations through their actual completion.
It prepares jobs FIFO and supplies a thread-safe predecessor barrier for ordered
side effects, allowing preparation to overlap earlier remote waits. Started work
keeps its capacity and ordering ownership after caller cancellation; failed
preparation still settles its started native work and preceding barriers. Studio
uses two slots for memory and a separate `BackgroundLane` for acoustic UI. Native
model calls in both paths retain the same process-wide admission and capacity.

`scheduling/reuse.py` supplies an operation-local pool through `reuse_scope` and
`reuse_call`. `background_scope` captures it with the job's existing worker
context. Adapters opt in only for pure inference with immutable model state,
using exact request keys and immutable CPU `ReusableResult` payloads. The pool
holds at most 64 completed entries and two MiB of payloads, coalesces identical
concurrent misses outside device/model locks, and returns independent copies.
Failures and unsupported/oversized results are not retained. Cancellation of a
waiting consumer does not abandon started native work. Separate jobs do not
share results; `reuse_scope(enabled=False)` disables reuse in captured children.

The built-in local encoder reuses only matching NumPy batches, including input
order, options, prompt configuration and model parameter versions/device/dtype.
Input text is represented by a digest in the key. Training models, explicit
placement, token/tensor output, custom options and device-lock owners bypass it.
Mutable tokenizer/model configuration requires the opt-out. Injected providers
and database reads retain their existing paths. Reuse emits `backend=cache`
timings without resetting native cost or background recovery estimates.

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
Fresh fair/serious/critical categories space background calls by half/one/two
times the last conflicting background native duration, capped at two seconds.
Escalation is immediate; five seconds of fresh lower-state readings permit
recovery to that state's budget. Unknown, unavailable or stale readings disable
thermal pacing. Foreground inference bypasses the gap; the existing bounded
background deferral remains in force.

Periodic audio adapters can declare `cadence_s` through `inference_call`.
Successful streaming ASR chunks use their unchanged 600 ms input cadence;
final tails, failures and lock/queue waits do not contribute. Three consecutive
native calls consuming at least half their cadence, with 0.25 EWMA load also
at least half, enable background gaps equal to the last conflicting native
duration, capped at two seconds. Recovery requires load below 40%; stale
feedback expires after three seconds. A single startup spike does not activate
this budget. It works on both MLX and CUDA hosts without OS thermal support;
known distinct CUDA devices retain independent load and background histories.
It changes no recognition, synthesis or persistence settings. Thermal and
periodic-load pacing combine by maximum. Snapshots expose `pacing_level`, fresh
`realtime_pressure` count and conservative aggregate `cooling_ms`; these are
scheduling signals, not predicted device cooling times.

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
