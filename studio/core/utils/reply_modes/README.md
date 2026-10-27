# Local depth router adapters

The recommended router is VoiceMem-Jev, a task-specific encoder implementing
the existing `ThinkingDecision` boundary. It decides ordinary
versus deep reasoning; VoiceMem continues to own memory eligibility. The
router cannot end a voice turn or release private speculative output.

## VoiceMem-Jev (v6)

The public model release is named **VoiceMem-Jev**. Its current checkpoint is v6,
published at [Sorbett/VoiceMem-Jev](https://huggingface.co/Sorbett/VoiceMem-Jev).
It is a 102,274,064-parameter MacBERT encoder with a calibrated binary head.
The model repository contains two formats of the same checkpoint:

| Directory | Weight file | Recommended runtime selection |
| --- | --- | --- |
| `eager/` | `model.safetensors` | CUDA: PyTorch on `--device` / `STUDIO_DEVICE` |
| `onnx/` | `model.onnx` | Apple Silicon and other CPU router devices: CPU ONNX |

These are serialization/runtime formats, not hardware-specific model versions.
The current ONNX adapter explicitly selects CPUExecutionProvider; ONNX CUDA is
not enabled by choosing a CUDA-capable machine. PyTorch device availability
depends on the installed runtime and hardware.

Startup acquisition selects one format, pins Hub revision
`f2ede1f81a39c785fe191a7f00327a122d799f26`, and validates its runtime files
against source-controlled SHA-256 values. Its cache root is
`studio/models/reply-router/VoiceMem-Jev` (or the corresponding path under
`STUDIO_MODELS_DIR`). Complete caches need no network. Missing files resume
downloading; checksum mismatches trigger a fresh download and remain fatal if
validation still fails. `--check` only inspects files. Acquisition completes
before router warmup and live input; routing itself never downloads weights.
Recommended startup does not download or load Qwen3-0.6B. Legacy Qwen classes
remain available to explicit evaluation callers. `STUDIO_ROUTER_BACKEND` no
longer selects the recommended implementation.

Keep `depth_encoder.json`, `encoder/`, and `tokenizer/` beside each weight file.
The original checkpoint includes the custom decision head. Loading only a base
MacBERT encoder or a generic sequence-classification pipeline is insufficient.
Keep `LICENSE`, `NOTICE.txt`, `SOURCE.json`, and the scoped `EVALUATION.json` at
the model repository root. A release-specific `SHA256SUMS` must describe the
files actually distributed.
The ONNX weight SHA-256 is
`35ef159d8a14aa4b81eb959663af01508e76cc80609860e94956abcd3a939290`.
Weights are runtime data and do not belong in this working tree's source diff.

No adapter configuration is needed for the recommended selection. To use an
explicit local checkpoint instead, choose **Settings → Components → Router →
Adapter**, using the complete directory for the chosen format:

- Factory: `studio.core.utils.reply_modes.encoder_depth:create`
- Model: `studio/models/reply-router/VoiceMem-Jev/onnx` from the repository root,
  or an absolute path to the complete bundle.
- Options: `{"backend":"onnx","device":"cpu","allow_prototype":true}`
- API Key and Base URL: empty.

For the original checkpoint, change Model to the complete `eager` directory and
Options to `{"backend":"eager","device":"cuda:0","allow_prototype":true}`
for CUDA, or use `"device":"cpu"` for CPU. Changing the file format alone does
not select a device. The custom adapter accepts local directories only; custom
Hub repositories must be downloaded before selecting their format directory.

The same startup selection is:

```dotenv
STUDIO_COMPONENTS='{"router":{"provider":"adapter","factory":"studio.core.utils.reply_modes.encoder_depth:create","model":"studio/models/reply-router/VoiceMem-Jev/onnx","options":{"backend":"onnx","device":"cpu","allow_prototype":true}}}'
```

Selection uses the existing apply/restart flow. Restore the recommended router
to re-enable automatic device binding and acquisition. The release metadata
retains its prototype marker; recommended startup explicitly accepts this pinned
release, while custom adapter callers must still opt in with `allow_prototype`.
This does not establish natural-dialogue accuracy. The fixed classifier is trained
through its checkpoint, so editing a generative router prompt does not change
its learned decision policy. Main reply, TTS and custom filler prompts remain
request-scoped as usual.

CPU ONNX uses two non-spinning threads and a model-private lock, without acquiring
the Torch/MPS device lock. Async work uses the engine input executor and
cancellation scope; cancelled work cannot publish a late turn result. EOT,
silence and voiced-span observations are passed only to adapters declaring
support. Missing measurements remain absent. A continued input uses the joined
voiced span and current silence state.

## Waiting speech

Default long waiting speech uses four neutral phrases per language. Startup
prepares all eight PCM16 clips using the selected speech provider and default
speech rate. Stable voice identities persist clips under ignored
`cache/work-filler`; provider identity, language, exact text and TTS instructions
are part of the cache key. Providers without a stable identity keep memory-only
banks. No classifier, chat API, disk I/O or TTS inference runs on a warmed hit.
Each conversation avoids its last three emitted phrases.

Changed voice preferences use the prepared text and synthesize with the captured
instructions rather than replay stale audio. Explicit `filler_system` or
`filler_input` edits keep dynamic generation. Qwen uses its existing local
short-text generator; the encoder adapter delegates this optional customized
generation to `independent_filler.py`, following the selected reply API or
explicit `STUDIO_FILLER_*` configuration. Default cached fillers make no such
API requests. Local reply users who customize filler prompts can configure an
OpenAI-compatible loopback auxiliary service explicitly.

The 1.5-second default eligibility threshold, cooldown, readiness race, silent
preference, full clip playback and client completion acknowledgement remain in
the conversation handoff owner. Preparing phrases does not authorize playing
them on ordinary turns or while the user is speaking.

## Native comparison

```bash
.venv/bin/python -m evals.voice_router_compare \
  --checkpoint studio/models/reply-router/VoiceMem-Jev/onnx \
  --order ABBA --out results/voice-router-mac-comparison.json
```

This uses uncached, synthetic Chinese/English text and bounded history on native
Apple Silicon. A is MLX Qwen with its current policy; B is the CPU ONNX encoder.
It reports cold load separately from per-turn median/p95, false-deep and missed-
deep decisions, process CPU time and cumulative process peak RSS. Peak RSS
includes both loaded models and is not a per-model memory comparison. The
script uses no APIs, recordings or Memory Spaces. Training independence of
existing synthetic checks is not established. It does not measure actual ASR,
GPU contention, browser playback, end-to-end voice delay, power or sustained heat.
