# Local depth router adapters

The recommended router is VoiceMem-Jev, a task-specific encoder implementing
the existing `ThinkingDecision` boundary. It decides ordinary
versus deep reasoning; VoiceMem continues to own memory eligibility. The
router cannot end a voice turn or release private speculative output.

## VoiceMem-Jev

The recommended router uses [Sorbett/VoiceMem-Jev](https://huggingface.co/Sorbett/VoiceMem-Jev)
as its single model and runtime source. Run Studio normally; first startup
automatically downloads the required format and the repository's Python runtime.
No router configuration or separate package installation is needed.

| Router device | Downloaded format | Runtime |
| --- | --- | --- |
| CUDA | `eager/` | Published PyTorch router on the selected device |
| Mac or CPU | `onnx/` | Published CPU ONNX router |

Both formats belong to the same model repository. The cache is
`studio/models/reply-router/VoiceMem-Jev`, or its equivalent under
`STUDIO_MODELS_DIR`. Complete caches work offline. Missing weights, tokenizer or
runtime files are acquired before warmup. Downloads use the Hub's default
revision; Studio carries no release SHA or per-file hash catalog.

The bundled `voicemem_jev` runtime owns model structure, tokenization,
preprocessing, observation layout and calibrated predictions. Studio retains
its engine scheduling, cancellation, exact-request cache and independent filler
interface. CPU ONNX does not acquire a Torch/MPS device lock. Live routing does
not download files or install dependencies.

Explicit local adapters retain the existing factory:
`studio.core.utils.reply_modes.encoder_depth:create`. Choose the local `onnx/`
or `eager/` model directory, with options such as
`{"backend":"onnx","device":"cpu","allow_prototype":true}` or
`{"backend":"eager","device":"cuda:0","allow_prototype":true}`.
Keep `python/voicemem_jev` beside the format directories, or install the model's
Python runtime into the environment. API Key and Base URL stay empty. Published
Studio adapter subclasses retain their existing load hook.

The model's prototype opt-in remains explicit for custom adapters. The
recommended default accepts the published model. Learned decisions do not use
the editable generative-router prompt; reply, TTS and custom filler prompts keep
their existing request scope. Memory eligibility, EOT and speculative output
commitment remain owned by their existing pipeline stages.

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
