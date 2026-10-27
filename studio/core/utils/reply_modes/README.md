# Local depth router adapters

The recommended router is Qwen3-0.6B on MLX or Torch. The optional task-specific
encoder implements the same `ThinkingDecision` boundary. It decides ordinary
versus deep reasoning; VoiceMem continues to own memory eligibility. Neither
adapter can end a voice turn or release private speculative output.

## Trained v6 checkpoint

The curated checkpoint is available on the repository's
[`feat/audio-router` branch](https://github.com/lang-jiaqi/VoiceMem-Studio/tree/feat/audio-router/artifacts/voice-router-v6).
It is a 102,274,064-parameter MacBERT encoder with a calibrated binary head.
Keep `model.onnx`, `depth_encoder.json`, `encoder/`, and `tokenizer/` together.
The ONNX weight SHA-256 is
`35ef159d8a14aa4b81eb959663af01508e76cc80609860e94956abcd3a939290`.
Weights are runtime data and do not belong in this working tree's source diff.

For native Mac testing, place the CPU bundle under
`studio/models/reply-router/voice-router-v6/onnx`. In **Settings → Components →
Router → Adapter**, use:

- Factory: `studio.core.utils.reply_modes.encoder_depth:create`
- Model: `studio/models/reply-router/voice-router-v6/onnx` from the repository root,
  or an absolute path to the complete bundle.
- Options: `{"backend":"onnx","device":"cpu","allow_prototype":true}`
- API Key and Base URL: empty.

The same startup selection is:

```dotenv
STUDIO_COMPONENTS='{"router":{"provider":"adapter","factory":"studio.core.utils.reply_modes.encoder_depth:create","model":"studio/models/reply-router/voice-router-v6/onnx","options":{"backend":"onnx","device":"cpu","allow_prototype":true}}}'
```

Selection uses the existing apply/restart flow. Restore the recommended router
to compare against Qwen. `allow_prototype` explicitly selects an experiment;
it does not establish natural-dialogue accuracy. The fixed classifier is trained
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
  --checkpoint studio/models/reply-router/voice-router-v6/onnx \
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
