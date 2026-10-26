"""Correlate confirmed input and output timing without recording dialogue."""


def playback_timing(pending, timeline, now):
    """Return stage durations, retaining overlap for adopted speculative output."""
    speech_end = getattr(pending, 'speech_end', 0.0)
    if not speech_end:
        return {}
    source = getattr(pending, 'timing', None) or {}
    generation = getattr(timeline, 'latency', None) or {}
    names = ('endpoint_ms', 'asr_tail_ms', 'confirm_ms', 'gate_ms',
             'gate_queue_ms', 'gate_worker_ms', 'gate_timeout',
             'retrieval_ms', 'final_queue_ms', 'final_compute_ms',
             'snapshot_reused', 'capture_ms', 'route_ms')
    result = {name: source[name] for name in names if name in source}
    if 'route_started_at' in source and 'captured_at' in source:
        result['before_route_ms'] = (source['route_started_at']-source['captured_at'])*1000
    if 'started_at' in generation:
        result['effective_head_ms'] = (generation['started_at']-speech_end)*1000
        if 'route_finished_at' in source:
            gap = (generation['started_at']-source['route_finished_at'])*1000
            result['handoff_ms'] = max(0.0, gap)
            result['overlap_ms'] = max(0.0, -gap)
    for name in ('llm_ms', 'segment_ms', 'tts_ms'):
        if name in generation:
            result[name] = generation[name]
    if 'first_audio_at' in generation:
        result['delivery_ms'] = (now-generation['first_audio_at'])*1000
    result['playback_ms'] = (now-speech_end)*1000
    return result


def log_playback_timing(pending, timeline, now):
    """Print one metadata-only row after a confirmed output starts playing."""
    values = playback_timing(pending, timeline, now)
    if values:
        fields = ' '.join(f'{name}={value:.0f}' for name, value in values.items())
        print(f'[lat-breakdown] turn={getattr(pending, "input_turn_id", "") or "-"}'
              f' output={timeline.output_id} {fields}', flush=True)
