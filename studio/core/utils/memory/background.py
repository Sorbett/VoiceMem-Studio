"""Bind bounded memory preparation and acoustic native work to one Studio agent."""
from voicemem.engine import BackgroundLane, BackgroundPipeline


def background_lane(owner):
    """Reuse the agent's independent acoustic UI lane."""
    lane = getattr(owner, "_BACKGROUND_LANE", None)
    if lane is None:
        lane = owner._BACKGROUND_LANE = BackgroundLane()
    return lane


def memory_pipeline(owner):
    """Keep at most two confirmed turns in preparation or durable completion."""
    pipeline = getattr(owner, "_MEMORY_PIPELINE", None)
    if pipeline is None:
        pipeline = owner._MEMORY_PIPELINE = BackgroundPipeline(max_active=2)
    return pipeline
