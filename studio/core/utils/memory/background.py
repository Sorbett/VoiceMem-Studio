"""Bind the engine's native-work lane to one Studio agent."""
from voicemem.engine import BackgroundLane


def background_lane(owner):
    """Reuse the agent's lane for ingestion and acoustic UI inference."""
    lane = getattr(owner, "_BACKGROUND_LANE", None)
    if lane is None:
        lane = owner._BACKGROUND_LANE = BackgroundLane()
    return lane
