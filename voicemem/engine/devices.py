"""Canonical device labels without importing an optional model runtime."""
import sys


def device_key(device):
    """Normalize explicit CUDA aliases; leave unspecified placement conservative."""
    if device is None:
        return None
    name = str(device).lower()
    if name == "cuda":
        torch = sys.modules.get("torch")
        cuda = getattr(torch, "cuda", None)
        index = cuda.current_device() if cuda is not None and cuda.is_initialized() else 0
        return f"cuda:{index}"
    if name.startswith("cuda:"):
        return f"cuda:{int(name.split(':', 1)[1])}"
    return name
