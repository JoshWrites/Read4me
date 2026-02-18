"""
Torch device resolution utility.
"""
import torch


def resolve_torch_device(device: str) -> str:
    """Resolve 'auto' to the best available device; pass through explicit values."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
