"""
Minimal fallback utilities used by ChatterboxTTS.from_pretrained() when a
requested language model isn't found locally.
"""
import os

try:
    import folder_paths as _fp
    _models_dir = _fp.models_dir
except Exception:
    _models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")


def get_models_dir() -> str:
    """Return the root models directory."""
    return _models_dir


def try_local_first(
    search_paths: list,
    local_loader,
    fallback_loader,
    fallback_name: str = "English",
    original_request: str = "",
):
    """
    Try each path in search_paths with local_loader; call fallback_loader if none work.
    """
    for path in search_paths:
        if os.path.isdir(path):
            try:
                return local_loader(path)
            except Exception:
                continue
    return fallback_loader()
