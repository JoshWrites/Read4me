"""
Engine registry.

Engines are registered as *classes* (not instances) so nothing heavy is
imported at startup.  The adapter is instantiated lazily the first time
get_engine() is called for that engine name.

Adding a new engine:
    from .my_engine import MyEngine
    register(MyEngine)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import TTSEngine

_CLASSES: dict[str, type] = {}
_INSTANCES: dict[str, "TTSEngine"] = {}


def register(cls: type) -> type:
    """Register an engine class.  May be used as a decorator."""
    _CLASSES[cls.name] = cls
    return cls


def get_engine(name: str) -> "TTSEngine":
    """Return the (lazily created) singleton instance for the given engine name."""
    if name not in _CLASSES:
        available = list(_CLASSES.keys())
        raise KeyError(f"Unknown engine '{name}'. Available: {available}")
    if name not in _INSTANCES:
        _INSTANCES[name] = _CLASSES[name]()
    return _INSTANCES[name]


def available_engines() -> list[tuple[str, str]]:
    """
    Return options suitable for a Textual Select widget:
        [(display_label, value), ...]
    """
    return [(cls.display_name, name) for name, cls in _CLASSES.items()]


# ── Register built-in engines ────────────────────────────────────────────────
# Importing the adapter module is cheap — heavy deps (torch, etc.) are only
# imported inside the adapter's load() method.

from .chatterbox_adapter import ChatterboxAdapter  # noqa: E402
register(ChatterboxAdapter)

from .chatterbox_turbo_adapter import ChatterboxTurboAdapter  # noqa: E402
register(ChatterboxTurboAdapter)
