"""
Canonical TTS parameter registry.

Each entry defines the *user-facing* name, description, and sensible range
for a concept that may be called different things by different engines
(e.g. "speed", "pace", "rate", "speaking_rate" all map to canonical "speed").

When an EngineParam sets canonical="speed", the TUI automatically displays
the canonical label and description instead of the engine-specific ones,
keeping controls visually consistent as the user switches engines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CanonicalParam:
    """User-facing definition of a normalised TTS control."""
    id: str
    label: str
    description: str
    type: str                       # "float" | "int" | "select" | "str"
    default: Any = None
    min_val: float | None = None
    max_val: float | None = None


# ── Registry ──────────────────────────────────────────────────────────────────
#
# Keys are the canonical IDs that EngineParam.canonical should reference.
# Add new entries here as more engines are integrated.

CANONICAL: dict[str, CanonicalParam] = {

    # ── Prosody ───────────────────────────────────────────────────────────────
    "speed": CanonicalParam(
        id="speed",
        label="Speed",
        description="Speaking rate  (1.0 = normal, 0.5 = half, 2.0 = double)",
        type="float",
        default=1.0,
        min_val=0.1,
        max_val=3.0,
    ),
    "pitch": CanonicalParam(
        id="pitch",
        label="Pitch",
        description="Voice pitch offset in semitones  (0 = unchanged)",
        type="float",
        default=0.0,
        min_val=-12.0,
        max_val=12.0,
    ),
    "volume": CanonicalParam(
        id="volume",
        label="Volume",
        description="Output loudness multiplier  (1.0 = unchanged)",
        type="float",
        default=1.0,
        min_val=0.0,
        max_val=2.0,
    ),

    # ── Voice character ───────────────────────────────────────────────────────
    "emotion": CanonicalParam(
        id="emotion",
        label="Emotion",
        description="Emotional expressiveness / exaggeration  (0 – 1.5)",
        type="float",
        default=0.5,
        min_val=0.0,
        max_val=1.5,
    ),

    # ── Sampling / model ──────────────────────────────────────────────────────
    "temperature": CanonicalParam(
        id="temperature",
        label="Temperature",
        description="Sampling randomness — higher = more varied output  (0.1 – 1.5)",
        type="float",
        default=0.8,
        min_val=0.1,
        max_val=1.5,
    ),
    "guidance": CanonicalParam(
        id="guidance",
        label="Guidance",
        description="Classifier-free guidance strength  (0 = off, 1 = strong)",
        type="float",
        default=0.5,
        min_val=0.0,
        max_val=1.0,
    ),
    "seed": CanonicalParam(
        id="seed",
        label="Seed",
        description="Random seed for reproducible output  (-1 = random each time)",
        type="int",
        default=-1,
    ),

    # ── Language / model variant ──────────────────────────────────────────────
    "language": CanonicalParam(
        id="language",
        label="Language",
        description="Language / model variant",
        type="select",
        default="English",
    ),
}


def resolve(param_canonical: str | None) -> CanonicalParam | None:
    """Return the CanonicalParam for a given canonical id, or None."""
    if param_canonical is None:
        return None
    return CANONICAL.get(param_canonical)
