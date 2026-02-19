"""
Abstract base for all TTS engines.

To add a new engine:
  1. Create src/engines/my_engine_adapter.py  implementing TTSEngine.
  2. Register it in src/engines/registry.py with register(MyEngineAdapter).
  The CLI and TUI will pick it up automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class EngineParam:
    """
    Describes a single configurable parameter for a TTS engine.

    The TUI reads this list to build the engine settings panel dynamically.

    Set `canonical` to a key from `src/engines/canonical.CANONICAL` to opt
    into the normalised label/description system.  The TUI will then display
    the canonical label (e.g. "Speed") regardless of what the engine calls
    this parameter internally (e.g. "pace", "rate", "speaking_rate"), keeping
    the UI visually consistent as the user switches between engines.
    """
    id: str                            # kwarg key passed to generate_audio() / set_voice()
    label: str                         # fallback display name (used when canonical is None)
    type: Literal["float", "int", "str", "select"]
    default: Any                       # shown in the input field; None means required
    required: bool = False             # True → user MUST fill this in (shown with ◆ indicator)
    description: str = ""              # fallback hint (used when canonical is None)
    options: list[tuple[str, str]] | None = None  # (display_label, value) for "select" type
    min_val: float | None = None
    max_val: float | None = None
    canonical: str | None = None       # key into canonical.CANONICAL  e.g. "speed", "emotion"


class TTSEngine(ABC):
    """
    Abstract interface every TTS engine adapter must implement.

    Lifecycle per generation call:
        engine.load(device, **params)        ← idempotent; skip if already loaded
        engine.set_voice(voice_path, **params)  ← only called if requires_voice_file
        audio, sr = engine.generate_audio(text, **params)
    """

    # Class-level constants — override in each subclass
    name: str = ""           # unique slug  e.g. "chatterbox"
    display_name: str = ""   # shown in TUI e.g. "Chatterbox"
    requires_voice_file: bool = True  # False for engines with built-in voices
    # Maximum characters per chunk sent to generate_audio().
    # Engines with tighter context windows should set a smaller value.
    chunk_chars: int = 800

    @classmethod
    @abstractmethod
    def params(cls) -> list[EngineParam]:
        """
        Return the ordered list of engine-specific parameters.
        The TUI renders these into the right-hand panel.
        Required params (required=True) are shown with a ◆ indicator.
        """
        ...

    @abstractmethod
    def load(self, device: str, **params) -> None:
        """
        Load / initialise the model.  Called before every generate() but
        implementations should cache and skip if nothing changed.

        Args:
            device: 'auto', 'cuda', 'mps', or 'cpu'
            **params: the engine params (e.g. language, api_key, …)
        """
        ...

    @abstractmethod
    def set_voice(self, voice_path: str, **params) -> None:
        """
        Prepare the reference voice.
        Only called when requires_voice_file is True.
        """
        ...

    @abstractmethod
    def generate_audio(self, text: str, **params) -> tuple[Any, int]:
        """
        Synthesise speech.

        Returns:
            (audio, sample_rate)
            audio may be a torch.Tensor (any shape) or a numpy ndarray.
            src/generate.py normalises it to a 1-D float32 numpy array.
        """
        ...
