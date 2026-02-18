"""
Chatterbox TTS engine adapter.
"""
from __future__ import annotations

from .base import TTSEngine, EngineParam


_LANGUAGES: list[tuple[str, str]] = [
    ("English",    "English"),
    ("German",     "German"),
    ("French",     "French"),
    ("Norwegian",  "Norwegian"),
    ("Italian",    "Italian"),
    ("Spanish",    "Spanish"),
    ("Russian",    "Russian"),
    ("Arabic",     "Arabic"),
    ("Turkish",    "Turkish"),
    ("Vietnamese", "Vietnamese"),
]


class ChatterboxAdapter(TTSEngine):
    name = "chatterbox"
    display_name = "Chatterbox"
    requires_voice_file = True

    def __init__(self) -> None:
        self._tts = None
        self._loaded_device: str | None = None
        self._loaded_language: str | None = None

    @classmethod
    def params(cls) -> list[EngineParam]:
        return [
            EngineParam(
                id="language",
                label="Language",
                type="select",
                default="English",
                required=False,
                description="Language model variant",
                options=_LANGUAGES,
            ),
            EngineParam(
                id="exaggeration",
                label="Exaggeration",
                type="float",
                default=0.5,
                required=False,
                description="Emotion intensity  (0 – 1.5)",
                min_val=0.0,
                max_val=1.5,
            ),
            EngineParam(
                id="temperature",
                label="Temperature",
                type="float",
                default=0.8,
                required=False,
                description="Sampling randomness  (0.1 – 1.5)",
                min_val=0.1,
                max_val=1.5,
            ),
            EngineParam(
                id="cfg_weight",
                label="CFG Weight",
                type="float",
                default=0.5,
                required=False,
                description="Guidance strength  (0 – 1)",
                min_val=0.0,
                max_val=1.0,
            ),
        ]

    # ------------------------------------------------------------------

    def load(self, device: str, **params) -> None:
        language = params.get("language", "English")
        if (self._tts is not None
                and self._loaded_device == device
                and self._loaded_language == language):
            return  # already loaded, skip

        from engines.chatterbox_standalone import ChatterboxTTS
        print(f"Loading Chatterbox  language={language}  device={device}")
        self._tts = ChatterboxTTS.from_pretrained(device, language=language)
        self._loaded_device = device
        self._loaded_language = language

    def set_voice(self, voice_path: str, **params) -> None:
        exaggeration = float(params.get("exaggeration", 0.5))
        self._tts.prepare_conditionals(voice_path, exaggeration=exaggeration)

    def generate_audio(self, text: str, **params) -> tuple:
        audio = self._tts.generate(
            text,
            audio_prompt_path=None,
            exaggeration=float(params.get("exaggeration", 0.5)),
            cfg_weight=float(params.get("cfg_weight", 0.5)),
            temperature=float(params.get("temperature", 0.8)),
        )
        return audio, self._tts.sr
