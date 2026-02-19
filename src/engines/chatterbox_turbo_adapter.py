"""
Chatterbox Turbo TTS engine adapter.

Uses the pip-installed `chatterbox-tts` package (ChatterboxTurboTTS), which is
distinct from the vendored `chatterbox_standalone/` used by ChatterboxAdapter.

Key differences from Chatterbox (original):
  - English only — no language selector
  - 350M parameters; single-step mel decoder (faster, lower VRAM)
  - cfg_weight defaults to 0.0 (guidance is minimal by design)
  - exaggeration defaults to 0.0 in generate(), but 0.5 in prepare_conditionals()
  - Paralinguistic tags: inline markers in the text string trigger natural vocal
    reactions performed in the cloned voice — no post-processing required.

Supported paralinguistic tags (place anywhere in the script text):
    [laugh]        [chuckle]      [sigh]         [gasp]
    [cough]        [groan]        [sniff]        [clear throat]
"""
from __future__ import annotations

from .base import TTSEngine, EngineParam

# Paralinguistic tags displayed as a hint in the TUI engine panel.
_TAGS_HINT = "[laugh] [chuckle] [sigh] [gasp] [cough] [groan] [sniff] [clear throat]"


class ChatterboxTurboAdapter(TTSEngine):
    name = "chatterbox-turbo"
    display_name = "Chatterbox Turbo"
    requires_voice_file = True
    # Turbo's GPT2-medium backbone has a 1024-token context window shared
    # between: 375 conditioning tokens + text tokens + generated speech tokens.
    # At ~4.5 chars/token and ~16 speech tokens/second, 400 chars ≈ 90 text
    # tokens, leaving ~560 speech token budget ≈ 35 seconds — safely above
    # the ~32 seconds a 400-char chunk of natural speech actually takes.
    chunk_chars: int = 400

    def __init__(self) -> None:
        self._tts = None
        self._loaded_device: str | None = None

    @classmethod
    def params(cls) -> list[EngineParam]:
        return [
            EngineParam(
                id="exaggeration",
                label="Exaggeration",
                type="float",
                default=0.5,
                required=False,
                description="Emotion intensity  (0 – 1.5)",
                min_val=0.0,
                max_val=1.5,
                canonical="emotion",
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
                canonical="temperature",
            ),
            EngineParam(
                id="cfg_weight",
                label="CFG Weight",
                type="float",
                default=0.0,
                required=False,
                description="Guidance strength — keep near 0 for Turbo  (0 – 1)",
                min_val=0.0,
                max_val=1.0,
                canonical="guidance",
            ),
            # Read-only hint — the adapter ignores this kwarg at generation time.
            EngineParam(
                id="_tags_hint",
                label="Paralinguistic Tags",
                type="str",
                default=_TAGS_HINT,
                required=False,
                description="Place these tags inline in your script to trigger natural vocal reactions",
            ),
        ]

    # ------------------------------------------------------------------

    def load(self, device: str, **params) -> None:
        from utils.device import resolve_torch_device
        resolved = resolve_torch_device(device)

        if self._tts is not None and self._loaded_device == resolved:
            return  # already loaded

        from chatterbox.tts_turbo import ChatterboxTurboTTS
        print(f"Loading Chatterbox Turbo  device={resolved}")
        self._tts = ChatterboxTurboTTS.from_pretrained(resolved)
        self._loaded_device = resolved
        self._patch_float32(self._tts)

    @staticmethod
    def _patch_float32(tts) -> None:
        """
        Patch norm_loudness to always return float32.

        pyloudnorm returns float64 scalars; multiplying a float32 numpy array by
        a float64 scalar silently upcasts the whole array to float64.  MPS
        (Apple Silicon GPU) refuses float64 tensors, so we must clamp back to
        float32 before the audio ever touches torch.from_numpy().
        """
        import math
        import types
        import numpy as np

        try:
            import pyloudnorm as ln
        except ImportError:
            return

        def _norm_loudness_f32(self, wav, sr, target_lufs=-27):
            try:
                meter = ln.Meter(sr)
                loudness = meter.integrated_loudness(wav)
                gain_db = target_lufs - loudness
                gain_linear = float(10.0 ** (gain_db / 20.0))  # plain Python float → no float64 upcast
                if math.isfinite(gain_linear) and gain_linear > 0.0:
                    wav = wav * np.float32(gain_linear)
            except Exception as e:
                print(f"Warning: norm_loudness skipped: {e}")
            return np.asarray(wav, dtype=np.float32)

        tts.norm_loudness = types.MethodType(_norm_loudness_f32, tts)

    def set_voice(self, voice_path: str, **params) -> None:
        exaggeration = float(params.get("exaggeration", 0.5))
        self._tts.prepare_conditionals(voice_path, exaggeration=exaggeration)

    def generate_audio(self, text: str, **params) -> tuple:
        audio = self._tts.generate(
            text,
            audio_prompt_path=None,
            exaggeration=float(params.get("exaggeration", 0.5)),
            cfg_weight=float(params.get("cfg_weight", 0.0)),
            temperature=float(params.get("temperature", 0.8)),
            # repetition_penalty=1.2 (library default) progressively suppresses
            # phoneme tokens that have already appeared.  Natural speech constantly
            # reuses the same sounds, so the penalty causes increasing silence and
            # stuttering as generation proceeds.  Disable it for clean output.
            repetition_penalty=1.0,
        )
        return audio, self._tts.sr
