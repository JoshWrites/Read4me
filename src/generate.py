"""
Core TTS generation API — engine-agnostic.

Both the CLI (text_to_speech.py) and the TUI (tui.py) call this module.
Neither interface knows which TTS engine is being used.

Usage:
    from src.generate import generate

    wav_path = generate(
        text="Hello world.",
        voice_path="voices/jej2.mp3",
        output_dir=".",
        engine_name="chatterbox",   # default
        device="auto",
        # any engine-specific kwargs:
        language="English",
        exaggeration=0.5,
    )

To add a new engine: see src/engines/registry.py.
"""

from __future__ import annotations

import os
import sys

# Ensure repo root and src/ are on sys.path so engine and folder_paths imports resolve
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def generate(
    text: str,
    voice_path: str | None,
    output_dir: str = ".",
    engine_name: str = "chatterbox",
    device: str = "auto",
    output_filename: str | None = None,
    **engine_params,
) -> str:
    """
    Synthesise speech and write a WAV file.

    Args:
        text:             Text to read aloud.
        voice_path:       Reference voice file (.wav / .mp3).
                          May be None if the engine doesn't require one.
        output_dir:       Directory for the output WAV.
        engine_name:      Registered engine slug (default: 'chatterbox').
        device:           Torch device — 'auto', 'cuda', 'mps', or 'cpu'.
        output_filename:  Output filename (default: 'output.wav').
        **engine_params:  Passed to engine.load(), set_voice(), generate_audio().

    Returns:
        Absolute path to the written WAV file.
    """
    import numpy as np
    import soundfile as sf
    from engines.registry import get_engine

    text = text.strip()
    if not text:
        raise ValueError("text must not be empty")

    engine = get_engine(engine_name)

    if engine.requires_voice_file:
        if not voice_path or not os.path.isfile(voice_path):
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

    os.makedirs(output_dir, exist_ok=True)
    fname = (output_filename or "output.wav")
    if not fname.lower().endswith(".wav"):
        fname += ".wav"
    out_path = os.path.abspath(os.path.join(output_dir, fname))

    # Load model (adapter caches; skips if device+params unchanged)
    engine.load(device, **engine_params)

    # Prepare voice
    if engine.requires_voice_file:
        engine.set_voice(voice_path, **engine_params)

    # Generate
    audio, sr = engine.generate_audio(text, **engine_params)

    # Normalise to 1-D float32 numpy array
    if hasattr(audio, "cpu"):                          # torch.Tensor
        audio = audio.squeeze().detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).squeeze()

    sf.write(out_path, audio, sr, subtype="FLOAT")
    print(f"Saved: {out_path}  ({sr} Hz, {len(audio) / sr:.1f}s)")

    return out_path
