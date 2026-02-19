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
import re
import sys

# Ensure repo root and src/ are on sys.path so engine and folder_paths imports resolve
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Maximum characters per chunk sent to the engine.
# Chatterbox's T3 transformer has a finite context window; feeding it more
# than ~200 words causes it to run out of tokens and loop its output.
# At ~5 chars/word, 800 chars ≈ 160 words — comfortably inside the window.
_CHUNK_CHARS = 800


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries, preserving whitespace."""
    # Split after . ! ? followed by whitespace or end-of-string.
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _chunk_text(text: str, max_chars: int = _CHUNK_CHARS) -> list[str]:
    """
    Group sentences into chunks no longer than max_chars.

    Each chunk ends at a sentence boundary so the engine never receives a
    fragment mid-sentence.  A single sentence longer than max_chars is kept
    whole — splitting mid-sentence produces unnatural speech.
    """
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Start a new chunk.  If this single sentence exceeds max_chars,
            # keep it whole — splitting mid-sentence sounds worse than long.
            current = sentence

    if current:
        chunks.append(current)

    return chunks


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

    Long text is automatically split into sentence-boundary chunks and
    generated separately, then concatenated into a single WAV.  A short
    silence is inserted between chunks so the output sounds natural.

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

    # Prepare voice (once — reused across all chunks)
    if engine.requires_voice_file:
        engine.set_voice(voice_path, **engine_params)

    # Split text into chunks sized for this engine's context window
    chunks = _chunk_text(text, max_chars=engine.chunk_chars)
    print(f"Generating {len(chunks)} chunk(s)  ({len(text)} chars total)")

    audio_parts: list[np.ndarray] = []
    sr: int = 24000  # will be set from first chunk

    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}/{len(chunks)}: {len(chunk)} chars")
        audio, sr = engine.generate_audio(chunk, **engine_params)

        # Normalise to 1-D float32 numpy array
        if hasattr(audio, "cpu"):
            audio = audio.squeeze().detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        audio_parts.append(audio)

        # Add a short natural pause between chunks (0.3 s of silence)
        if i < len(chunks):
            audio_parts.append(np.zeros(int(sr * 0.3), dtype=np.float32))

    full_audio = np.concatenate(audio_parts)
    sf.write(out_path, full_audio, sr, subtype="FLOAT")
    print(f"Saved: {out_path}  ({sr} Hz, {len(full_audio) / sr:.1f}s)")

    return out_path
