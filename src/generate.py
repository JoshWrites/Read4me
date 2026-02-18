"""
Core TTS generation API.

Both the CLI (text_to_speech.py) and the TUI (tui.py) call this module.
Neither interface imports from the engine directly.

Usage:
    from src.generate import generate

    wav_path = generate(
        text="Hello world.",
        voice_path="voices/jej2.mp3",
        output_dir=".",
    )
"""

import os
import sys

# Ensure repo root is on the path so `import folder_paths` resolves
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def generate(
    text: str,
    voice_path: str,
    output_dir: str = ".",
    language: str = "English",
    device: str = "auto",
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    cfg_weight: float = 0.5,
    output_filename: str | None = None,
) -> str:
    """
    Generate speech from text using a reference voice and save it as a WAV file.

    Args:
        text:             The text to read aloud.
        voice_path:       Path to a reference voice file (.wav or .mp3).
        output_dir:       Directory where the output WAV will be written.
        language:         Chatterbox language model to use (default: 'English').
        device:           Torch device — 'auto', 'cuda', 'mps', or 'cpu'.
        exaggeration:     Emotion exaggeration factor (0.0–1.0+).
        temperature:      Sampling temperature (higher = more varied).
        cfg_weight:       Classifier-free guidance weight.
        output_filename:  Optional explicit filename (without directory).
                          Defaults to 'output.wav'.

    Returns:
        Absolute path to the written WAV file.

    Raises:
        FileNotFoundError: If voice_path does not exist.
        ValueError:        If text is empty.
    """
    import soundfile as sf
    from engines.chatterbox_standalone import ChatterboxTTS

    # Validate inputs
    text = text.strip()
    if not text:
        raise ValueError("text must not be empty")
    if not os.path.isfile(voice_path):
        raise FileNotFoundError(f"Voice file not found: {voice_path}")

    os.makedirs(output_dir, exist_ok=True)

    fname = output_filename or "output.wav"
    if not fname.lower().endswith(".wav"):
        fname += ".wav"
    out_path = os.path.abspath(os.path.join(output_dir, fname))

    print(f"Loading TTS model ({language}, {device})...")
    tts = ChatterboxTTS.from_pretrained(device, language=language)

    print(f"Loading voice reference: {voice_path}")
    tts.prepare_conditionals(voice_path, exaggeration=exaggeration)

    print("Generating speech...")
    audio = tts.generate(
        text,
        audio_prompt_path=None,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )

    wav = audio.squeeze(0).cpu().numpy()
    sf.write(out_path, wav, tts.sr, subtype="FLOAT")

    duration = len(wav) / tts.sr
    print(f"Saved: {out_path}  ({tts.sr} Hz, {duration:.1f}s)")

    return out_path
