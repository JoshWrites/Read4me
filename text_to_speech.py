#!/usr/bin/env python3
"""
Read4me — CLI

Generate a WAV file of text read aloud in a target voice.

Usage:
    python text_to_speech.py --text-file scripts/my_script.txt \\
                             --voice voices/jej2.mp3 \\
                             --out-dir .

    python text_to_speech.py "Short sentence." --voice voices/jej2.mp3

Options:
    TEXT                Inline text to speak (or use --text-file)
    -f, --text-file     Path to .txt or .md file
    -v, --voice         Reference voice file (.wav or .mp3)  [required]
    -o, --out-dir       Output directory (default: current dir)
    -n, --filename      Output filename (default: output.wav)
    -l, --language      Language model (default: English)
    -d, --device        Torch device: auto, cuda, mps, cpu (default: auto)
    --exaggeration      Emotion 0.0–1.0+ (default: 0.5)
    --temperature       Sampling temperature (default: 0.8)
    --cfg-weight        CFG weight (default: 0.5)
"""

import argparse
import os
import sys

# Ensure repo root is on path
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read4me: text to speech in a target voice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("text", nargs="?", default=None, help="Inline text to speak")
    parser.add_argument("-f", "--text-file", help="Path to .txt or .md script file")
    parser.add_argument("-v", "--voice", required=True, help="Reference voice file (.wav/.mp3)")
    parser.add_argument("-o", "--out-dir", default=".", help="Output directory")
    parser.add_argument("-n", "--filename", default="output.wav", help="Output filename")
    parser.add_argument("-l", "--language", default="English", help="Language model")
    parser.add_argument("-d", "--device", default="auto", help="Torch device")
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    args = parser.parse_args()

    # Resolve text
    if args.text_file:
        if not os.path.isfile(args.text_file):
            print(f"Error: text file not found: {args.text_file}")
            return 1
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            print(f"Error: text file is empty: {args.text_file}")
            return 1
        print(f"Read {len(text):,} characters from {args.text_file}")
    elif args.text:
        text = args.text
    else:
        parser.error("Provide text via --text-file or as a positional argument")

    from src.generate import generate

    try:
        out_path = generate(
            text=text,
            voice_path=args.voice,
            output_dir=args.out_dir,
            language=args.language,
            device=args.device,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
            output_filename=args.filename,
        )
        print(f"\nDone: {out_path}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
