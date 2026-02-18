#!/usr/bin/env python3
"""
Read4me — CLI

Usage:
    python text_to_speech.py --text-file scripts/script.txt \\
                             --voice voices/jej2.mp3

    python text_to_speech.py "Short phrase." --voice voices/jej2.mp3

    # Use a different engine (when one is registered):
    python text_to_speech.py -f script.txt -v voice.mp3 --engine my_engine

Options:
    TEXT                Inline text (or use --text-file)
    -f, --text-file     Path to .txt or .md script
    -v, --voice         Reference voice file (.wav / .mp3)
    -o, --out-dir       Output directory  (default: .)
    -n, --filename      Output filename   (default: output.wav)
    -e, --engine        Engine name       (default: chatterbox)
    -d, --device        Torch device: auto, cuda, mps, cpu  (default: auto)

Engine-specific options (passed as --param-NAME VALUE):
    --param-language     English
    --param-exaggeration 0.5
    --param-temperature  0.8
    --param-cfg_weight   0.5
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read4me: text to speech in a target voice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("text",         nargs="?",  default=None)
    parser.add_argument("-f", "--text-file")
    parser.add_argument("-v", "--voice",  default=None)
    parser.add_argument("-o", "--out-dir", "--output-dir",
                        default=os.path.join(_REPO_ROOT, "Output"),
                        metavar="DIR")
    parser.add_argument("-n", "--filename", default="output.wav")
    parser.add_argument("-e", "--engine",  default="chatterbox")
    parser.add_argument("-d", "--device",  default="auto")

    # Collect engine-specific params as --param-NAME VALUE
    args, extra = parser.parse_known_args()

    engine_params: dict = {}
    it = iter(extra)
    for token in it:
        if token.startswith("--param-"):
            key = token[len("--param-"):]
            try:
                val_raw = next(it)
            except StopIteration:
                print(f"Error: --param-{key} requires a value")
                return 1
            # Auto-cast to float if possible
            try:
                engine_params[key] = float(val_raw)
            except ValueError:
                engine_params[key] = val_raw
        else:
            print(f"Warning: unrecognised argument '{token}' — ignored")

    # Resolve text
    if args.text_file:
        if not os.path.isfile(args.text_file):
            print(f"Error: text file not found: {args.text_file}")
            return 1
        with open(args.text_file, encoding="utf-8") as fh:
            text = fh.read().strip()
        if not text:
            print(f"Error: text file is empty: {args.text_file}")
            return 1
        print(f"Read {len(text):,} characters from {args.text_file}")
    elif args.text:
        text = args.text
    else:
        parser.error("Provide text via --text-file or as a positional argument")

    # Validate voice (engines that don't need it will skip)
    from src.generate import generate

    try:
        out_path = generate(
            text=text,
            voice_path=args.voice,
            output_dir=args.out_dir,
            engine_name=args.engine,
            device=args.device,
            output_filename=args.filename,
            **engine_params,
        )
        print(f"\nDone: {out_path}")
        return 0
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
