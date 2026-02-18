# Read4me

Read any text aloud using a target voice — locally, with no cloud dependency.

Built on [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) (ResembleAI).

---

## What it does

- Takes a text file (`.txt` or `.md`) and a reference voice file (`.wav` or `.mp3`).
- Clones the voice using Chatterbox's TTS engine.
- Outputs a 24 kHz `.wav` file.
- Runs entirely on your local hardware (CPU, CUDA, or Apple MPS).

---

## Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/Read4me.git
cd Read4me

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

The Chatterbox model (~1 GB) is downloaded from HuggingFace automatically on first use
and cached in `models/TTS/chatterbox/`.

---

## Usage

### TUI (recommended)

```bash
python tui.py
```

The TUI walks you through:

1. Select a script file — file picker opens in `scripts/` by default
2. Select a voice file — file picker opens in `voices/` by default
3. Adjust optional settings (language, emotion, temperature, CFG weight, device)
4. Select an output directory
5. Press **Generate** (or `Ctrl+G`)

### CLI

```bash
python text_to_speech.py --text-file scripts/my_script.txt \
                         --voice voices/jej2.mp3 \
                         --out-dir .
```

Or with inline text:

```bash
python text_to_speech.py "Hello. This is the target voice." \
                         --voice voices/jej2.mp3
```

Full options:

```
  TEXT                Inline text to speak (or use --text-file)
  -f, --text-file     Path to .txt or .md file
  -v, --voice         Reference voice file (.wav or .mp3)  [required]
  -o, --out-dir       Output directory (default: .)
  -n, --filename      Output filename (default: output.wav)
  -l, --language      Language model (default: English)
  -d, --device        Torch device: auto, cuda, mps, cpu (default: auto)
  --exaggeration      Emotion 0.0–1.0+ (default: 0.5)
  --temperature       Sampling temperature (default: 0.8)
  --cfg-weight        CFG weight (default: 0.5)
```

---

## Directory layout

```
Read4me/
├── scripts/        ← put your .txt / .md scripts here
├── voices/         ← put your reference voice files here
├── models/         ← Chatterbox model weights (auto-downloaded)
├── tui.py          ← TUI entrypoint
├── text_to_speech.py ← CLI entrypoint
├── src/
│   ├── generate.py ← core API (used by both CLI and TUI)
│   └── engines/
│       └── chatterbox_standalone/  ← bundled TTS engine
└── requirements.txt
```

---

## Tips

- Drop your target voice files (`.wav` or `.mp3`) in `voices/` for easy access.
- For best results, use a clean 10–30 second voice sample with minimal background noise.
- For long texts, break them into shorter paragraphs — Chatterbox handles ~1–2 paragraphs per call best.
- `--device auto` picks CUDA → MPS → CPU automatically.
