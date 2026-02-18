# AGENTS.md — Read4me Agent Instructions

Guidelines for AI agents contributing to this repository.
Read this file before making any changes to the codebase.

---

## 1. Adding a New TTS Engine

### Maintain the Canonical Translation Layer

`src/engines/canonical.py` is the single source of truth for user-facing
parameter labels.  The goal is that the user always sees the same control
name (e.g. **"Speed"**) regardless of what the underlying engine calls it
internally (`pace`, `rate`, `speaking_rate`, etc.).

**Every time you add or modify an engine adapter, follow these steps:**

#### Step 1 — Audit existing canonicals first

Open `src/engines/canonical.py` and read through the `CANONICAL` dict before
creating any `EngineParam`.  Check whether the concept you need already has
a canonical entry:

| If the engine param is about… | Use canonical |
|-------------------------------|---------------|
| speaking rate / pace / speed / rate | `"speed"` |
| voice pitch / tone / semitones | `"pitch"` |
| output loudness / gain / volume | `"volume"` |
| emotion / expressiveness / exaggeration | `"emotion"` |
| sampling randomness / creativity / variability | `"temperature"` |
| guidance / CFG scale / classifier weight | `"guidance"` |
| reproducibility / random seed | `"seed"` |
| language / locale / model variant | `"language"` |

#### Step 2 — Map each EngineParam to a canonical where one exists

```python
# ✅ CORRECT — user sees "Emotion" regardless of engine
EngineParam(
    id="exaggeration",        # engine-internal kwarg name — do NOT change
    label="Exaggeration",     # fallback if canonical lookup fails
    type="float",
    default=0.5,
    min_val=0.0,
    max_val=1.5,
    canonical="emotion",      # ← resolves to canonical label + description
)

# ❌ WRONG — omitting canonical when one clearly applies
EngineParam(id="pace", label="Pace", type="float", default=1.0)
```

#### Step 3 — Add a new canonical entry when needed

If the engine introduces a genuinely new concept not in `CANONICAL`, add it:

```python
# In src/engines/canonical.py → CANONICAL dict:
"breathiness": CanonicalParam(
    id="breathiness",
    label="Breathiness",
    description="Amount of breath noise in the voice  (0 – 1)",
    type="float",
    default=0.0,
    min_val=0.0,
    max_val=1.0,
),
```

> **Rule:** Never skip the audit (Step 1).  New canonicals should be rare —
> most engine params map to something that already exists.

#### Step 4 — Register the adapter

In `src/engines/registry.py`, import your adapter and call `register()`:

```python
from .my_engine_adapter import MyEngineAdapter
register(MyEngineAdapter)
```

The CLI and TUI will pick it up automatically.

---

## 2. Repository Layout

```
Read4me/
├── src/
│   ├── engines/
│   │   ├── canonical.py          ← canonical param registry  (edit when adding concepts)
│   │   ├── base.py               ← TTSEngine ABC + EngineParam dataclass
│   │   ├── registry.py           ← engine registration
│   │   ├── chatterbox_adapter.py ← reference adapter implementation
│   │   └── chatterbox_standalone/← vendored Chatterbox source (do not refactor)
│   ├── generate.py               ← engine-agnostic generate() API
│   └── utils/                    ← folder_paths mock, downloaders, device utils
├── scripts/                      ← input .txt / .md files (user content)
├── voices/                       ← reference voice audio files (user content)
├── Output/                       ← generated .wav files land here by default
├── models/                       ← auto-downloaded model weights (git-ignored)
├── tui.py                        ← Textual TUI entry point
└── text_to_speech.py             ← CLI entry point
```

---

## 3. Dependency Rules

- Add every new Python dependency to `requirements.txt` with a minimum
  version pin (`>=`).  Never leave silent `ImportError` paths that the user
  won't see until generation time.
- Models are fetched from HuggingFace Hub on first use and cached under
  `models/` (git-ignored).  Use `.safetensors` format where available;
  avoid the `"pt"` format string in `language_models.py` unless the repo
  genuinely does not ship `.safetensors` files.

---

## 4. TUI Conventions

- All layout heights must be explicit in `DEFAULT_CSS` (e.g. `height: 3`).
  Never rely on Textual's default `height: 1fr` for widgets inside
  `VerticalScroll` — it silently expands them and pushes other content
  off-screen.
- Guard `Select.Changed` and similar events with the `self._dom_ready` flag
  so they do not fire during `compose()` / `mount()`.

---

## 5. Commit Style

- One logical change per commit.
- Mention affected components in the subject line
  (e.g. `"Add SpeechT5 engine adapter + canonical speed/pitch mappings"`).
