#!/usr/bin/env python3
"""
Read4me — TUI

A Textual-based interface to generate speech from text in a target voice.

Usage:
    python tui.py
"""

import os
import sys
import threading
from pathlib import Path

# Ensure repo root is on path
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
_VOICES_DIR = os.path.join(_REPO_ROOT, "voices")

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Slider,
    Static,
)


# ---------------------------------------------------------------------------
# File-picker modal
# ---------------------------------------------------------------------------

class FilePicker(ModalScreen):
    """A modal that lets the user browse and select a file."""

    BINDINGS = [
        Binding("escape", "dismiss(None)", "Cancel"),
    ]

    DEFAULT_CSS = """
    FilePicker {
        align: center middle;
    }
    FilePicker > Vertical {
        width: 70;
        height: 30;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    FilePicker Label {
        margin-bottom: 1;
        text-style: bold;
    }
    FilePicker DirectoryTree {
        height: 20;
        border: solid $panel;
    }
    FilePicker Horizontal {
        height: 3;
        margin-top: 1;
        align: right middle;
    }
    """

    def __init__(self, start_path: str, title: str = "Select file") -> None:
        super().__init__()
        self._start_path = start_path
        self._title = title
        self._selected: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title)
            yield DirectoryTree(self._start_path, id="picker-tree")
            with Horizontal():
                yield Button("Cancel", variant="default", id="picker-cancel")
                yield Button("Select", variant="primary", id="picker-select")

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self._selected = str(event.path)
        self.query_one("#picker-select", Button).label = Path(self._selected).name

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "picker-cancel":
            self.dismiss(None)
        elif event.button.id == "picker-select":
            self.dismiss(self._selected)


# ---------------------------------------------------------------------------
# Directory-picker modal
# ---------------------------------------------------------------------------

class DirPicker(ModalScreen):
    """A modal that lets the user browse and select a directory."""

    BINDINGS = [
        Binding("escape", "dismiss(None)", "Cancel"),
    ]

    DEFAULT_CSS = """
    DirPicker {
        align: center middle;
    }
    DirPicker > Vertical {
        width: 70;
        height: 30;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    DirPicker Label {
        margin-bottom: 1;
        text-style: bold;
    }
    DirPicker DirectoryTree {
        height: 20;
        border: solid $panel;
    }
    DirPicker Horizontal {
        height: 3;
        margin-top: 1;
        align: right middle;
    }
    """

    def __init__(self, start_path: str, title: str = "Select directory") -> None:
        super().__init__()
        self._start_path = start_path
        self._title = title
        self._current_dir: str = start_path

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title)
            yield DirectoryTree(self._start_path, id="dir-tree")
            with Horizontal():
                yield Button("Cancel", variant="default", id="dir-cancel")
                yield Button("Select this folder", variant="primary", id="dir-select")

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        self._current_dir = str(event.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dir-cancel":
            self.dismiss(None)
        elif event.button.id == "dir-select":
            self.dismiss(self._current_dir)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

LANGUAGES = [
    ("English", "English"),
    ("German", "German"),
    ("French", "French"),
    ("Norwegian", "Norwegian"),
    ("Italian", "Italian"),
    ("Spanish", "Spanish"),
]

DEVICES = [
    ("auto", "auto"),
    ("cuda", "cuda"),
    ("mps", "mps"),
    ("cpu", "cpu"),
]


class Read4meApp(App):
    """Read4me — Text to Speech TUI"""

    TITLE = "Read4me"
    SUB_TITLE = "Text to speech in your target voice"

    BINDINGS = [
        Binding("ctrl+g", "generate", "Generate", priority=True),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    DEFAULT_CSS = """
    Read4meApp {
        background: $background;
    }
    .section-label {
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }
    .path-display {
        background: $panel;
        border: solid $primary-darken-2;
        padding: 0 1;
        height: 3;
        content-align: left middle;
    }
    .path-display.unset {
        color: $text-muted;
    }
    .row {
        height: auto;
        margin-bottom: 1;
    }
    .browse-btn {
        width: 12;
        margin-left: 1;
    }
    .slider-row {
        height: 5;
        margin-bottom: 1;
    }
    .slider-label {
        width: 20;
        content-align: left middle;
    }
    .slider-value {
        width: 6;
        content-align: right middle;
    }
    #main-scroll {
        height: 1fr;
        padding: 1 2;
    }
    #generate-btn {
        margin-top: 1;
        width: 100%;
        height: 3;
    }
    #status {
        height: 3;
        margin-top: 1;
        background: $panel;
        border: solid $panel-darken-1;
        padding: 0 1;
        content-align: left middle;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._script_path: str | None = None
        self._voice_path: str | None = None
        self._output_dir: str = _REPO_ROOT
        self._generating: bool = False

    # ---- Layout ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer(id="main-scroll"):
            # Script file
            yield Label("1. Script file", classes="section-label")
            with Horizontal(classes="row"):
                yield Static(
                    "No file selected — browse scripts/",
                    id="script-display",
                    classes="path-display unset",
                )
                yield Button("Browse", id="browse-script", classes="browse-btn")

            # Voice file
            yield Label("2. Voice file", classes="section-label")
            with Horizontal(classes="row"):
                yield Static(
                    "No file selected — browse voices/",
                    id="voice-display",
                    classes="path-display unset",
                )
                yield Button("Browse", id="browse-voice", classes="browse-btn")

            # Settings
            yield Label("3. Settings  (optional)", classes="section-label")
            with Horizontal(classes="row"):
                yield Label("Language", classes="slider-label")
                yield Select(LANGUAGES, value="English", id="language-select")
            with Horizontal(classes="row"):
                yield Label("Device", classes="slider-label")
                yield Select(DEVICES, value="auto", id="device-select")

            with Horizontal(classes="slider-row"):
                yield Label("Exaggeration", classes="slider-label")
                yield Slider(
                    min=0.0, max=1.5, step=0.05, value=0.5, id="exaggeration-slider"
                )
                yield Static("0.50", id="exaggeration-val", classes="slider-value")

            with Horizontal(classes="slider-row"):
                yield Label("Temperature", classes="slider-label")
                yield Slider(
                    min=0.1, max=1.5, step=0.05, value=0.8, id="temperature-slider"
                )
                yield Static("0.80", id="temperature-val", classes="slider-value")

            with Horizontal(classes="slider-row"):
                yield Label("CFG Weight", classes="slider-label")
                yield Slider(
                    min=0.0, max=1.0, step=0.05, value=0.5, id="cfg-slider"
                )
                yield Static("0.50", id="cfg-val", classes="slider-value")

            # Output directory
            yield Label("4. Output directory", classes="section-label")
            with Horizontal(classes="row"):
                yield Static(
                    _REPO_ROOT, id="outdir-display", classes="path-display"
                )
                yield Button("Browse", id="browse-outdir", classes="browse-btn")

            # Generate
            yield Label("5. Generate", classes="section-label")
            yield Button(
                "Generate  [Ctrl+G]",
                id="generate-btn",
                variant="success",
            )
            yield Static("Ready.", id="status")

        yield Footer()

    # ---- Slider sync -----------------------------------------------------

    def on_slider_changed(self, event: Slider.Changed) -> None:
        val = f"{event.value:.2f}"
        mapping = {
            "exaggeration-slider": "exaggeration-val",
            "temperature-slider":  "temperature-val",
            "cfg-slider":          "cfg-val",
        }
        if event.slider.id in mapping:
            self.query_one(f"#{mapping[event.slider.id]}", Static).update(val)

    # ---- File/dir pickers ------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id

        if btn_id == "browse-script":
            self.push_screen(
                FilePicker(
                    start_path=_SCRIPTS_DIR if os.path.isdir(_SCRIPTS_DIR) else _REPO_ROOT,
                    title="Select script file (.txt / .md)",
                ),
                callback=self._on_script_selected,
            )

        elif btn_id == "browse-voice":
            self.push_screen(
                FilePicker(
                    start_path=_VOICES_DIR if os.path.isdir(_VOICES_DIR) else _REPO_ROOT,
                    title="Select voice file (.wav / .mp3)",
                ),
                callback=self._on_voice_selected,
            )

        elif btn_id == "browse-outdir":
            self.push_screen(
                DirPicker(
                    start_path=self._output_dir,
                    title="Select output directory",
                ),
                callback=self._on_outdir_selected,
            )

        elif btn_id == "generate-btn":
            self.action_generate()

    def _on_script_selected(self, path: str | None) -> None:
        if path:
            self._script_path = path
            w = self.query_one("#script-display", Static)
            w.update(path)
            w.remove_class("unset")

    def _on_voice_selected(self, path: str | None) -> None:
        if path:
            self._voice_path = path
            w = self.query_one("#voice-display", Static)
            w.update(path)
            w.remove_class("unset")

    def _on_outdir_selected(self, path: str | None) -> None:
        if path:
            self._output_dir = path
            self.query_one("#outdir-display", Static).update(path)

    # ---- Generation ------------------------------------------------------

    def action_generate(self) -> None:
        if self._generating:
            self._set_status("Already generating — please wait...")
            return

        if not self._script_path:
            self._set_status("Please select a script file first.")
            return
        if not self._voice_path:
            self._set_status("Please select a voice file first.")
            return

        # Read text from the script file
        try:
            with open(self._script_path, encoding="utf-8") as f:
                text = f.read().strip()
        except OSError as exc:
            self._set_status(f"Could not read script: {exc}")
            return

        if not text:
            self._set_status("Script file is empty.")
            return

        # Collect settings
        language   = self.query_one("#language-select", Select).value
        device     = self.query_one("#device-select", Select).value
        exaggeration = float(self.query_one("#exaggeration-slider", Slider).value)
        temperature  = float(self.query_one("#temperature-slider", Slider).value)
        cfg_weight   = float(self.query_one("#cfg-slider", Slider).value)

        voice_path = self._voice_path
        output_dir = self._output_dir

        self._generating = True
        self._set_status("Generating… this may take a while.")
        self.query_one("#generate-btn", Button).disabled = True

        # Run in background thread so the TUI stays responsive
        def _run():
            try:
                from src.generate import generate
                out_path = generate(
                    text=text,
                    voice_path=voice_path,
                    output_dir=output_dir,
                    language=language,
                    device=device,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                )
                self.call_from_thread(self._on_generation_done, out_path, None)
            except Exception as exc:
                self.call_from_thread(self._on_generation_done, None, str(exc))

        threading.Thread(target=_run, daemon=True).start()

    def _on_generation_done(self, out_path: str | None, error: str | None) -> None:
        self._generating = False
        self.query_one("#generate-btn", Button).disabled = False
        if error:
            self._set_status(f"Error: {error}")
        else:
            self._set_status(f"Done!  Saved to: {out_path}")

    def _set_status(self, msg: str) -> None:
        self.query_one("#status", Static).update(msg)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    Read4meApp().run()
