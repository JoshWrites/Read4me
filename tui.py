#!/usr/bin/env python3
"""
Read4me — TUI

Two-panel interface:
  LEFT  — script, voice, engine, device, output directory, generate
  RIGHT — engine-specific settings (dynamic, ◆ marks required fields)

Usage:
    python tui.py
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

# Prevent HuggingFace tokenizer and OpenMP from forking worker processes.
# When running inside Textual's event loop those forks inherit the terminal
# file descriptors and fail with "bad value(s) in fds_to_keep" on Python 3.12+.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_REPO_ROOT, "src")
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
_VOICES_DIR = os.path.join(_REPO_ROOT, "voices")
_OUTPUT_DIR = os.path.join(_REPO_ROOT, "Output")

for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Footer, Header, Input, Label, Select, Static


# ── File picker modal ─────────────────────────────────────────────────────────

class FilePicker(ModalScreen):
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    DEFAULT_CSS = """
    FilePicker { align: center middle; }
    FilePicker > Vertical {
        width: 72; height: 32;
        border: thick $primary; background: $surface; padding: 1 2;
    }
    FilePicker .modal-title { text-style: bold; margin-bottom: 1; }
    FilePicker DirectoryTree { height: 22; border: solid $panel; }
    FilePicker .modal-actions { height: 3; margin-top: 1; align: right middle; }
    FilePicker Button { margin-left: 1; }
    """

    def __init__(self, start_path: str, title: str = "Select file") -> None:
        super().__init__()
        self._start = start_path
        self._title = title
        self._selected: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title, classes="modal-title")
            yield DirectoryTree(self._start, id="fp-tree")
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel",  variant="default", id="fp-cancel")
                yield Button("Select",  variant="primary", id="fp-select")

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self._selected = str(event.path)
        self.query_one("#fp-select", Button).label = f"✔  {Path(self._selected).name}"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(self._selected if event.button.id == "fp-select" else None)


# ── Directory picker modal ────────────────────────────────────────────────────

class DirPicker(ModalScreen):
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    DEFAULT_CSS = """
    DirPicker { align: center middle; }
    DirPicker > Vertical {
        width: 72; height: 32;
        border: thick $accent; background: $surface; padding: 1 2;
    }
    DirPicker .modal-title { text-style: bold; margin-bottom: 1; }
    DirPicker DirectoryTree { height: 22; border: solid $panel; }
    DirPicker .modal-actions { height: 3; margin-top: 1; align: right middle; }
    DirPicker Button { margin-left: 1; }
    """

    def __init__(self, start_path: str, title: str = "Select directory") -> None:
        super().__init__()
        self._start = start_path
        self._current = start_path
        self._title = title

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title, classes="modal-title")
            yield DirectoryTree(self._start, id="dp-tree")
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel",           variant="default", id="dp-cancel")
                yield Button("Use this folder",  variant="primary",  id="dp-select")

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self._current = str(event.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(self._current if event.button.id == "dp-select" else None)


# ── Main app ──────────────────────────────────────────────────────────────────

_DEVICES: list[tuple[str, str]] = [
    ("Auto",  "auto"),
    ("CUDA",  "cuda"),
    ("MPS",   "mps"),
    ("CPU",   "cpu"),
]


class Read4meApp(App):
    TITLE = "Read4me"
    SUB_TITLE = "Text to Speech"

    BINDINGS = [
        Binding("ctrl+g", "generate", "Generate", priority=True),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    DEFAULT_CSS = """
    /* Every widget stacks naturally; nothing expands to fill space */
    #main-scroll { height: 1fr; padding: 0 3; }

    .section-label { text-style: bold; color: $accent; height: 1; margin-top: 1; }

    /* File rows: fixed height, browse button flush right */
    .file-row    { height: 3; margin-bottom: 1; }
    .path-display {
        width: 1fr; height: 3;
        background: $panel; border: solid $primary-darken-2;
        padding: 0 1; content-align: left middle; overflow: hidden;
    }
    .path-display.unset { color: $text-muted; }
    .browse-btn { width: 10; height: 3; margin-left: 1; }

    /* Engine + Device row: each column fixed height */
    .eng-dev-row { height: 6; margin-bottom: 1; }
    .eng-dev-col { width: 1fr; height: 6; }

    /* Generate button and status */
    #generate-btn { width: 100%; height: 3; margin-top: 1; }
    #status {
        height: 3; margin-top: 1;
        background: $panel; border: solid $panel-darken-1;
        padding: 0 1; content-align: left middle; color: $text-muted;
    }
    #status.busy  { color: $warning; }
    #status.done  { color: $success; }
    #status.error { color: $error; }

    /* Section divider */
    .divider { height: 1; margin-top: 1; color: $panel-darken-2; }

    /* Engine param panel */
    #bottom-panel  { height: auto; }
    .engine-title  { text-style: bold; color: $accent; height: 1; margin-top: 1; }
    .required-header { color: $error; text-style: bold; height: 1; margin-top: 1; }
    .optional-header { color: $text-muted; text-style: bold; height: 1; margin-top: 1; }
    .param-label   { height: 1; margin-top: 1; }
    .param-label.required { color: $error; text-style: bold; }
    .param-desc    { height: 1; color: $text-muted; }
    .param-input   { height: 3; }
    .param-select  { height: 3; }
    .no-params     { height: 1; color: $text-muted; margin-top: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._script_path: str | None = None
        self._voice_path:  str | None = None
        self._output_dir:  str = _OUTPUT_DIR
        self._generating:  bool = False
        self._current_engine: str | None = None
        self._dom_ready: bool = False  # True only after on_mount — guards Select.Changed

    # ── Layout ───────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        from engines.registry import available_engines
        engines = available_engines()
        default_engine = engines[0][1] if engines else ""

        yield Header()

        # Everything is a flat direct child of VerticalScroll.
        # No intermediate wrappers — avoids height-collapse bugs in Textual 8.
        with VerticalScroll(id="main-scroll"):

            # ── Script ────────────────────────────────────────────────────
            yield Label("Script", classes="section-label")
            with Horizontal(classes="file-row"):
                yield Static(
                    "browse scripts/  →",
                    id="script-display", classes="path-display unset",
                )
                yield Button("Browse", id="browse-script", classes="browse-btn")

            # ── Voice (each element tagged so show/hide needs no wrapper) ───
            yield Label("Voice", classes="section-label voice-row")
            with Horizontal(classes="file-row voice-row"):
                yield Static(
                    "browse voices/  →",
                    id="voice-display", classes="path-display unset",
                )
                yield Button("Browse", id="browse-voice", classes="browse-btn")

            # ── Engine + Device ───────────────────────────────────────────
            with Horizontal(classes="eng-dev-row"):
                with Vertical(classes="eng-dev-col"):
                    yield Label("Engine", classes="section-label")
                    yield Select(engines, value=default_engine, id="engine-select")
                with Vertical(classes="eng-dev-col"):
                    yield Label("Device", classes="section-label")
                    yield Select(_DEVICES, value="auto", id="device-select")

            # ── Output Directory ──────────────────────────────────────────
            yield Label("Output Directory", classes="section-label")
            with Horizontal(classes="file-row"):
                yield Static(
                    _OUTPUT_DIR, id="outdir-display", classes="path-display",
                )
                yield Button("Browse", id="browse-outdir", classes="browse-btn")

            # ── Generate + Status ─────────────────────────────────────────
            yield Button("⚡  Generate  [Ctrl+G]", id="generate-btn", variant="success")
            yield Static("Ready.", id="status")

            # ── Divider ───────────────────────────────────────────────────
            yield Static("", classes="divider")

            # ── Engine params (populated in on_mount) ─────────────────────
            with Vertical(id="bottom-panel"):
                pass

        yield Footer()

    def on_mount(self) -> None:
        from engines.registry import available_engines
        engines = available_engines()
        if engines:
            self._refresh_engine_panel(engines[0][1])
        self._dom_ready = True  # allow on_select_changed to act from here on

    # ── Engine panel ─────────────────────────────────────────────────────────

    def _refresh_engine_panel(self, engine_name: str) -> None:
        self._current_engine = engine_name

        from engines.registry import get_engine
        engine = get_engine(engine_name)

        panel = self.query_one("#bottom-panel")
        try:
            panel.remove_children()
        except AttributeError:
            for child in list(panel.children):
                child.remove()

        # Show/hide voice rows depending on whether this engine needs one
        for w in self.query(".voice-row"):
            w.display = engine.requires_voice_file

        for widget in self._build_param_widgets(engine):
            panel.mount(widget)

    def _build_param_widgets(self, engine) -> list:
        widgets: list = []
        all_params = engine.params()

        widgets.append(Label(engine.display_name, classes="engine-title"))

        if not all_params:
            widgets.append(Static("No configurable parameters.", classes="no-params"))
            return widgets

        required = [p for p in all_params if p.required]
        optional = [p for p in all_params if not p.required]

        if required:
            widgets.append(Label("◆  Required", classes="required-header"))
            for param in required:
                widgets.extend(self._build_param_row(param))

        if optional:
            widgets.append(Label("Settings", classes="optional-header"))
            for param in optional:
                widgets.extend(self._build_param_row(param))

        return widgets

    def _build_param_row(self, param) -> list:
        from engines.canonical import resolve as resolve_canonical
        widgets: list = []
        widget_id = f"param-{param.id}"

        # Resolve canonical overrides for label / description / range
        canon = resolve_canonical(param.canonical)
        display_label = canon.label if canon else param.label
        display_desc  = canon.description if canon else param.description
        min_val = canon.min_val if (canon and canon.min_val is not None) else param.min_val
        max_val = canon.max_val if (canon and canon.max_val is not None) else param.max_val

        # Label — ◆ prefix for required
        prefix = "◆  " if param.required else ""
        label_classes = "param-label" + (" required" if param.required else "")
        widgets.append(Label(f"{prefix}{display_label}", classes=label_classes))

        # Input widget
        if param.type == "select" and param.options:
            widgets.append(
                Select(
                    options=param.options,
                    value=param.default,
                    id=widget_id,
                    classes="param-select",
                )
            )
        else:
            range_hint = (
                f"  ({min_val} – {max_val})"
                if min_val is not None and max_val is not None
                else ""
            )
            placeholder = f"required{range_hint}" if param.required else range_hint.strip()
            default_val  = "" if param.required else str(param.default)
            widgets.append(
                Input(
                    value=default_val,
                    placeholder=placeholder,
                    id=widget_id,
                    classes="param-input",
                )
            )

        # Description hint
        if display_desc:
            widgets.append(Static(display_desc, classes="param-desc"))

        return widgets

    # ── Collect param values from right panel ─────────────────────────────────

    def _collect_engine_params(self, engine_name: str) -> dict:
        from engines.registry import get_engine
        engine = get_engine(engine_name)
        result: dict = {}

        for param in engine.params():
            widget_id = f"param-{param.id}"
            try:
                if param.type == "select":
                    val = self.query_one(f"#{widget_id}", Select).value
                    result[param.id] = (
                        str(val) if val != Select.BLANK else param.default
                    )
                else:
                    raw = self.query_one(f"#{widget_id}", Input).value.strip()
                    if raw:
                        if param.type == "float":
                            result[param.id] = float(raw)
                        elif param.type == "int":
                            result[param.id] = int(raw)
                        else:
                            result[param.id] = raw
                    elif param.required:
                        raise ValueError(f"'{param.label}' is required but not set.")
                    else:
                        result[param.id] = param.default
            except (ValueError, TypeError) as exc:
                if param.required:
                    raise
                result[param.id] = param.default

        return result

    # ── Event handlers ────────────────────────────────────────────────────────

    def on_select_changed(self, event: Select.Changed) -> None:
        # Ignore events that fire during compose/mount before DOM is ready.
        # on_mount handles the initial render; we only act on genuine user changes.
        if not self._dom_ready:
            return
        if event.select.id == "engine-select" and event.value != Select.BLANK:
            new_engine = str(event.value)
            if new_engine != self._current_engine:
                self._refresh_engine_panel(new_engine)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "browse-script":
            self.push_screen(
                FilePicker(
                    start_path=_SCRIPTS_DIR if os.path.isdir(_SCRIPTS_DIR) else _REPO_ROOT,
                    title="Select script file  (.txt / .md)",
                ),
                callback=self._on_script_selected,
            )
        elif bid == "browse-voice":
            self.push_screen(
                FilePicker(
                    start_path=_VOICES_DIR if os.path.isdir(_VOICES_DIR) else _REPO_ROOT,
                    title="Select voice file  (.wav / .mp3)",
                ),
                callback=self._on_voice_selected,
            )
        elif bid == "browse-outdir":
            self.push_screen(
                DirPicker(start_path=self._output_dir, title="Select output directory"),
                callback=self._on_outdir_selected,
            )
        elif bid == "generate-btn":
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

    # ── Generation ────────────────────────────────────────────────────────────

    def action_generate(self) -> None:
        if self._generating:
            self._set_status("Already generating — please wait.", kind="busy")
            return

        if not self._script_path:
            self._set_status("Select a script file first.", kind="error")
            return

        engine_val = self.query_one("#engine-select", Select).value
        if engine_val == Select.BLANK:
            self._set_status("Select an engine.", kind="error")
            return
        engine_name = str(engine_val)

        from engines.registry import get_engine
        engine = get_engine(engine_name)

        if engine.requires_voice_file and not self._voice_path:
            self._set_status("Select a voice file first.", kind="error")
            return

        try:
            with open(self._script_path, encoding="utf-8") as fh:
                text = fh.read().strip()
        except OSError as exc:
            self._set_status(f"Could not read script: {exc}", kind="error")
            return

        if not text:
            self._set_status("Script file is empty.", kind="error")
            return

        try:
            engine_params = self._collect_engine_params(engine_name)
        except ValueError as exc:
            self._set_status(str(exc), kind="error")
            return

        device_val = self.query_one("#device-select", Select).value
        device = str(device_val) if device_val != Select.BLANK else "auto"

        # Snapshot mutable state before handing off to the thread
        voice_path = self._voice_path
        output_dir = self._output_dir

        self._generating = True
        self._set_status("Generating…  this may take a while.", kind="busy")
        self.query_one("#generate-btn", Button).disabled = True

        def _run() -> None:
            # Ensure subprocess-forking libraries stay single-threaded inside
            # the Textual event loop (belt-and-suspenders alongside the module-
            # level os.environ.setdefault calls at the top of this file).
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["OMP_NUM_THREADS"] = "1"
            try:
                from src.generate import generate
                out = generate(
                    text=text,
                    voice_path=voice_path,
                    output_dir=output_dir,
                    engine_name=engine_name,
                    device=device,
                    **engine_params,
                )
                self.call_from_thread(self._on_generation_done, out, None)
            except ValueError as exc:
                # "bad value(s) in fds_to_keep" and similar process-spawn errors
                if "fds_to_keep" in str(exc):
                    msg = (
                        "Subprocess fork conflict — model loaded but generation "
                        "failed.  Try running: python text_to_speech.py instead."
                    )
                else:
                    msg = str(exc)
                self.call_from_thread(self._on_generation_done, None, msg)
            except Exception as exc:  # noqa: BLE001
                self.call_from_thread(self._on_generation_done, None, str(exc))

        threading.Thread(target=_run, daemon=True).start()

    def _on_generation_done(self, out_path: str | None, error: str | None) -> None:
        self._generating = False
        self.query_one("#generate-btn", Button).disabled = False
        if error:
            self._set_status(f"Error: {error}", kind="error")
        else:
            self._set_status(f"Done →  {out_path}", kind="done")

    def _set_status(self, msg: str, kind: str = "") -> None:
        w = self.query_one("#status", Static)
        w.update(msg)
        for cls in ("busy", "done", "error"):
            w.remove_class(cls)
        if kind:
            w.add_class(kind)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Read4meApp().run()
