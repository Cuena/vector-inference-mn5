#!/usr/bin/env python3
"""Textual TUI launcher for scripts/launch_and_tunnel.sh."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG_DIR = (
    Path(__file__).resolve().parents[1] / "vec_inf" / "config" / "marenostrum5"
)
if "VEC_INF_CONFIG_DIR" not in os.environ and DEFAULT_CONFIG_DIR.exists():
    os.environ["VEC_INF_CONFIG_DIR"] = str(DEFAULT_CONFIG_DIR)

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    Static,
)

from vec_inf.client._slurm_vars import CACHED_MODEL_CONFIG_PATH
from vec_inf.client._utils import load_config
from vec_inf.client.config import ModelConfig


@dataclass(frozen=True)
class ModelEntry:
    name: str
    config: ModelConfig


@dataclass(frozen=True)
class LaunchRequest:
    model_name: str
    launch_only: bool


class ConfirmScreen(Screen[Optional[LaunchRequest]]):
    BINDINGS = [
        ("y", "confirm", "Launch & tunnel"),
        ("l", "launch_only", "Launch-only"),
        ("n", "cancel", "Cancel"),
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, model: ModelEntry) -> None:
        super().__init__()
        self.model = model

    def compose(self) -> ComposeResult:
        summary = format_confirm_markdown(self.model.config)
        yield Container(
            Static(f"Launch {self.model.name}?", id="confirm-title"),
            Markdown(summary, id="confirm-summary"),
            Horizontal(
                Button("Launch & tunnel", id="confirm-yes", variant="success"),
                Button("Launch only", id="confirm-launch-only", variant="primary"),
                Button("Cancel", id="confirm-no", variant="error"),
                id="confirm-buttons",
            ),
            id="confirm-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-yes":
            self.dismiss(LaunchRequest(self.model.name, launch_only=False))
        elif event.button.id == "confirm-launch-only":
            self.dismiss(LaunchRequest(self.model.name, launch_only=True))
        else:
            self.dismiss(None)

    def action_confirm(self) -> None:
        self.dismiss(LaunchRequest(self.model.name, launch_only=False))

    def action_launch_only(self) -> None:
        self.dismiss(LaunchRequest(self.model.name, launch_only=True))

    def action_cancel(self) -> None:
        self.dismiss(None)


class LaunchTui(App[Optional[LaunchRequest]]):
    TITLE = "Vector Inference Launcher"
    SUB_TITLE = "Select a model, then launch (with or without tunnel)"
    COMPACT_WIDTH = 118
    COMPACT_HEIGHT = 33

    CSS = """
    Screen {
        background: $surface;
    }

    #status {
        border: round $panel;
        background: $panel 60%;
        color: $text 85%;
        padding: 0 1;
        margin: 0 1;
        height: auto;
    }

    #main {
        height: 1fr;
        margin: 0 1;
        layout: horizontal;
    }

    .panel {
        border: round $panel;
        background: $panel;
        height: 1fr;
        padding: 0;
    }

    #left-pane {
        width: 40%;
        min-width: 24;
        margin-right: 1;
    }

    #right-pane {
        width: 1fr;
    }

    #models-title,
    #details-title {
        height: 1;
        padding: 0;
        background: $primary 25%;
        color: $text;
        text-style: bold;
    }

    #filter {
        margin: 0;
        background: $surface;
        border: round $primary 40%;
    }

    #filter:focus {
        border: round $primary;
    }

    #model-list {
        margin: 0;
        height: 1fr;
        background: $surface 40%;
        border: round $panel;
    }

    #details-scroll {
        margin: 0;
        height: 1fr;
        background: $surface 35%;
        border: round $panel;
        padding: 0;
    }

    #details {
        margin: 0;
    }

    .model-row {
        height: 1;
        padding: 0;
        align: left middle;
    }

    .model-name {
        width: 1fr;
        text-overflow: ellipsis;
    }

    .badges {
        width: auto;
        height: 1;
        align: right middle;
    }

    .badge {
        border: none;
        background: $surface 60%;
        color: $text 90%;
        min-width: 7;
        text-align: center;
        padding: 0 1;
        margin-left: 0;
        height: 1;
        text-style: bold;
    }

    .badge-node {
        background: $primary 20%;
    }

    .badge-gpu {
        background: $secondary 35%;
    }

    ScrollBar {
        background: $panel;
        color: $text 30%;
    }

    ScrollBar:hover {
        color: $text 45%;
    }

    #model-list {
        scrollbar-size: 0 0;
        scrollbar-color: transparent;
        scrollbar-background: transparent;
        scrollbar-color-hover: transparent;
        scrollbar-background-hover: transparent;
        scrollbar-color-active: transparent;
        scrollbar-background-active: transparent;
    }

    #model-list ScrollBar,
    #model-list ScrollBarCorner {
        display: none;
    }

    #details-scroll {
        scrollbar-color: $text 30%;
        scrollbar-background: $panel;
        scrollbar-color-hover: $text 45%;
        scrollbar-background-hover: $panel;
        scrollbar-size: 1 1;
    }



    ListView:focus ListItem.--highlight {
        background: $primary 40%;
    }

    ListItem.--highlight {
        background: $primary 20%;
    }

    ListItem.--highlight .model-name {
        text-style: bold;
    }

    #confirm-dialog {
        border: round $panel;
        background: $panel;
        width: 1fr;
        max-width: 96;
        min-width: 40;
        max-height: 80%;
        height: auto;
        padding: 1 1;
        align: center middle;
    }

    #confirm-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #confirm-summary {
        margin-bottom: 1;
    }

    #confirm-buttons {
        height: auto;
        align: right middle;
    }

    Screen.compact #status {
        display: none;
    }

    Screen.compact Header {
        display: none;
    }

    Screen.compact Footer {
        display: none;
    }

    Screen.compact #main {
        layout: vertical;
        margin: 0;
    }

    Screen.compact .panel {
        padding: 0 0;
        border: round $panel;
    }

    Screen.compact #left-pane,
    Screen.compact #right-pane {
        width: 1fr;
        min-width: 0;
    }

    Screen.compact #left-pane {
        margin-right: 0;
        margin-bottom: 0;
    }

    Screen.compact .badges {
        display: none;
    }

    Screen.compact #filter {
        margin: 0 0 0 0;
    }

    Screen.compact #model-list,
    Screen.compact #details-scroll {
        margin: 0;
    }

    Screen.compact #confirm-dialog {
        min-width: 0;
        max-width: 1fr;
        width: 1fr;
    }
    """

    BINDINGS = [
        ("enter", "confirm", "Launch"),
        ("/", "focus_filter", "Filter"),
        ("tab", "focus_list", "List"),
        ("shift+tab", "focus_filter", "Filter"),
        ("d", "focus_details", "Details"),
        ("r", "reload", "Reload"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.theme = "dracula"
        self._models: list[ModelEntry] = []
        self._filtered: list[ModelEntry] = []
        self._last_launched_model: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(build_status_line(), id="status", markup=True)
        with Container(id="main"):
            with Vertical(id="left-pane", classes="panel"):
                yield Static("Models", id="models-title")
                yield Input(
                    placeholder="Type to filter models…  (Tab: list, Enter: launch)",
                    id="filter",
                )
                yield ListView(id="model-list")
            with Vertical(id="right-pane", classes="panel"):
                yield Static("Details", id="details-title")
                with VerticalScroll(id="details-scroll"):
                    yield Markdown("", id="details")
        yield Footer()

    def on_mount(self) -> None:
        self._sync_layout()
        self.load_models()
        self._last_launched_model = load_last_launched_model()
        self.populate_list(preferred_model=self._last_launched_model)
        list_view = self.query_one(ListView)
        if hasattr(list_view, "show_vertical_scrollbar"):
            list_view.show_vertical_scrollbar = False
        if hasattr(list_view, "show_horizontal_scrollbar"):
            list_view.show_horizontal_scrollbar = False
        list_view.focus()

    def on_resize(self, event: Resize) -> None:
        self._sync_layout()

    def _sync_layout(self) -> None:
        size = self.size
        compact = size.width < self.COMPACT_WIDTH or size.height < self.COMPACT_HEIGHT
        self.screen.set_class(compact, "compact")

    def load_models(self) -> None:
        configs = load_config()
        entries = [ModelEntry(config.model_name, config) for config in configs]
        self._models = sorted(entries, key=lambda item: item.name.lower())

    def populate_list(self, preferred_model: str | None = None) -> None:
        filter_value = self.query_one(Input).value.strip().lower()
        self._filtered = [
            model for model in self._models if filter_value in model.name.lower()
        ]
        list_view = self.query_one(ListView)
        list_view.clear()
        for model in self._filtered:
            list_view.append(self._make_model_item(model))

        if self._filtered:
            # Ensure there is always a valid selection.
            selected_index = 0
            if preferred_model:
                for index, model in enumerate(self._filtered):
                    if model.name == preferred_model:
                        selected_index = index
                        break
            list_view.index = selected_index
            self.update_details(self._filtered[selected_index])
        else:
            list_view.index = None
            self.query_one("#details", Markdown).update(
                "_No models match the current filter._"
            )

    def _make_model_item(self, model: ModelEntry) -> ListItem:
        node_count = model.config.num_nodes
        gpu_count = model.config.gpus_per_node
        node_label = f"{node_count} node" if node_count == 1 else f"{node_count} nodes"
        gpu_label = f"{gpu_count} GPU" if gpu_count == 1 else f"{gpu_count} GPUs"
        row = Horizontal(
            Label(model.name, classes="model-name"),
            Horizontal(
                Static(node_label, classes="badge badge-node"),
                Static(gpu_label, classes="badge badge-gpu"),
                classes="badges",
            ),
            classes="model-row",
        )
        return ListItem(row)

    def update_details(self, model: ModelEntry) -> None:
        details = format_details(model.config)
        self.query_one("#details", Markdown).update(details)

    def get_selected_model(self) -> Optional[ModelEntry]:
        list_view = self.query_one(ListView)
        if list_view.index is None:
            return None
        if list_view.index >= len(self._filtered):
            return None
        return self._filtered[list_view.index]

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter":
            self.populate_list()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "filter":
            self.action_confirm()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        model = self.get_selected_model()
        if model:
            self.update_details(model)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.action_confirm()

    def action_focus_filter(self) -> None:
        self.query_one(Input).focus()

    def action_focus_list(self) -> None:
        list_view = self.query_one(ListView)
        if self._filtered:
            if list_view.index is None or list_view.index >= len(self._filtered):
                list_view.index = 0
                self.update_details(self._filtered[0])
            list_view.focus()
        else:
            self.query_one(Input).focus()

    def action_focus_details(self) -> None:
        self.query_one("#details-scroll", VerticalScroll).focus()

    def action_reload(self) -> None:
        self.load_models()
        self.populate_list(preferred_model=self._last_launched_model)

    def action_confirm(self) -> None:
        model = self.get_selected_model()
        if not model:
            self.notify("Select a model to launch.", severity="warning")
            return
        self.push_screen(ConfirmScreen(model), self._handle_confirm)

    def _handle_confirm(self, request: LaunchRequest | None) -> None:
        if request:
            save_last_launched_model(request.model_name)
            self.exit(request)


def build_status_line() -> str:
    config_hint = resolve_model_config_path()
    env_path = Path(__file__).resolve().parent / ".launch.env"
    env_status = "[green]found[/green]" if env_path.exists() else "[red]missing[/red]"
    return (
        f"[b]Config[/b]: {config_hint}  |  "
        f"[b]scripts/.launch.env[/b]: {env_status}  |  "
        "[b]Keys[/b]: / filter, Tab list, Enter launch, q quit"
    )


def load_last_launched_model() -> str | None:
    path = _tui_state_path()
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    return value or None


def save_last_launched_model(model_name: str) -> None:
    path = _tui_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model_name, encoding="utf-8")
    except OSError:
        return


def _tui_state_path() -> Path:
    state_dir = os.getenv("XDG_STATE_HOME")
    if state_dir:
        base_dir = Path(state_dir)
    else:
        base_dir = Path.home() / ".local" / "state"
    return base_dir / "vector-inference" / "last_tui_model.txt"


def resolve_model_config_path() -> str:
    env_config = os.getenv("VEC_INF_MODEL_CONFIG")
    if env_config:
        return env_config
    env_dir = os.getenv("VEC_INF_CONFIG_DIR")
    if env_dir:
        return str(Path(env_dir) / "models.yaml")
    if CACHED_MODEL_CONFIG_PATH.exists():
        return str(CACHED_MODEL_CONFIG_PATH)
    default_path = DEFAULT_CONFIG_DIR / "models.yaml"
    if not default_path.exists():
        default_path = (
            Path(__file__).resolve().parents[1] / "vec_inf" / "config" / "models.yaml"
        )
    return str(default_path)


def format_details(config: ModelConfig) -> str:
    lines: list[str] = []
    lines.append(f"## `{config.model_name}`")
    lines.append("")

    lines.append("### Model")
    lines.append(f"**Type**: `{config.model_type}`")
    lines.append(f"**Family**: `{config.model_family}`")
    if config.model_variant:
        lines.append(f"**Variant**: `{config.model_variant}`")
    if config.engine:
        lines.append(f"**Engine**: `{config.engine}`")

    lines.append("")
    lines.append("### Job")
    lines.append(
        f"**Resources**: `{config.num_nodes} node(s)`, `{config.gpus_per_node} GPU(s)/node`, `{config.cpus_per_task} CPU(s)/task`"
    )
    if config.mem_per_node:
        lines.append(f"**Memory per node**: `{config.mem_per_node}`")
    if config.partition:
        lines.append(f"**Partition**: `{config.partition}`")
    if config.qos:
        lines.append(f"**QOS**: `{config.qos}`")
    lines.append(f"**Time**: `{config.time}`")

    lines.append("")
    lines.append("### Paths")
    lines.append("```text")
    lines.append(f"Venv: {config.venv}")
    lines.append(f"Log dir: {config.log_dir}")
    lines.append(f"Weights dir: {config.model_weights_parent_dir}")
    if config.work_dir:
        lines.append(f"Work dir: {config.work_dir}")
    lines.append("```")

    if config.vllm_args:
        lines.append("")
        lines.append("### vLLM args")
        lines.append("```yaml")
        lines.extend(_yaml_lines(config.vllm_args))
        lines.append("```")

    if config.sglang_args:
        lines.append("")
        lines.append("### SGLang args")
        lines.append("```yaml")
        lines.extend(_yaml_lines(config.sglang_args))
        lines.append("```")

    if config.env:
        lines.append("")
        lines.append("### Environment")
        lines.append("```yaml")
        lines.extend(_yaml_lines(config.env))
        lines.append("```")

    return "\n".join(lines)


def format_confirm_markdown(config: ModelConfig) -> str:
    lines: list[str] = []
    lines.append("### Summary")
    lines.append(f"- **Type**: `{config.model_type}`")
    if config.engine:
        lines.append(f"- **Engine**: `{config.engine}`")
    lines.append(
        f"- **Resources**: `{config.num_nodes} node(s)`, `{config.gpus_per_node} GPU(s)/node`"
    )
    if config.partition or config.qos:
        parts: list[str] = []
        if config.partition:
            parts.append(str(config.partition))
        if config.qos:
            parts.append(str(config.qos))
        lines.append(f"- **Slurm**: `{', '.join(parts)}`")
    lines.append(f"- **Time**: `{config.time}`")
    lines.append("")
    lines.append("Press **Launch** to continue, or **Cancel**.")
    return "\n".join(lines)


def _yaml_lines(values: dict[str, object]) -> list[str]:
    lines: list[str] = []
    for key in sorted(values.keys()):
        value = values[key]
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for inner_key in sorted(value.keys()):
                lines.append(f"  {inner_key}: {_yaml_scalar(value[inner_key])}")
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    return lines


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return "[" + ", ".join(_yaml_scalar(item) for item in value) + "]"
    return str(value)


def run_launch(request: LaunchRequest) -> None:
    script_path = Path(__file__).resolve().parent / "launch_and_tunnel.sh"
    if not script_path.exists():
        raise SystemExit(f"Launch script not found: {script_path}")
    if request.launch_only:
        os.execvp("bash", ["bash", str(script_path), "--launch-only", request.model_name])
    else:
        os.execvp("bash", ["bash", str(script_path), request.model_name])


def main() -> None:
    app = LaunchTui()
    result = app.run()
    if isinstance(result, LaunchRequest):
        run_launch(result)


if __name__ == "__main__":
    main()
