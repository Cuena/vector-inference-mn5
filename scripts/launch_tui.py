#!/usr/bin/env python3
"""Textual TUI launcher for scripts/launch_and_tunnel.sh."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping, Optional

from rich.console import Group
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
LAUNCH_ENV_PATH = SCRIPT_DIR / ".launch.env"
DEFAULT_MODEL_NAME = "Llama-3.2-3B-Instruct"

DEFAULT_CONFIG_DIR = (
    REPO_ROOT / "vec_inf" / "config" / "marenostrum5"
)


def parse_launch_env(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE launch env file."""
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value == "":
            values[key] = ""
            continue
        try:
            tokens = shlex.split(raw_value, posix=True)
        except ValueError:
            values[key] = raw_value.strip("\"'")
            continue
        values[key] = tokens[0] if len(tokens) == 1 else " ".join(tokens)
    return values


def derive_project_root(values: MutableMapping[str, str]) -> str:
    """Infer the remote project root from launch settings."""
    explicit = values.get("VEC_INF_PROJECT_ROOT", "").strip()
    if explicit:
        return explicit
    config_dir_remote = values.get("VEC_INF_CONFIG_DIR_REMOTE", "").strip()
    suffix = "/vec_inf/config/marenostrum5"
    if config_dir_remote.endswith(suffix):
        return config_dir_remote[: -len(suffix)]
    return values.get("RSYNC_DEST", "").strip()


def apply_launch_env_defaults(
    values: dict[str, str],
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Apply launch env values as defaults without overriding explicit shell env."""
    target = environ if environ is not None else os.environ
    for key, value in values.items():
        target.setdefault(key, value)
    project_root = derive_project_root(target)
    if project_root:
        target.setdefault("VEC_INF_PROJECT_ROOT", project_root)


def resolve_preferred_model_name(last_launched_model: str | None) -> str:
    """Pick the preferred default model for initial TUI selection."""
    if last_launched_model:
        return last_launched_model
    return DEFAULT_MODEL_NAME


_launch_env_values = parse_launch_env(LAUNCH_ENV_PATH)

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.screen import Screen
from textual.widgets import (
    Button,
    Input,
    Label,
    ListItem,
    ListView,
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


@dataclass(frozen=True)
class EffectiveLaunchSettings:
    remote_launch_host: str | None
    config_dir_remote: str | None
    rsync_dest: str | None
    vec_inf_env: str | None
    remote_work_dir: str | None
    remote_account: str | None
    remote_qos: str | None


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
        summary = format_confirm_summary(self.model.config)
        yield Container(
            Static(f"Launch {self.model.name}?", id="confirm-title"),
            Static(summary, id="confirm-summary"),
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
    COMPACT_WIDTH = 94
    COMPACT_HEIGHT = 0

    CSS = """
    Screen {
        background: $background;
    }

    #topbar {
        margin: 0 1 1 1;
        padding: 0 1;
        height: auto;
        border: round $panel;
        background: $surface;
    }

    #topbar-status {
        color: $text 75%;
        padding: 0;
    }

    #main {
        height: 1fr;
        margin: 0 1 1 1;
        layout: horizontal;
    }

    .panel {
        border: none;
        background: $surface 90%;
        height: 1fr;
        padding: 0 1 1 1;
    }

    #left-pane {
        width: 34%;
        min-width: 30;
        max-width: 42;
        margin-right: 1;
    }

    #right-pane {
        width: 1fr;
    }

    #filter {
        margin: 0 0 1 0;
        background: $background 20%;
        border: round $panel;
    }

    #filter:focus {
        border: round $primary;
    }

    #model-list {
        margin: 0;
        height: 1fr;
        background: $background 15%;
        border: none;
    }

    #details-scroll {
        margin: 0;
        height: 1fr;
        background: $background 15%;
        border: none;
        padding: 0;
    }

    #details {
        margin: 0;
        padding: 1;
    }

    .model-row {
        height: 1;
        padding: 0 1;
        align: left middle;
    }

    .model-name {
        width: 1fr;
        text-overflow: ellipsis;
    }

    .model-meta {
        width: 8;
        color: $text 60%;
        text-align: right;
    }

    ListItem {
        padding: 0;
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
        background: $primary 22%;
    }

    ListItem.--highlight {
        background: $primary 18%;
    }

    ListItem.--highlight .model-name {
        text-style: bold;
    }

    ListItem.--highlight .model-meta {
        color: $text 85%;
    }

    #confirm-dialog {
        border: round $panel;
        background: $surface;
        width: 1fr;
        max-width: 80;
        min-width: 40;
        max-height: 80%;
        height: auto;
        padding: 1 2;
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

    Screen.compact #topbar {
        margin: 0;
        border-left: none;
        border-right: none;
        border-top: none;
    }

    Screen.compact #main {
        layout: vertical;
        margin: 0;
    }

    Screen.compact .panel {
        padding: 0 1 1 1;
    }

    Screen.compact #left-pane,
    Screen.compact #right-pane {
        width: 1fr;
        min-width: 0;
    }

    Screen.compact #left-pane {
        margin-right: 0;
        margin-bottom: 1;
        height: 10;
        min-height: 10;
    }

    Screen.compact #right-pane {
        height: 1fr;
    }

    Screen.compact #filter {
        margin: 0 0 1 0;
    }

    Screen.compact #model-list {
        margin: 0;
    }

    Screen.compact #details-scroll {
        margin: 0;
    }

    Screen.compact .model-meta {
        width: 8;
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
        ("tab", "cycle_focus", "Next pane"),
        ("shift+tab", "cycle_focus_reverse", "Prev pane"),
        ("t", "toggle_theme", "Theme"),
        ("r", "reload", "Reload"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.theme = "textual-dark"
        self._models: list[ModelEntry] = []
        self._filtered: list[ModelEntry] = []
        self._last_launched_model: str | None = None
        self._launch_settings = resolve_effective_launch_settings()

    def compose(self) -> ComposeResult:
        with Vertical(id="topbar"):
            yield Static(build_status_line(), id="topbar-status", markup=True)
        with Container(id="main"):
            with Vertical(id="left-pane", classes="panel"):
                yield Input(
                    placeholder="Filter models",
                    id="filter",
                )
                yield ListView(id="model-list")
            with Vertical(id="right-pane", classes="panel"):
                with VerticalScroll(id="details-scroll"):
                    yield Static("", id="details")

    def on_mount(self) -> None:
        self._sync_layout()
        self._launch_settings = resolve_effective_launch_settings()
        self.load_models()
        self._last_launched_model = load_last_launched_model()
        self.populate_list(
            preferred_model=resolve_preferred_model_name(self._last_launched_model)
        )
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
        compact = size.width < self.COMPACT_WIDTH
        self.screen.set_class(compact, "compact")
        self.query_one("#topbar-status", Static).update(build_status_line(compact=compact))

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
            self.query_one("#details", Static).update(
                Text("No models match the current filter.", style="dim")
            )

    def _make_model_item(self, model: ModelEntry) -> ListItem:
        row = Horizontal(
            Label(model.name, classes="model-name"),
            Static(format_model_resources(model.config), classes="model-meta"),
            classes="model-row",
        )
        return ListItem(row)

    def update_details(self, model: ModelEntry) -> None:
        details = format_details(
            model.config,
            dark_mode=self.current_theme.dark,
            launch_settings=self._launch_settings,
        )
        self.query_one("#details", Static).update(details)

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

    def action_cycle_focus(self) -> None:
        focused = self.focused
        if isinstance(focused, Input):
            self.action_focus_list()
            return
        if isinstance(focused, ListView):
            self.action_focus_details()
            return
        self.action_focus_list()

    def action_cycle_focus_reverse(self) -> None:
        focused = self.focused
        if isinstance(focused, VerticalScroll):
            self.action_focus_list()
            return
        if isinstance(focused, ListView):
            self.action_focus_filter()
            return
        self.action_focus_details()

    def action_reload(self) -> None:
        self._launch_settings = resolve_effective_launch_settings()
        self.load_models()
        self.populate_list(
            preferred_model=resolve_preferred_model_name(self._last_launched_model)
        )

    def action_toggle_theme(self) -> None:
        self.theme = (
            "textual-light" if self.theme == "textual-dark" else "textual-dark"
        )
        model = self.get_selected_model()
        if model is not None:
            self.update_details(model)

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


def build_status_line(compact: bool = False) -> str:
    config_hint = _short_path_hint(resolve_model_config_path())
    env_status = "[green]ready[/green]" if LAUNCH_ENV_PATH.exists() else "[red]missing[/red]"
    if compact:
        return (
            f"[b]Config[/b] {config_hint}  [dim]•[/dim]  "
            f"[b]Env[/b] {env_status}  [dim]•[/dim]  "
            "[b]/[/b] filter  [dim]•[/dim]  [b]Tab[/b] panes  [dim]•[/dim]  [b]t[/b] theme  [dim]•[/dim]  [b]Enter[/b] launch"
        )
    return (
        f"[b]Config[/b] {config_hint}  [dim]•[/dim]  "
        f"[b]Env[/b] {env_status}  [dim]•[/dim]  "
        "[b]Keys[/b] / filter, Tab panes, t theme, Enter launch, q quit"
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
    base_dir = Path(state_dir) if state_dir else Path.home() / ".local" / "state"
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


def resolve_effective_launch_settings(
    environ: MutableMapping[str, str] | None = None,
) -> EffectiveLaunchSettings:
    """Resolve the effective shell-script launch settings shown in the TUI."""
    values = environ if environ is not None else os.environ

    remote_user = values.get("REMOTE_USER", "").strip()
    rsync_dest = values.get("RSYNC_DEST", "").strip()
    if not rsync_dest and remote_user:
        rsync_dest = f"/home/bsc/{remote_user}/repos/vector-inference"

    vec_inf_env = values.get("VEC_INF_ENV", "").strip()
    if not vec_inf_env and rsync_dest:
        vec_inf_env = f"{rsync_dest}/.venv"

    remote_work_dir = values.get("REMOTE_WORK_DIR", "").strip()
    if remote_work_dir == "RSYNC_DEST":
        remote_work_dir = rsync_dest

    remote_qos = values.get("REMOTE_QOS", "").strip()
    if remote_qos == "NONE":
        remote_qos = ""

    remote_launch_host = values.get("REMOTE_LAUNCH_HOST", "").strip() or None
    config_dir_remote = values.get("VEC_INF_CONFIG_DIR_REMOTE", "").strip()
    remote_account = values.get("REMOTE_ACCOUNT", "").strip() or None

    return EffectiveLaunchSettings(
        remote_launch_host=remote_launch_host,
        config_dir_remote=(
            None
            if not config_dir_remote or config_dir_remote == "NONE"
            else config_dir_remote
        ),
        rsync_dest=rsync_dest or None,
        vec_inf_env=vec_inf_env or None,
        remote_work_dir=remote_work_dir or None,
        remote_account=remote_account,
        remote_qos=remote_qos or None,
    )


def format_details(
    config: ModelConfig,
    dark_mode: bool,
    launch_settings: EffectiveLaunchSettings | None = None,
) -> Group:
    blocks: list[object] = []

    title = Text(config.model_name, style="bold")
    subtitle = Text()
    subtitle.append(config.model_type, style="bold")
    subtitle.append("  •  ", style="dim")
    subtitle.append(config.model_family, style="dim")
    if config.model_variant:
        subtitle.append("  •  ", style="dim")
        subtitle.append(str(config.model_variant), style="dim")
    if config.engine:
        subtitle.append("  •  ", style="dim")
        subtitle.append(str(config.engine), style="dim")

    job_rows = [
        (
            "Resources",
            (
                f"{config.num_nodes} node(s), {config.gpus_per_node} GPU(s)/node, "
                f"{config.cpus_per_task} CPU(s)/task"
            ),
        ),
        ("Time", config.time),
    ]
    if config.mem_per_node:
        job_rows.append(("Memory", config.mem_per_node))
    if config.partition:
        job_rows.append(("Partition", config.partition))
    if config.qos:
        job_rows.append(("QOS", config.qos))

    path_rows = [
        ("Venv", config.venv),
        ("Log dir", config.log_dir),
        ("Weights", config.model_weights_parent_dir),
    ]
    if config.work_dir:
        path_rows.append(("Work dir", config.work_dir))

    blocks.extend(
        [
            title,
            subtitle,
            Text(""),
            _detail_table("Job", job_rows),
        ]
    )

    if launch_settings is not None:
        blocks.extend(
            [
                Text(""),
                _detail_table(
                    "Launch",
                    [
                        ("Host", launch_settings.remote_launch_host or "-"),
                        ("Config dir", launch_settings.config_dir_remote or "-"),
                        ("Remote env", launch_settings.vec_inf_env or "-"),
                        ("Job work dir", _effective_work_dir(config, launch_settings)),
                        ("Account", _effective_account(config, launch_settings)),
                        ("QOS", _effective_qos(config, launch_settings)),
                    ],
                ),
            ]
        )

    if config.vllm_args:
        blocks.extend([Text(""), _yaml_block("vLLM args", config.vllm_args, dark_mode)])

    if config.sglang_args:
        blocks.extend(
            [Text(""), _yaml_block("SGLang args", config.sglang_args, dark_mode)]
        )

    if config.env:
        blocks.extend([Text(""), _yaml_block("Environment", config.env, dark_mode)])

    blocks.extend([Text(""), _detail_table("Paths", path_rows)])

    return Group(*blocks)


def format_confirm_summary(config: ModelConfig) -> Group:
    settings = resolve_effective_launch_settings()
    rows: list[tuple[str, str]] = [
        ("Type", config.model_type),
    ]
    if config.engine:
        rows.append(("Engine", str(config.engine)))
    rows.append(("Resources", f"{config.num_nodes} node(s), {config.gpus_per_node} GPU(s)/node"))
    rows.append(("Remote env", settings.vec_inf_env or "-"))
    rows.append(("Job work dir", _effective_work_dir(config, settings)))
    rows.append(("Account", _effective_account(config, settings)))
    rows.append(("QOS", _effective_qos(config, settings)))
    if config.partition or config.qos:
        parts = [p for p in (config.partition, config.qos) if p]
        rows.append(("Slurm", ", ".join(str(p) for p in parts)))
    rows.append(("Time", config.time))
    return Group(
        _detail_table("Summary", rows),
        Text(""),
        Text("Press Launch to continue, or Cancel.", style="dim"),
    )


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


def format_model_resources(config: ModelConfig) -> str:
    return f"{config.num_nodes:>2}n  {config.gpus_per_node:>2}g"


def _effective_work_dir(
    config: ModelConfig,
    launch_settings: EffectiveLaunchSettings,
) -> str:
    return launch_settings.remote_work_dir or str(config.work_dir or "-")


def _effective_account(
    config: ModelConfig,
    launch_settings: EffectiveLaunchSettings,
) -> str:
    if launch_settings.remote_account:
        return launch_settings.remote_account
    if config.account:
        return str(config.account)
    return "$VEC_INF_ACCOUNT (remote)"


def _effective_qos(
    config: ModelConfig,
    launch_settings: EffectiveLaunchSettings,
) -> str:
    return launch_settings.remote_qos or str(config.qos or "-")


def _detail_table(title: str, rows: list[tuple[str, object]]) -> Group:
    heading = Text(title, style="bold")
    table = Table.grid(padding=(0, 2), expand=False)
    table.add_column(style="dim", width=10, no_wrap=True)
    table.add_column(overflow="fold")
    for label, value in rows:
        table.add_row(label, str(value))
    return Group(heading, table)


def _yaml_block(title: str, values: dict[str, object], dark_mode: bool) -> Group:
    syntax = Syntax(
        "\n".join(_yaml_lines(values)),
        "yaml",
        theme="ansi_dark" if dark_mode else "ansi_light",
        line_numbers=False,
        word_wrap=True,
    )
    return Group(Text(title, style="bold"), syntax)


def _short_path_hint(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return path.name
    return path_str


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
    script_path = SCRIPT_DIR / "launch_and_tunnel.sh"
    if not script_path.exists():
        raise SystemExit(f"Launch script not found: {script_path}")
    if request.launch_only:
        os.execvp("bash", ["bash", str(script_path), "--launch-only", request.model_name])
    else:
        os.execvp("bash", ["bash", str(script_path), request.model_name])


def main() -> None:
    apply_launch_env_defaults(_launch_env_values)
    if "VEC_INF_CONFIG_DIR" not in os.environ and DEFAULT_CONFIG_DIR.exists():
        os.environ["VEC_INF_CONFIG_DIR"] = str(DEFAULT_CONFIG_DIR)
    app = LaunchTui()
    result = app.run()
    if isinstance(result, LaunchRequest):
        run_launch(result)


if __name__ == "__main__":
    main()
