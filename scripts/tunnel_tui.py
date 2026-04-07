#!/usr/bin/env python3
"""Textual TUI for inspecting active local vec-inf tunnels."""

from __future__ import annotations

import subprocess

from rich.console import Group
from rich.table import Table
from rich.text import Text

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.events import Resize
from textual.widgets import DataTable, Static

from tunnel_tool import (
    TunnelRecord,
    ValidationCommand,
    build_validation_commands,
    build_validation_environment,
    discover_ssh_tunnels,
    enrich_records,
    load_registered_sessions,
    merge_tunnel_records,
)


def load_tunnel_records() -> list[TunnelRecord]:
    """Load and enrich active local tunnel records."""
    registered = load_registered_sessions()
    discovered = discover_ssh_tunnels()
    records = enrich_records(merge_tunnel_records(registered, discovered))
    return sorted(records.values(), key=lambda record: record.local_port)


def _health_style(record: TunnelRecord) -> tuple[str, str]:
    """Return (label, rich style) for health status."""
    if record.health_ok is True:
        return "ok", "green"
    if record.probe_error:
        return "down", "red"
    return "?", "yellow"


def _kv_table(rows: list[tuple[str, str]], label_width: int = 14) -> Table:
    table = Table.grid(padding=(0, 2), expand=False)
    table.add_column(style="dim", width=label_width, no_wrap=True)
    table.add_column(overflow="fold")
    for label, value in rows:
        table.add_row(label, value)
    return table


def build_status_line(record_count: int, loading: bool = False) -> str:
    noun = "tunnel" if record_count == 1 else "tunnels"
    suffix = "  loading…" if loading else ""
    return f" {record_count} active {noun}{suffix}"


def format_details(record: TunnelRecord | None) -> Group | Text:
    if record is None:
        return Text("No active tunnels.", style="dim italic")

    health_label, health_style = _health_style(record)

    rows: list[tuple[str, str]] = [
        ("URL", record.base_url),
        ("Model", record.display_model_name()),
        ("Health", health_label),
        ("Job", record.job_id or "-"),
        ("SSH target", record.ssh_target or "-"),
    ]
    if record.remote_server_host or record.remote_server_port:
        host = record.remote_server_host or record.remote_server_ip or "-"
        port = record.remote_server_port or "-"
        rows.append(("Remote", f"{host}:{port}"))
    if record.probe_error:
        rows.append(("Error", record.probe_error))

    kv = _kv_table(rows)

    health_row_idx = next(i for i, (l, _) in enumerate(rows) if l == "Health")
    kv.columns[1]._cells[health_row_idx] = Text(health_label, style=f"bold {health_style}")

    return Group(Text("Details", style="bold"), kv)


def format_commands(record: TunnelRecord | None) -> Group | Text:
    if record is None:
        return Text("Select a tunnel to see commands.", style="dim italic")

    specs = build_validation_commands(record)
    env_lines = build_validation_environment(record)

    parts: list[Text | Table] = [Text("Commands", style="bold")]

    env_table = Table.grid(padding=(0, 1), expand=False)
    env_table.add_column(style="dim")
    for line in env_lines:
        env_table.add_row(line)
    parts.append(env_table)
    parts.append(Text(""))

    cmd_table = Table.grid(padding=(0, 2), expand=False)
    cmd_table.add_column(style="bold cyan", width=3, no_wrap=True)
    cmd_table.add_column(style="bold", width=12, no_wrap=True)
    cmd_table.add_column(style="dim", overflow="fold")
    for spec in specs:
        cmd_table.add_row(spec.key, spec.label, spec.command)
    parts.append(cmd_table)

    return Group(*parts)


def format_output(
    record: TunnelRecord | None,
    command: ValidationCommand | None = None,
) -> Group | Text:
    if record is None:
        return Text("Select a tunnel, then press a key to run a check.", style="dim italic")
    if command is None:
        keys = ", ".join(f"[bold cyan]{s.key}[/bold cyan]" for s in build_validation_commands(record))
        return Text.from_markup(
            f"Port {record.local_port} selected.  Press {keys} to run a check."
        )
    return Text(f"Running {command.label}…", style="dim italic")


def render_command_output(
    command: ValidationCommand,
    exit_code: int,
    stdout: str,
    stderr: str,
) -> Group:
    stdout = stdout.strip()
    stderr = stderr.strip()

    status_style = "green" if exit_code == 0 else "red"
    header = Text.assemble(
        ("$ ", "dim"),
        (command.command, ""),
        ("  →  ", "dim"),
        (str(exit_code), f"bold {status_style}"),
    )

    parts: list[Text] = [header, Text("")]
    if stdout:
        parts.append(Text(stdout))
    else:
        parts.append(Text("<no output>", style="dim italic"))
    if stderr:
        parts.append(Text(""))
        parts.append(Text(stderr, style="red"))

    return Group(*parts)


class TunnelTui(App[None]):
    """Inspect active local vec-inf tunnels in a Textual TUI."""

    TITLE = "vec-inf tunnels"
    COMPACT_WIDTH = 94
    AUTO_REFRESH_SECONDS = 30.0
    COMMAND_TIMEOUT_SECONDS = 20.0

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
        min-width: 28;
        max-width: 46;
        margin-right: 1;
    }
    #right-pane {
        width: 1fr;
    }

    #tunnel-table {
        height: 1fr;
        background: $background 15%;
        border: none;
    }

    #info-scroll {
        height: auto;
        max-height: 60%;
        background: $background 15%;
        border: none;
        margin-bottom: 1;
    }
    #output-scroll {
        height: 1fr;
        background: $background 15%;
        border: none;
    }

    #info, #output {
        padding: 0 1;
    }

    #keyhints {
        height: 1;
        dock: bottom;
        background: $surface;
        color: $text 60%;
        padding: 0 1;
    }

    Screen.compact #topbar {
        margin: 0;
        border-left: none;
        border-right: none;
        border-top: none;
    }
    Screen.compact #main {
        layout: vertical;
        margin: 0 1;
    }
    Screen.compact .panel {
        padding: 0 1 1 1;
    }
    Screen.compact #left-pane,
    Screen.compact #right-pane {
        width: 1fr;
        min-width: 0;
        max-width: 1fr;
        margin-right: 0;
    }
    Screen.compact #left-pane {
        margin-bottom: 1;
        height: 10;
        min-height: 10;
    }
    Screen.compact #right-pane {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("h", "run_command('h')", "Health"),
        ("m", "run_command('m')", "Models"),
        ("t", "run_command('t')", "Metrics"),
        ("c", "run_command('c')", "Chat"),
        ("s", "run_command('s')", "Status"),
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.theme = "textual-dark"
        self._records: list[TunnelRecord] = []
        self._refresh_pending = False
        self._command_pending = False
        self._output_port: int | None = None
        self._output_has_result = False
        self._rebuilding_table = False

    def compose(self) -> ComposeResult:
        with Vertical(id="topbar"):
            yield Static(build_status_line(0, loading=True), id="topbar-status")
        with Container(id="main"):
            with Vertical(id="left-pane", classes="panel"):
                yield DataTable(id="tunnel-table")
            with Vertical(id="right-pane", classes="panel"):
                with VerticalScroll(id="info-scroll"):
                    yield Static(id="info")
                with VerticalScroll(id="output-scroll"):
                    yield Static(id="output")
        yield Static(
            " [b cyan]h[/]ealth  [b cyan]m[/]odels  me[b cyan]t[/]rics  "
            "[b cyan]c[/]hat  [b cyan]s[/]tatus  "
            "[b cyan]r[/]efresh  [b cyan]q[/]uit",
            id="keyhints",
            markup=True,
        )

    def on_mount(self) -> None:
        self._sync_layout()
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Port", "Job", "Health", "Model")
        self.set_interval(self.AUTO_REFRESH_SECONDS, self.schedule_refresh)
        self.schedule_refresh()
        table.focus()

    def on_resize(self, event: Resize) -> None:
        self._sync_layout()

    def _sync_layout(self) -> None:
        compact = self.size.width < self.COMPACT_WIDTH
        self.screen.set_class(compact, "compact")

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def action_refresh(self) -> None:
        self.schedule_refresh()

    def schedule_refresh(self) -> None:
        if self._refresh_pending:
            return
        self._refresh_pending = True
        selected_port = self._selected_port()
        self.query_one("#topbar-status", Static).update(
            build_status_line(len(self._records), loading=True)
        )
        self._refresh_worker(selected_port)

    @work(thread=True, exclusive=True, group="refresh", exit_on_error=False)
    def _refresh_worker(self, selected_port: int | None) -> None:
        try:
            records = load_tunnel_records()
        except Exception as exc:
            self.call_from_thread(self._on_refresh_error, str(exc))
            return
        self.call_from_thread(self._on_refresh_done, records, selected_port)

    def _on_refresh_done(
        self,
        records: list[TunnelRecord],
        selected_port: int | None,
    ) -> None:
        table = self.query_one(DataTable)
        self._records = records
        self.query_one("#topbar-status", Static).update(
            build_status_line(len(self._records))
        )
        self._rebuilding_table = True
        table.clear(columns=False)
        for rec in self._records:
            hlabel, hstyle = _health_style(rec)
            table.add_row(
                str(rec.local_port),
                rec.job_id or "-",
                Text(hlabel, style=hstyle),
                rec.display_model_name(),
            )
        if self._records:
            row = self._find_row_for_port(selected_port)
            table.move_cursor(row=row, column=0)
            self._update_info_pane(row)
            new_port = self._records[row].local_port
            if new_port != self._output_port:
                self._reset_output(self._records[row])
        else:
            self._update_info_pane(None)
            if self._output_port is not None:
                self._reset_output(None)
        self._rebuilding_table = False
        self._refresh_pending = False

    def _on_refresh_error(self, error: str) -> None:
        self.query_one("#topbar-status", Static).update(
            build_status_line(len(self._records))
        )
        self.query_one("#output", Static).update(
            Text(f"Refresh failed: {error}", style="red")
        )
        self._refresh_pending = False

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _selected_port(self) -> int | None:
        table = self.query_one(DataTable)
        row = table.cursor_row
        if row is None or row < 0 or row >= len(self._records):
            return None
        return self._records[row].local_port

    def _find_row_for_port(self, port: int | None) -> int:
        if port is None:
            return 0
        for i, rec in enumerate(self._records):
            if rec.local_port == port:
                return i
        return 0

    def _selected_record(self) -> TunnelRecord | None:
        table = self.query_one(DataTable)
        row = table.cursor_row
        if row is None or row < 0 or row >= len(self._records):
            return None
        return self._records[row]

    def _update_info_pane(self, row: int | None) -> None:
        rec = None if row is None else self._records[row]
        details = format_details(rec)
        commands = format_commands(rec)
        self.query_one("#info", Static).update(Group(details, Text(""), commands))

    def _reset_output(self, rec: TunnelRecord | None) -> None:
        self._output_port = rec.local_port if rec else None
        self._output_has_result = False
        self.query_one("#output", Static).update(format_output(rec))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if getattr(self, "_rebuilding_table", False):
            return
        if event.cursor_row is None or event.cursor_row >= len(self._records):
            self._update_info_pane(None)
            self._reset_output(None)
            return
        rec = self._records[event.cursor_row]
        self._update_info_pane(event.cursor_row)
        if rec.local_port != self._output_port:
            self._reset_output(rec)

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def action_run_command(self, key: str) -> None:
        if self._command_pending:
            self.query_one("#output", Static).update(
                Text("A command is already running…", style="yellow")
            )
            return

        rec = self._selected_record()
        if rec is None:
            self.query_one("#output", Static).update(format_output(None))
            return

        spec = next(
            (s for s in build_validation_commands(rec) if s.key == key), None
        )
        if spec is None:
            self.query_one("#output", Static).update(
                Text(f"No command for key '{key}'.", style="yellow")
            )
            return

        self.query_one("#output", Static).update(format_output(rec, spec))
        self._command_pending = True
        self._command_worker(rec, spec)

    @work(thread=True, group="command", exit_on_error=False)
    def _command_worker(self, record: TunnelRecord, command: ValidationCommand) -> None:
        script = "\n".join(build_validation_environment(record) + [command.command])
        try:
            result = subprocess.run(
                ["bash", "-c", script],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.COMMAND_TIMEOUT_SECONDS,
            )
            output = render_command_output(
                command, result.returncode, result.stdout, result.stderr
            )
        except subprocess.TimeoutExpired as exc:
            output = render_command_output(
                command, -1, exc.stdout or "", (exc.stderr or "").strip() or "Timed out."
            )
        except Exception as exc:
            output = render_command_output(command, -1, "", str(exc))
        self.call_from_thread(self._on_command_done, output)

    def _on_command_done(self, output: Group) -> None:
        self.query_one("#output", Static).update(output)
        self._output_has_result = True
        self._command_pending = False


def main() -> None:
    """Run the tunnel inspector TUI."""
    TunnelTui().run()


if __name__ == "__main__":
    main()
