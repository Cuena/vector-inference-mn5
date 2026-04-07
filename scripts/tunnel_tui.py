#!/usr/bin/env python3
"""Textual TUI for inspecting active local vec-inf tunnels."""

from __future__ import annotations

import subprocess

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


def format_health(record: TunnelRecord) -> str:
    """Format a compact health label."""
    if record.health_ok is True:
        return "ok"
    if record.probe_error:
        return "down"
    return "unknown"


def build_status_line(record_count: int, loading: bool = False) -> str:
    """Build the top-level status summary."""
    noun = "tunnel" if record_count == 1 else "tunnels"
    loading_text = " | loading..." if loading else ""
    return f"{record_count} active local {noun}{loading_text} | refresh every 5s"


def format_details_text(record: TunnelRecord | None) -> str:
    """Render the details pane for a selected tunnel."""
    if record is None:
        return "No active local vec-inf tunnels found."

    lines = [
        f"Port: {record.local_port}",
        f"Base URL: {record.base_url}",
        f"Served model(s): {record.display_model_name()}",
        f"Requested model: {record.requested_model_name or '-'}",
        f"Job ID: {record.job_id or '-'}",
        f"Health: {format_health(record)}",
        f"SSH target: {record.ssh_target or '-'}",
    ]
    if record.remote_server_host or record.remote_server_port:
        remote_host = record.remote_server_host or record.remote_server_ip or "-"
        remote_port = record.remote_server_port or "-"
        lines.append(f"Remote server: {remote_host}:{remote_port}")
    if record.probe_error:
        lines.append(f"Probe error: {record.probe_error}")
    return "\n".join(lines)


def format_commands_text(record: TunnelRecord | None) -> str:
    """Render canned validation commands for a selected tunnel."""
    if record is None:
        return "Select a live tunnel to see commands."

    command_specs = build_validation_commands(record)
    sections = ["Environment:"]
    sections.extend(build_validation_environment(record))
    sections.append("")
    sections.append("Commands:")
    for command in command_specs:
        sections.append(f"[{command.key}] {command.label}")
        sections.append(command.command)
        sections.append("")
    return "\n".join(sections).rstrip()


def format_output_text(record: TunnelRecord | None, command: ValidationCommand | None = None) -> str:
    """Render the output pane placeholder for the selected tunnel."""
    if record is None:
        return "Select a live tunnel, then press h/m/t/c/s to run a predefined check."
    if command is None:
        available = ", ".join(spec.key for spec in build_validation_commands(record))
        return (
            f"Selected tunnel: port {record.local_port}\n"
            f"Available commands: {available}\n"
            "Press the matching key to run a check."
        )
    return f"Running {command.label} for port {record.local_port}..."


def render_command_output(command: ValidationCommand, exit_code: int, stdout: str, stderr: str) -> str:
    """Render command execution output as plain text."""
    stdout = stdout.strip()
    stderr = stderr.strip()
    sections = [
        f"$ {command.command}",
        f"Exit code: {exit_code}",
        "",
        "STDOUT:",
        stdout or "<empty>",
    ]
    if stderr:
        sections.extend(["", "STDERR:", stderr])
    return "\n".join(sections)


class TunnelTui(App[None]):
    """Inspect active local vec-inf tunnels in a Textual TUI."""

    TITLE = "Vector Inference Tunnel Inspector"
    SUB_TITLE = "Browse active local tunnels and validation calls"
    COMPACT_WIDTH = 94
    AUTO_REFRESH_SECONDS = 5.0
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

    .section-title {
        height: 1;
        color: $text;
        text-style: bold;
        margin-bottom: 0;
    }

    #tunnel-table {
        height: 1fr;
        background: $background 15%;
        border: none;
    }

    #details-scroll,
    #commands-scroll,
    #output-scroll {
        background: $background 15%;
        border: none;
    }

    #details-scroll {
        height: 9;
        margin-bottom: 1;
    }

    #commands-scroll {
        height: 12;
        margin-bottom: 1;
    }

    #output {
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
        ("h", "run_health", "Health"),
        ("m", "run_models", "Models"),
        ("t", "run_metrics", "Metrics"),
        ("c", "run_completion", "Completion"),
        ("s", "run_status", "Status"),
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.theme = "textual-dark"
        self._records: list[TunnelRecord] = []
        self._refresh_pending = False
        self._command_pending = False

    def compose(self) -> ComposeResult:
        with Vertical(id="topbar"):
            yield Static(build_status_line(0, loading=True), id="topbar-status")
        with Container(id="main"):
            with Vertical(id="left-pane", classes="panel"):
                yield DataTable(id="tunnel-table")
            with Vertical(id="right-pane", classes="panel"):
                yield Static("Details", classes="section-title")
                with VerticalScroll(id="details-scroll"):
                    yield Static(format_details_text(None), id="details")
                yield Static("Commands", classes="section-title")
                with VerticalScroll(id="commands-scroll"):
                    yield Static(format_commands_text(None), id="commands")
                yield Static("Output", classes="section-title")
                with VerticalScroll(id="output-scroll"):
                    yield Static(format_output_text(None), id="output")

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
        size = self.size
        compact = size.width < self.COMPACT_WIDTH
        self.screen.set_class(compact, "compact")

    def action_refresh(self) -> None:
        self.schedule_refresh()

    def action_run_health(self) -> None:
        self.run_selected_command("h")

    def action_run_models(self) -> None:
        self.run_selected_command("m")

    def action_run_metrics(self) -> None:
        self.run_selected_command("t")

    def action_run_completion(self) -> None:
        self.run_selected_command("c")

    def action_run_status(self) -> None:
        self.run_selected_command("s")

    def schedule_refresh(self) -> None:
        """Schedule a refresh so the UI can paint before probing tunnels."""
        if self._refresh_pending:
            return
        self._refresh_pending = True
        selected_port = self._selected_port()
        self.query_one("#topbar-status", Static).update(
            build_status_line(len(self._records), loading=True)
        )
        self.refresh_records_worker(selected_port)

    @work(thread=True, exclusive=True, group="refresh", exit_on_error=False)
    def refresh_records_worker(self, selected_port: int | None) -> None:
        """Refresh active tunnel records without blocking the UI thread."""
        try:
            records = load_tunnel_records()
        except Exception as exc:
            self.call_from_thread(self._finish_refresh_error, str(exc))
            return

        self.call_from_thread(self._apply_refreshed_records, records, selected_port)

    def _selected_port(self) -> int | None:
        """Return the currently selected port before a refresh."""
        table = self.query_one(DataTable)
        cursor_row = table.cursor_row
        if cursor_row is None or cursor_row < 0 or cursor_row >= len(self._records):
            return None
        return self._records[cursor_row].local_port

    def _find_row_for_port(self, port: int | None) -> int:
        """Return the best row to highlight after a refresh."""
        if port is None:
            return 0
        for index, record in enumerate(self._records):
            if record.local_port == port:
                return index
        return 0

    def update_panels(self, row_index: int | None) -> None:
        """Update detail/command panes for the selected row."""
        record = None if row_index is None else self._records[row_index]
        self.query_one("#details", Static).update(format_details_text(record))
        self.query_one("#commands", Static).update(format_commands_text(record))
        self.query_one("#output", Static).update(format_output_text(record))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.cursor_row is None or event.cursor_row >= len(self._records):
            self.update_panels(None)
            return
        self.update_panels(event.cursor_row)

    def selected_record(self) -> TunnelRecord | None:
        """Return the currently selected tunnel record."""
        table = self.query_one(DataTable)
        cursor_row = table.cursor_row
        if cursor_row is None or cursor_row < 0 or cursor_row >= len(self._records):
            return None
        return self._records[cursor_row]

    def run_selected_command(self, key: str) -> None:
        """Run a predefined validation command for the selected tunnel."""
        if self._command_pending:
            self.query_one("#output", Static).update(
                "A validation command is already running."
            )
            return

        record = self.selected_record()
        if record is None:
            self.query_one("#output", Static).update(format_output_text(None))
            return

        command = next(
            (spec for spec in build_validation_commands(record) if spec.key == key),
            None,
        )
        if command is None:
            self.query_one("#output", Static).update(
                f"No command is available for key '{key}' on port {record.local_port}."
            )
            return

        self.query_one("#output", Static).update(format_output_text(record, command))
        self._command_pending = True
        self.run_selected_command_worker(record, command)

    @work(thread=True, group="command", exit_on_error=False)
    def run_selected_command_worker(
        self,
        record: TunnelRecord,
        command: ValidationCommand,
    ) -> None:
        """Run a canned validation command without blocking the UI thread."""
        script = "\n".join(build_validation_environment(record) + [command.command])
        try:
            result = subprocess.run(
                ["bash", "-lc", script],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.COMMAND_TIMEOUT_SECONDS,
            )
            output = render_command_output(
                command,
                result.returncode,
                result.stdout,
                result.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            output = render_command_output(
                command,
                -1,
                exc.stdout or "",
                (exc.stderr or "").strip() or "Command timed out.",
            )
        except Exception as exc:
            output = render_command_output(command, -1, "", str(exc))
        self.call_from_thread(self._finish_command, output)

    def _apply_refreshed_records(
        self,
        records: list[TunnelRecord],
        selected_port: int | None,
    ) -> None:
        """Apply refreshed records from a background worker."""
        table = self.query_one(DataTable)
        self._records = records
        self.query_one("#topbar-status", Static).update(build_status_line(len(self._records)))
        table.clear(columns=False)
        for record in self._records:
            table.add_row(
                str(record.local_port),
                record.job_id or "-",
                format_health(record),
                record.display_model_name(),
            )

        if self._records:
            selected_row = self._find_row_for_port(selected_port)
            table.move_cursor(row=selected_row, column=0)
            self.update_panels(selected_row)
        else:
            self.update_panels(None)
        self._refresh_pending = False

    def _finish_refresh_error(self, error: str) -> None:
        """Reset refresh state after a background refresh failure."""
        self.query_one("#topbar-status", Static).update(
            build_status_line(len(self._records))
        )
        self.query_one("#output", Static).update(f"Refresh failed: {error}")
        self._refresh_pending = False

    def _finish_command(self, output: str) -> None:
        """Update the output pane when a background command finishes."""
        self.query_one("#output", Static).update(output)
        self._command_pending = False


def main() -> None:
    """Run the tunnel inspector TUI."""
    TunnelTui().run()


if __name__ == "__main__":
    main()
