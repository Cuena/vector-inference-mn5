"""Tests for the tunnel inspector TUI helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from rich.console import Console


def _load_tunnel_tui_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "tunnel_tui.py"
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("tunnel_tui_test_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _render_to_text(renderable) -> str:
    """Render a Rich renderable to plain text for assertion."""
    console = Console(file=None, force_terminal=False, no_color=True, width=200)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def test_build_status_line_pluralizes() -> None:
    module = _load_tunnel_tui_module()

    assert "1 active tunnel" in module.build_status_line(1)
    assert "2 active tunnels" in module.build_status_line(2)
    assert "loading" in module.build_status_line(1, loading=True)


def test_format_commands_uses_openai_compatible_calls() -> None:
    module = _load_tunnel_tui_module()
    record = module.TunnelRecord(
        local_port=5678,
        base_url="http://127.0.0.1:5678/v1",
        served_model_names=["Exact-Served-Model"],
        job_id="12345",
    )

    text = _render_to_text(module.format_commands(record))

    assert "/chat/completions" in text
    assert "Say hi" in text
    assert "Exact-Served-Model" in text
    assert "Health" in text
    assert "scontrol show job" in text
    assert "12345" in text


def test_find_row_for_port_preserves_selection_when_present() -> None:
    module = _load_tunnel_tui_module()
    app = module.TunnelTui()
    app._records = [
        module.TunnelRecord(local_port=5678, base_url="http://127.0.0.1:5678/v1"),
        module.TunnelRecord(local_port=6789, base_url="http://127.0.0.1:6789/v1"),
    ]

    assert app._find_row_for_port(6789) == 1
    assert app._find_row_for_port(9999) == 0


def test_format_output_lists_shortcuts_for_selected_tunnel() -> None:
    module = _load_tunnel_tui_module()
    record = module.TunnelRecord(
        local_port=5678,
        base_url="http://127.0.0.1:5678/v1",
        job_id="12345",
    )

    text = _render_to_text(module.format_output(record))

    assert "5678" in text
    assert "h" in text
    assert "m" in text
    assert "c" in text


def test_format_details_shows_model_and_health() -> None:
    module = _load_tunnel_tui_module()
    record = module.TunnelRecord(
        local_port=5678,
        base_url="http://127.0.0.1:5678/v1",
        served_model_names=["MyModel"],
        health_ok=True,
        job_id="999",
    )

    text = _render_to_text(module.format_details(record))

    assert "MyModel" in text
    assert "ok" in text
    assert "999" in text
