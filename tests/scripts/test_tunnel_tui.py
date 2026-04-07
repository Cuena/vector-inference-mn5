"""Tests for the tunnel inspector TUI helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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


def test_build_status_line_pluralizes() -> None:
    """The TUI status line should describe the active tunnel count."""
    module = _load_tunnel_tui_module()

    assert module.build_status_line(1).startswith("1 active local tunnel")
    assert module.build_status_line(2).startswith("2 active local tunnels")
    assert "loading..." in module.build_status_line(1, loading=True)


def test_format_commands_text_uses_openai_compatible_calls() -> None:
    """The command pane should render the shared OpenAI-style validation calls."""
    module = _load_tunnel_tui_module()
    record = module.TunnelRecord(
        local_port=5678,
        base_url="http://127.0.0.1:5678/v1",
        served_model_names=["Exact-Served-Model"],
        job_id="12345",
    )

    commands_text = module.format_commands_text(record)

    assert "/completions" in commands_text
    assert "Tell a short joke about GPUs" in commands_text
    assert "Exact-Served-Model" in commands_text
    assert "[h] Health" in commands_text
    assert "vec-inf status 12345" in commands_text


def test_find_row_for_port_preserves_selection_when_present() -> None:
    """The auto-refresh path should keep the same tunnel selected when possible."""
    module = _load_tunnel_tui_module()
    app = module.TunnelTui()
    app._records = [
        module.TunnelRecord(local_port=5678, base_url="http://127.0.0.1:5678/v1"),
        module.TunnelRecord(local_port=6789, base_url="http://127.0.0.1:6789/v1"),
    ]

    assert app._find_row_for_port(6789) == 1
    assert app._find_row_for_port(9999) == 0


def test_format_output_text_lists_shortcuts_for_selected_tunnel() -> None:
    """The output pane should explain how to run the predefined checks."""
    module = _load_tunnel_tui_module()
    record = module.TunnelRecord(
        local_port=5678,
        base_url="http://127.0.0.1:5678/v1",
        job_id="12345",
    )

    output = module.format_output_text(record)

    assert "Selected tunnel: port 5678" in output
    assert "h, m, t, c, s" in output
