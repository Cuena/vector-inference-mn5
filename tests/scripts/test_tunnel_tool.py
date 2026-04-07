"""Tests for the local tunnel helper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_tunnel_tool_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "tunnel_tool.py"
    spec = importlib.util.spec_from_file_location("tunnel_tool_test_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_local_forward_spec_supports_compact_and_bound_forms() -> None:
    """SSH -L parsing should support common forwarding syntaxes."""
    module = _load_tunnel_tool_module()

    assert module.parse_local_forward_spec("5678:10.0.0.5:8000") == (
        5678,
        "10.0.0.5",
        8000,
    )
    assert module.parse_local_forward_spec("127.0.0.1:5678:node123:8000") == (
        5678,
        "node123",
        8000,
    )


def test_extract_local_forwards_from_ssh_command_reads_target_and_port() -> None:
    """The helper should recover the SSH target and forwarded port from ps output."""
    module = _load_tunnel_tool_module()

    target, forwards = module.extract_local_forwards_from_ssh_command(
        "ssh -o ExitOnForwardFailure=yes -N -L 5678:10.0.0.5:8000 alice@alogin1.bsc.es"
    )

    assert target == "alice@alogin1.bsc.es"
    assert forwards == {5678: ("10.0.0.5", 8000)}


def test_load_registered_sessions_ignores_stale_owner_pid(tmp_path: Path) -> None:
    """State files for dead launcher shells should be ignored."""
    module = _load_tunnel_tool_module()
    session_dir = tmp_path / ".tunnel-sessions"
    session_dir.mkdir()
    (session_dir / "123.json").write_text(
        json.dumps(
            {
                "job_id": "123",
                "requested_model_name": "Llama-3.2-3B-Instruct",
                "local_port": 5678,
                "base_url": "http://127.0.0.1:5678/v1",
                "owner_pid": 999999,
            }
        ),
        encoding="utf-8",
    )

    records = module.load_registered_sessions(session_dir=session_dir)

    assert records == {}


def test_merge_tunnel_records_keeps_state_job_id_and_listener_remote_host() -> None:
    """Merging should preserve complementary metadata from both sources."""
    module = _load_tunnel_tool_module()
    state_record = module.TunnelRecord(
        local_port=5678,
        base_url="http://127.0.0.1:5678/v1",
        job_id="12345",
        requested_model_name="Alias-Name",
        sources=["state"],
    )
    listener_record = module.TunnelRecord(
        local_port=5678,
        base_url="http://127.0.0.1:5678/v1",
        ssh_target="alice@alogin1.bsc.es",
        remote_server_host="node123",
        remote_server_port=8000,
        sources=["listener"],
    )

    merged = module.merge_tunnel_records({5678: state_record}, {5678: listener_record})

    assert merged[5678].job_id == "12345"
    assert merged[5678].remote_server_host == "node123"
    assert merged[5678].ssh_target == "alice@alogin1.bsc.es"
    assert merged[5678].sources == ["state", "listener"]


def test_build_curl_commands_uses_served_model_name() -> None:
    """Generated curl snippets should target the exact served model when known."""
    module = _load_tunnel_tool_module()
    record = module.TunnelRecord(
        local_port=5678,
        base_url="http://127.0.0.1:5678/v1",
        job_id="12345",
        requested_model_name="Alias-Name",
        served_model_names=["Exact-Served-Model"],
    )

    commands = module.build_curl_commands(record)

    assert 'MODEL=Exact-Served-Model' in commands[1]
    assert any(command == "vec-inf status 12345" for command in commands)


def test_build_validation_commands_includes_shortcuts_and_optional_status() -> None:
    """Named validation commands should expose stable keys for the TUI."""
    module = _load_tunnel_tool_module()
    record = module.TunnelRecord(
        local_port=5678,
        base_url="http://127.0.0.1:5678/v1",
        requested_model_name="Alias-Name",
        job_id="12345",
    )

    commands = module.build_validation_commands(record)

    assert [command.key for command in commands] == ["h", "m", "t", "c", "s"]
    assert "/completions" in commands[3].command
    assert "Tell a short joke about GPUs" in commands[3].command
    assert commands[-1].command == "vec-inf status 12345"


def test_build_validation_environment_prefers_served_model_name() -> None:
    """Environment assignments should use the exact served model when available."""
    module = _load_tunnel_tool_module()
    record = module.TunnelRecord(
        local_port=5678,
        base_url="http://127.0.0.1:5678/v1",
        requested_model_name="Alias-Name",
        served_model_names=["Exact-Served-Model"],
    )

    environment = module.build_validation_environment(record)

    assert environment == [
        "BASE_URL=http://127.0.0.1:5678/v1",
        "MODEL=Exact-Served-Model",
    ]


def test_probe_helpers_tolerate_connection_reset() -> None:
    """Transient local probe failures should not crash the helper."""
    module = _load_tunnel_tool_module()
    original_urlopen = module.request.urlopen

    def _raise_reset(*args, **kwargs):
        raise ConnectionResetError(104, "Connection reset by peer")

    module.request.urlopen = _raise_reset
    try:
        assert module.probe_health("http://127.0.0.1:5678/v1") is None
        model_names, probe_error = module.probe_model_names(
            "http://127.0.0.1:5678/v1"
        )
    finally:
        module.request.urlopen = original_urlopen

    assert model_names == []
    assert "Connection reset by peer" in probe_error
