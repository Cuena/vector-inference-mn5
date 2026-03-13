"""Tests for launch_tui config bootstrap helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_launch_tui_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "launch_tui.py"
    )
    spec = importlib.util.spec_from_file_location("launch_tui_test_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_launch_env_reads_quoted_values(tmp_path: Path) -> None:
    """The TUI should parse the same simple env syntax as the shell scripts."""
    module = _load_launch_tui_module()
    env_path = tmp_path / ".launch.env"
    env_path.write_text(
        'REMOTE_USER="alice"\nLOCAL_PORT=5678\nREMOTE_QOS=""\n',
        encoding="utf-8",
    )

    values = module.parse_launch_env(env_path)

    assert values["REMOTE_USER"] == "alice"
    assert values["LOCAL_PORT"] == "5678"
    assert values["REMOTE_QOS"] == ""


def test_derive_project_root_prefers_config_dir_remote() -> None:
    """The remote project root should come from the MN5 config dir when available."""
    module = _load_launch_tui_module()

    project_root = module.derive_project_root(
        {
            "VEC_INF_CONFIG_DIR_REMOTE": (
                "/home/bsc/alice/repos/vector-inference-mn5/vec_inf/config/marenostrum5"
            ),
            "RSYNC_DEST": "/home/bsc/alice/repos/other-checkout",
        }
    )

    assert project_root == "/home/bsc/alice/repos/vector-inference-mn5"


def test_apply_launch_env_defaults_keeps_explicit_env() -> None:
    """The TUI should not overwrite explicit shell env with file defaults."""
    module = _load_launch_tui_module()
    environ = {
        "REMOTE_USER": "shell-user",
        "VEC_INF_CONFIG_DIR_REMOTE": (
            "/home/bsc/alice/repos/vector-inference-mn5/vec_inf/config/marenostrum5"
        ),
    }

    module.apply_launch_env_defaults(
        {
            "REMOTE_USER": "file-user",
            "RSYNC_DEST": "/home/bsc/alice/repos/vector-inference-mn5",
        },
        environ=environ,
    )

    assert environ["REMOTE_USER"] == "shell-user"
    assert environ["RSYNC_DEST"] == "/home/bsc/alice/repos/vector-inference-mn5"
    assert (
        environ["VEC_INF_PROJECT_ROOT"]
        == "/home/bsc/alice/repos/vector-inference-mn5"
    )


def test_resolve_preferred_model_name_defaults_to_llama_smoke_test() -> None:
    """The TUI should default to the lightweight Llama smoke-test model."""
    module = _load_launch_tui_module()

    assert module.resolve_preferred_model_name(None) == "Llama-3.2-3B-Instruct"
    assert module.resolve_preferred_model_name("Qwen3.5-27B") == "Qwen3.5-27B"
