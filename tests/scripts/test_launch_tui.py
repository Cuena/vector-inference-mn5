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


def test_resolve_effective_launch_settings_matches_shell_script_defaults() -> None:
    """The TUI preview should mirror launch_and_tunnel.sh fallback logic."""
    module = _load_launch_tui_module()

    settings = module.resolve_effective_launch_settings(
        {
            "REMOTE_USER": "alice",
            "REMOTE_LAUNCH_HOST": "alogin1.bsc.es",
            "VEC_INF_CONFIG_DIR_REMOTE": (
                "/home/bsc/alice/repos/vector-inference-mn5/vec_inf/config/marenostrum5"
            ),
            "REMOTE_WORK_DIR": "RSYNC_DEST",
            "REMOTE_QOS": "NONE",
        }
    )

    assert settings.remote_launch_host == "alogin1.bsc.es"
    assert settings.rsync_dest == "/home/bsc/alice/repos/vector-inference"
    assert settings.vec_inf_env == "/home/bsc/alice/repos/vector-inference/.venv"
    assert settings.remote_work_dir == "/home/bsc/alice/repos/vector-inference"
    assert settings.remote_qos is None


def test_effective_launch_helpers_prefer_script_overrides() -> None:
    """Launch previews should show shell-script overrides over profile defaults."""
    module = _load_launch_tui_module()
    settings = module.EffectiveLaunchSettings(
        remote_launch_host="alogin1.bsc.es",
        config_dir_remote="/remote/config",
        rsync_dest="/remote/repo",
        vec_inf_env="/remote/repo/.venv",
        remote_work_dir="/scratch/alice/vec-inf-work",
        remote_account="bsc70",
        remote_qos="acc_debug",
    )
    config = module.load_config()[0]

    assert module._effective_work_dir(config, settings) == "/scratch/alice/vec-inf-work"
    assert module._effective_account(config, settings) == "bsc70"
    assert module._effective_qos(config, settings) == "acc_debug"
