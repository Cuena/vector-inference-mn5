"""Tests for environment-based config defaults."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from vec_inf.client._slurm_vars import DEFAULT_ARGS
from vec_inf.client.config import ModelConfig


def test_gpus_per_node_defaults_from_environment() -> None:
    """ModelConfig uses default_args.gpus_per_node when configured."""
    kwargs = {
        "model_name": "test-model",
        "model_family": "test-family",
        "model_type": "LLM",
        "num_nodes": 1,
        "vocab_size": 32000,
    }

    if DEFAULT_ARGS.get("gpus_per_node"):
        cfg = ModelConfig(**kwargs)
        assert cfg.gpus_per_node == int(DEFAULT_ARGS["gpus_per_node"])
    else:
        with pytest.raises(ValidationError):
            ModelConfig(**kwargs)


def test_model_config_expands_env_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """ModelConfig expands simple env placeholders in path fields."""
    monkeypatch.setenv("TEST_VEC_INF_ROOT", "/tmp/vec-inf-root")

    cfg = ModelConfig(
        model_name="test-model",
        model_family="test-family",
        model_type="LLM",
        num_nodes=1,
        gpus_per_node=1,
        vocab_size=32000,
        log_dir="$TEST_VEC_INF_ROOT/logs",
        model_weights_parent_dir="$TEST_VEC_INF_ROOT/models",
    )

    assert cfg.log_dir == Path("/tmp/vec-inf-root/logs")
    assert cfg.model_weights_parent_dir == Path("/tmp/vec-inf-root/models")


def test_model_config_expands_env_paths_with_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ModelConfig expands shell-style default placeholders in path fields."""
    monkeypatch.delenv("TEST_VEC_INF_ROOT", raising=False)
    monkeypatch.setenv("TEST_STORAGE_USER", "example_user")

    cfg = ModelConfig(
        model_name="test-model",
        model_family="test-family",
        model_type="LLM",
        num_nodes=1,
        gpus_per_node=1,
        vocab_size=32000,
        model_weights_parent_dir=(
            "${TEST_VEC_INF_ROOT:-/gpfs/scratch/users/$TEST_STORAGE_USER/models}"
        ),
    )

    assert cfg.model_weights_parent_dir == Path("/gpfs/scratch/users/example_user/models")


def test_model_config_resolves_repo_relative_engine_args() -> None:
    """ModelConfig resolves @repo paths in engine argument dictionaries."""
    cfg = ModelConfig(
        model_name="test-model",
        model_family="test-family",
        model_type="LLM",
        num_nodes=1,
        gpus_per_node=1,
        vocab_size=32000,
        vllm_args={"--config": "@repo/compilation_configs/generic.yaml"},
    )

    config_path = cfg.vllm_args["--config"] if cfg.vllm_args else None
    assert config_path is not None
    assert Path(config_path).name == "generic.yaml"
    assert Path(config_path).exists()
