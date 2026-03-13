"""Tests for the MN5 setup wizard."""

from vec_inf.mn5_setup_wizard import (
    DEFAULT_LIGHTWEIGHT_MODEL,
    WizardConfig,
    build_launch_run_prompt,
    build_setup_run_prompt,
    describe_launch_effects,
    describe_setup_effects,
    build_defaults,
    render_launch_env,
)


def test_build_defaults_uses_repo_name_and_storage_user() -> None:
    """Derived defaults should stay aligned with the current repo layout."""
    defaults = build_defaults(remote_user="alice", storage_user="sharedalice")

    assert defaults["RSYNC_DEST"].endswith("/repos/vector-inference-mn5")
    assert defaults["VEC_INF_ENV"].endswith("/repos/vector-inference-mn5/.venv")
    assert defaults["VEC_INF_CONFIG_DIR_REMOTE"].endswith(
        "/repos/vector-inference-mn5/vec_inf/config/marenostrum5"
    )
    assert (
        defaults["VEC_INF_VLLM_IMAGE_PATH"]
        == "/gpfs/scratch/bsc70/singularity/vllm_openai_latest.sif"
    )
    assert defaults["MODEL_NAME"] == DEFAULT_LIGHTWEIGHT_MODEL


def test_render_launch_env_contains_mn5_profile_paths() -> None:
    """Rendered launcher config should include the wizard-managed MN5 paths."""
    defaults = build_defaults(remote_user="alice")
    config_text = render_launch_env(
        WizardConfig(
            remote_launch_host=defaults["REMOTE_LAUNCH_HOST"],
            remote_transfer_host=defaults["REMOTE_TRANSFER_HOST"],
            remote_internet_host=defaults["REMOTE_INTERNET_HOST"],
            remote_user=defaults["REMOTE_USER"],
            vec_inf_storage_user=defaults["VEC_INF_STORAGE_USER"],
            model_name=defaults["MODEL_NAME"],
            local_port=defaults["LOCAL_PORT"],
            auto_kill_stale_tunnel=defaults["AUTO_KILL_STALE_TUNNEL"],
            rsync_enabled=defaults["RSYNC_ENABLED"],
            rsync_src=defaults["RSYNC_SRC"],
            rsync_dest=defaults["RSYNC_DEST"],
            vec_inf_env=defaults["VEC_INF_ENV"],
            vec_inf_config_dir_remote=defaults["VEC_INF_CONFIG_DIR_REMOTE"],
            remote_work_dir=defaults["REMOTE_WORK_DIR"],
            remote_account=defaults["REMOTE_ACCOUNT"],
            remote_qos=defaults["REMOTE_QOS"],
            uv_sync_args=defaults["UV_SYNC_ARGS"],
            job_start_timeout=defaults["JOB_START_TIMEOUT"],
            server_ready_timeout=defaults["SERVER_READY_TIMEOUT"],
            vec_inf_vllm_image_path=defaults["VEC_INF_VLLM_IMAGE_PATH"],
            vec_inf_cached_model_config_path=defaults[
                "VEC_INF_CACHED_MODEL_CONFIG_PATH"
            ],
            vec_inf_model_weights_parent_dir=defaults[
                "VEC_INF_MODEL_WEIGHTS_PARENT_DIR"
            ],
        )
    )

    assert 'MODEL_NAME="Llama-3.2-3B-Instruct"' in config_text
    assert "VEC_INF_VLLM_IMAGE_PATH=" in config_text
    assert "VEC_INF_MODEL_WEIGHTS_PARENT_DIR=" in config_text
    assert "VEC_INF_CACHED_MODEL_CONFIG_PATH=" in config_text
    assert "VEC_INF_SGLANG_IMAGE_PATH" not in config_text
    assert "VEC_INF_IMAGE_PATH" not in config_text


def test_describe_setup_effects_explains_actions_in_words() -> None:
    """The setup preview should explain behavior in plain language."""
    defaults = build_defaults(remote_user="alice")
    description = describe_setup_effects(
        WizardConfig(
            remote_launch_host=defaults["REMOTE_LAUNCH_HOST"],
            remote_transfer_host=defaults["REMOTE_TRANSFER_HOST"],
            remote_internet_host=defaults["REMOTE_INTERNET_HOST"],
            remote_user=defaults["REMOTE_USER"],
            vec_inf_storage_user=defaults["VEC_INF_STORAGE_USER"],
            model_name=defaults["MODEL_NAME"],
            local_port=defaults["LOCAL_PORT"],
            auto_kill_stale_tunnel=defaults["AUTO_KILL_STALE_TUNNEL"],
            rsync_enabled=defaults["RSYNC_ENABLED"],
            rsync_src=defaults["RSYNC_SRC"],
            rsync_dest=defaults["RSYNC_DEST"],
            vec_inf_env=defaults["VEC_INF_ENV"],
            vec_inf_config_dir_remote=defaults["VEC_INF_CONFIG_DIR_REMOTE"],
            remote_work_dir=defaults["REMOTE_WORK_DIR"],
            remote_account=defaults["REMOTE_ACCOUNT"],
            remote_qos=defaults["REMOTE_QOS"],
            uv_sync_args=defaults["UV_SYNC_ARGS"],
            job_start_timeout=defaults["JOB_START_TIMEOUT"],
            server_ready_timeout=defaults["SERVER_READY_TIMEOUT"],
            vec_inf_vllm_image_path=defaults["VEC_INF_VLLM_IMAGE_PATH"],
            vec_inf_cached_model_config_path=defaults[
                "VEC_INF_CACHED_MODEL_CONFIG_PATH"
            ],
            vec_inf_model_weights_parent_dir=defaults[
                "VEC_INF_MODEL_WEIGHTS_PARENT_DIR"
            ],
        )
    )

    assert "create or reuse the remote checkout directory" in description
    assert "sync this local repo to `alice@transfer1.bsc.es:" in description
    assert "run `uv sync --frozen` in that checkout" in description
    assert "create or update the remote environment at" in description
    assert "./scripts/first_time_setup.sh --dry-run" in description


def test_describe_launch_effects_explains_actions_in_words() -> None:
    """The launch preview should explain the launch flow in plain language."""
    defaults = build_defaults(remote_user="alice")
    description = describe_launch_effects(
        WizardConfig(
            remote_launch_host=defaults["REMOTE_LAUNCH_HOST"],
            remote_transfer_host=defaults["REMOTE_TRANSFER_HOST"],
            remote_internet_host=defaults["REMOTE_INTERNET_HOST"],
            remote_user=defaults["REMOTE_USER"],
            vec_inf_storage_user=defaults["VEC_INF_STORAGE_USER"],
            model_name=defaults["MODEL_NAME"],
            local_port=defaults["LOCAL_PORT"],
            auto_kill_stale_tunnel=defaults["AUTO_KILL_STALE_TUNNEL"],
            rsync_enabled=defaults["RSYNC_ENABLED"],
            rsync_src=defaults["RSYNC_SRC"],
            rsync_dest=defaults["RSYNC_DEST"],
            vec_inf_env=defaults["VEC_INF_ENV"],
            vec_inf_config_dir_remote=defaults["VEC_INF_CONFIG_DIR_REMOTE"],
            remote_work_dir=defaults["REMOTE_WORK_DIR"],
            remote_account=defaults["REMOTE_ACCOUNT"],
            remote_qos=defaults["REMOTE_QOS"],
            uv_sync_args=defaults["UV_SYNC_ARGS"],
            job_start_timeout=defaults["JOB_START_TIMEOUT"],
            server_ready_timeout=defaults["SERVER_READY_TIMEOUT"],
            vec_inf_vllm_image_path=defaults["VEC_INF_VLLM_IMAGE_PATH"],
            vec_inf_cached_model_config_path=defaults[
                "VEC_INF_CACHED_MODEL_CONFIG_PATH"
            ],
            vec_inf_model_weights_parent_dir=defaults[
                "VEC_INF_MODEL_WEIGHTS_PARENT_DIR"
            ],
        )
    )

    assert "submit a vec-inf launch for `Llama-3.2-3B-Instruct`" in description
    assert "use the MN5 config from" in description
    assert "wait for the server to become ready" in description
    assert "open a local SSH tunnel on port `5678`" in description


def test_run_prompts_are_short_and_direct() -> None:
    """The execution prompts should stay concise once the explanation is already shown."""
    assert build_setup_run_prompt() == "Run first_time_setup.sh now?"
    assert build_launch_run_prompt("Llama-3.2-3B-Instruct") == (
        "Launch the smoke-test model `Llama-3.2-3B-Instruct` now?"
    )
