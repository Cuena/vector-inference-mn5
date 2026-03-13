"""Model configuration.

This module provides a Pydantic model for validating and managing model deployment
configurations, including hardware requirements and model specifications.
"""

import os
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vec_inf.client._slurm_vars import (
    DEFAULT_ARGS,
    MAX_CPUS_PER_TASK,
    MAX_GPUS_PER_NODE,
    MAX_NUM_NODES,
    MODEL_TYPES,
    PARTITION,
    QOS,
    RESOURCE_TYPE,
    expand_path_placeholders,
)


REPO_PATH_PREFIX = "@repo/"


def _repo_root() -> Path:
    """Return the repository root for repo-relative config assets."""
    override = os.getenv("VEC_INF_PROJECT_ROOT", "").strip()
    if override:
        return Path(expand_path_placeholders(override))
    return Path(__file__).resolve().parents[2]


def _resolve_special_path_tokens(value: str) -> str:
    """Resolve repo-relative tokens and simple env placeholders."""
    if value.startswith(REPO_PATH_PREFIX):
        return str(_repo_root() / value.removeprefix(REPO_PATH_PREFIX))
    if "$" in value or "~" in value:
        return expand_path_placeholders(value)
    return value


class ModelConfig(BaseModel):
    """Pydantic model for validating and managing model deployment configurations.

    A configuration class that handles validation and management of model deployment
    settings, including model specifications, hardware requirements, and runtime
    parameters.

    Parameters
    ----------
    model_name : str
        Name of the model, must be alphanumeric with allowed characters: '-', '_', '.'
    model_family : str
        Family/architecture of the model
    model_variant : str, optional
        Specific variant or version of the model family
    model_type : {'LLM', 'VLM', 'Text_Embedding', 'Reward_Modeling'}
        Type of model architecture
    gpus_per_node : int, optional
        Number of GPUs to use per node (1-MAX_GPUS_PER_NODE). Defaults to
        environment.yaml default_args.gpus_per_node when provided.
    num_nodes : int
        Number of nodes to use for deployment (1-MAX_NUM_NODES)
    cpus_per_task : int, optional
        Number of CPU cores per task (1-MAX_CPUS_PER_TASK)
    mem_per_node : str, optional
        Memory allocation per node in GB format (e.g., '32G')
    vocab_size : int
        Size of the model's vocabulary (1-1,000,000)
    account : str, optional
        Charge resources used by this job to specified account.
    work_dir : str, optional
        Set working directory for the batch job
    qos : Union[QOS, str], optional
        Quality of Service tier for job scheduling
    time : str, optional
        Time limit for the job in HH:MM:SS format
    partition : Union[PARTITION, str], optional
        Slurm partition for job scheduling
    resource_type : Union[RESOURCE_TYPE, str], optional
        Type of resource to request for the job
    venv : str, optional
        Virtual environment or container system to use
    log_dir : Path, optional
        Directory path for storing logs
    model_weights_parent_dir : Path, optional
        Base directory containing model weights
    engine: str, optional
        Inference engine to be used, supports 'vllm' and 'sglang'
    vllm_args : dict[str, Any], optional
        Additional arguments for vLLM engine configuration
    sglang_args : dict[str, Any], optional
        Additional arguments for SGLang engine configuration

    Notes
    -----
    All fields are validated using Pydantic's validation system. The model is
    configured to be immutable (frozen) and forbids extra fields.
    """

    model_config = ConfigDict(
        extra="ignore", str_strip_whitespace=True, validate_default=True, frozen=True
    )

    model_name: str = Field(..., min_length=3, pattern=r"^[a-zA-Z0-9\-_\.]+$")
    model_family: str = Field(..., min_length=2)
    model_variant: Optional[str] = Field(
        default=None, description="Specific variant/version of the model family"
    )
    model_type: MODEL_TYPES = Field(..., description="Type of model architecture")
    gpus_per_node: int = Field(
        default=int(DEFAULT_ARGS["gpus_per_node"])
        if DEFAULT_ARGS.get("gpus_per_node")
        else ...,
        gt=0,
        le=MAX_GPUS_PER_NODE,
        description="GPUs per node",
    )
    num_nodes: int = Field(..., gt=0, le=MAX_NUM_NODES, description="Number of nodes")
    ntasks: Optional[int] = Field(
        default=int(DEFAULT_ARGS["ntasks"]) if DEFAULT_ARGS.get("ntasks") else None,
        gt=0,
        description="Number of tasks",
    )
    ntasks_per_node: Optional[int] = Field(
        default=int(DEFAULT_ARGS["ntasks_per_node"])
        if DEFAULT_ARGS.get("ntasks_per_node")
        else None,
        gt=0,
        description="Number of tasks per node",
    )
    cpus_per_task: int = Field(
        default=int(DEFAULT_ARGS["cpus_per_task"]),
        gt=0,
        le=MAX_CPUS_PER_TASK,
        description="CPUs per task",
    )
    mem_per_node: Optional[str] = Field(
        default=DEFAULT_ARGS.get("mem_per_node"),
        pattern=r"^\d{1,4}G$",
        description="Memory per node",
    )
    vocab_size: int = Field(..., gt=0, le=1_000_000)
    account: Optional[str] = Field(
        default=None, description="Account name for job scheduling"
    )
    work_dir: Optional[str] = Field(
        default=DEFAULT_ARGS["work_dir"] if DEFAULT_ARGS.get("work_dir") else None,
        description="Working directory for the job",
    )
    qos: Optional[Union[QOS, str]] = Field(
        default=DEFAULT_ARGS["qos"] if DEFAULT_ARGS["qos"] != "" else None,
        description="Quality of Service tier",
    )
    time: str = Field(
        default=DEFAULT_ARGS["time"],
        pattern=r"^\d{2}:\d{2}:\d{2}$",
        description="HH:MM:SS time limit",
    )
    partition: Optional[Union[PARTITION, str]] = Field(
        default=DEFAULT_ARGS["partition"] if DEFAULT_ARGS["partition"] != "" else None,
        description="GPU partition type",
    )
    resource_type: Optional[Union[RESOURCE_TYPE, str]] = Field(
        default=DEFAULT_ARGS["resource_type"]
        if DEFAULT_ARGS["resource_type"] != ""
        else None,
        description="Resource type",
    )
    exclude: Optional[str] = Field(
        default=DEFAULT_ARGS["exclude"],
        description="Exclude certain nodes from the resources granted to the job",
    )
    nodelist: Optional[str] = Field(
        default=DEFAULT_ARGS["nodelist"],
        description="Request a specific list of nodes for deployment",
    )
    bind: Optional[str] = Field(
        default=DEFAULT_ARGS["bind"],
        description="Additional binds for the container",
    )
    venv: str = Field(
        default=DEFAULT_ARGS["venv"],
        description="Virtual environment/container system",
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Container image path override when using container runtime",
    )
    log_dir: Path = Field(
        default=Path(DEFAULT_ARGS["log_dir"]),
        description="Log directory path",
    )
    model_weights_parent_dir: Path = Field(
        default=Path(DEFAULT_ARGS["model_weights_parent_dir"]),
        description="Base directory for model weights",
    )
    model_weights_dir_name: Optional[str] = Field(
        default=None,
        description="Optional override for the model weights directory name.",
    )
    engine: Optional[str] = Field(
        default="vllm",
        description="Inference engine to be used, supports 'vllm' and 'sglang'",
    )
    vllm_args: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="vLLM engine arguments"
    )
    sglang_args: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="SGLang engine arguments"
    )
    env: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="Environment variables to be set"
    )
    unset_env_vars: Optional[list[str]] = Field(
        default_factory=list,
        description="Environment variable names to unset in generated sbatch script",
    )
    cuda_compat_shim: Optional[bool] = Field(
        default=None,
        description=(
            "Enable host-side CUDA compat shim bind for container runs."
        ),
    )
    cuda_compat_shim_source: Optional[str] = Field(
        default=None,
        description="Source path for libcuda.so.1 when cuda_compat_shim is enabled.",
    )

    @field_validator("log_dir", "model_weights_parent_dir", mode="before")
    @classmethod
    def _expand_path_fields(cls, value: Any) -> Any:
        """Expand environment placeholders in path fields."""
        if value is None:
            return value
        return Path(_resolve_special_path_tokens(str(value)))

    @field_validator(
        "work_dir", "venv", "image_path", "cuda_compat_shim_source", mode="before"
    )
    @classmethod
    def _expand_string_paths(cls, value: Any) -> Any:
        """Expand environment placeholders in string-based path fields."""
        if value is None:
            return value
        text = str(value)
        return _resolve_special_path_tokens(text)

    @field_validator("vllm_args", "sglang_args", mode="before")
    @classmethod
    def _expand_engine_arg_paths(cls, value: Any) -> Any:
        """Resolve special path tokens inside engine argument dictionaries."""
        if not isinstance(value, dict):
            return value

        resolved: dict[str, Any] = {}
        for key, raw_value in value.items():
            if isinstance(raw_value, str):
                resolved[key] = _resolve_special_path_tokens(raw_value)
            else:
                resolved[key] = raw_value
        return resolved
