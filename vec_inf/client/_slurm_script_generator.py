"""Class for generating Slurm scripts to run inference servers.

This module provides functionality to generate Slurm scripts for running inference
servers in both single-node and multi-node configurations.
"""

import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

from vec_inf.client._client_vars import SLURM_JOB_CONFIG_ARGS
from vec_inf.client._slurm_templates import (
    BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE,
    BATCH_SLURM_SCRIPT_TEMPLATE,
    SLURM_SCRIPT_TEMPLATE,
)
from vec_inf.client._slurm_vars import (
    CONTAINER_MODULE_NAME,
    CUDA_COMPAT_SHIM_DEFAULT,
    IMAGE_PATH,
)


class SlurmScriptGenerator:
    """A class to generate Slurm scripts for running inference servers.

    This class handles the generation of Slurm scripts for both single-node and
    multi-node configurations, supporting different virtualization environments
    (venv or singularity/apptainer).

    Parameters
    ----------
        params : dict[str, Any]
            Configuration parameters for the Slurm script.
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.engine = params.get("engine", "vllm")
        self.is_multinode = int(self.params["num_nodes"]) > 1
        self.use_container = self.params["venv"] == CONTAINER_MODULE_NAME
        self.additional_binds = (
            f",{self.params['bind']}" if self.params.get("bind") else ""
        )
        raw_cuda_shim = self.params.get("cuda_compat_shim")
        if raw_cuda_shim is None:
            self.cuda_compat_shim_enabled = bool(CUDA_COMPAT_SHIM_DEFAULT)
        else:
            self.cuda_compat_shim_enabled = self._parse_bool(raw_cuda_shim)
        self.cuda_compat_shim_source = str(
            self.params.get("cuda_compat_shim_source")
            or "/usr/local/cuda-12.9/compat/lib.real/libcuda.so.1"
        )
        model_weights_dir_name = str(
            self.params.get("model_weights_dir_name") or self.params["model_name"]
        )
        self.model_weights_path = str(
            Path(self.params["model_weights_parent_dir"], model_weights_dir_name)
        )
        self.env_str = self._generate_env_str()

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, int):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _cuda_compat_shim_bind(self) -> str:
        if not self.use_container or not self.cuda_compat_shim_enabled:
            return ""
        bind_value = str(self.params.get("bind") or "")
        if ":/usr/local/cuda/compat/lib" in bind_value:
            return ""
        return ",$HOME/cuda_shim:/usr/local/cuda/compat/lib"

    def _cuda_compat_shim_setup_lines(self) -> list[str]:
        if not self.use_container or not self.cuda_compat_shim_enabled:
            return []
        return [
            'mkdir -p "$HOME/cuda_shim"',
            f'cuda_shim_src="{self.cuda_compat_shim_source}"',
            'cp -f "$cuda_shim_src" "$HOME/cuda_shim/libcuda.so.1" || true',
            'ln -sf libcuda.so.1 "$HOME/cuda_shim/libcuda.so" || true',
        ]

    def _container_env_reset_lines(self) -> list[str]:
        if not self.use_container:
            return []
        return [
            "unset LD_PRELOAD",
            "unset SINGULARITYENV_LD_PRELOAD",
            "unset APPTAINERENV_LD_PRELOAD",
            "unset SINGULARITYENV_LD_LIBRARY_PATH",
            "unset APPTAINERENV_LD_LIBRARY_PATH",
        ]

    def _generate_env_str(self) -> str:
        """Generate the environment variables string for the Slurm script.

        Returns
        -------
        str
            Formatted env vars string for container or shell export commands.
        """
        env_dict: dict[str, str] = self.params.get("env", {})

        if not env_dict:
            return ""

        if self.use_container:
            # Format for container: --env KEY1=VAL1,KEY2=VAL2
            env_pairs = [f"{key}={val}" for key, val in env_dict.items()]
            env_pairs.extend(
                [
                    "VLLM_HOST_IP=${VLLM_HOST_IP:-}",
                    "HOST_IP=${HOST_IP:-}",
                    "RAY_ADDRESS=${RAY_ADDRESS:-}",
                    "RAY_NODE_IP_ADDRESS=${RAY_NODE_IP_ADDRESS:-}",
                    "RAY_OVERRIDE_NODE_IP_ADDRESS=${RAY_OVERRIDE_NODE_IP_ADDRESS:-}",
                    "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-}",
                    "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-}",
                    "NCCL_DEBUG=INFO",
                ]
            )
            return f"--env {','.join(env_pairs)}"
        # Format for shell: export KEY1=VAL1\nexport KEY2=VAL2
        export_lines = [f"export {key}={val}" for key, val in env_dict.items()]
        return "\n".join(export_lines)

    def _unset_env_lines(self) -> list[str]:
        """Generate shell `unset` commands for configured environment variables."""
        raw_unset_vars = self.params.get("unset_env_vars")
        if not raw_unset_vars:
            return []

        if isinstance(raw_unset_vars, str):
            candidates = [raw_unset_vars]
        elif isinstance(raw_unset_vars, list):
            candidates = raw_unset_vars
        else:
            return []

        unset_lines: list[str] = []
        for candidate in candidates:
            var_name = str(candidate).strip()
            if not var_name:
                continue
            if var_name[0].isdigit():
                continue
            if not all(ch.isalnum() or ch == "_" for ch in var_name):
                continue
            unset_lines.append(f"unset {var_name}")

        return unset_lines

    @staticmethod
    def _format_engine_arg_value(value: Any) -> str:
        """Format an engine argument value for safe use in generated bash."""
        return shlex.quote(str(value))

    def _generate_script_content(self) -> str:
        """Generate the complete Slurm script content.

        Returns
        -------
        str
            The complete Slurm script as a string.
        """
        script_content = []
        script_content.append(self._generate_shebang())
        script_content.append(self._generate_server_setup())
        script_content.append(self._generate_launch_cmd())
        return "\n".join(script_content)

    def _generate_shebang(self) -> str:
        """Generate the Slurm script shebang with job specifications.

        Returns
        -------
        str
            Slurm shebang containing job specifications.
        """
        shebang = [SLURM_SCRIPT_TEMPLATE["shebang"]["base"]]
        for arg, value in SLURM_JOB_CONFIG_ARGS.items():
            if self.params.get(value):
                shebang.append(f"#SBATCH --{arg}={self.params[value]}")
            if value == "model_name":
                shebang[-1] += "-vec-inf"
        if self.is_multinode:
            shebang += SLURM_SCRIPT_TEMPLATE["shebang"]["multinode"]

        # Provide sensible defaults for task layout, while still allowing overrides.
        if (
            self.is_multinode
            and not self.params.get("ntasks")
            and not self.params.get("ntasks_per_node")
        ):
            shebang.append("#SBATCH --ntasks-per-node=1")
        elif (
            not self.is_multinode
            and not self.params.get("ntasks")
            and not self.params.get("ntasks_per_node")
        ):
            shebang.append("#SBATCH --ntasks=1")

        return "\n".join(shebang)

    def _generate_server_setup(self) -> str:
        """Generate the server initialization script.

        Creates the script section that handles server setup, including Ray
        initialization for multi-node setups and port configuration.

        Returns
        -------
        str
            Server initialization script content.
        """
        server_script = ["\n"]
        unset_env_lines = self._unset_env_lines()

        if self.use_container:
            work_dir = str(self.params.get("work_dir", str(Path.home())))
            server_script.append("\n".join(SLURM_SCRIPT_TEMPLATE["container_setup"]))
            cuda_shim_setup = self._cuda_compat_shim_setup_lines()
            if cuda_shim_setup:
                server_script.append("\n".join(cuda_shim_setup))
            container_env_reset = self._container_env_reset_lines()
            if container_env_reset:
                server_script.append("\n".join(container_env_reset))
            if unset_env_lines:
                server_script.append("\n".join(unset_env_lines))
            # Ensure bind source exists on each compute node before container launch.
            server_script.append(f'mkdir -p "{work_dir}/.vec-inf-cache"')
            server_script.append(
                SLURM_SCRIPT_TEMPLATE["bind_path"].format(
                    work_dir=work_dir,
                    model_weights_path=self.model_weights_path,
                    additional_binds=self.additional_binds,
                    cuda_compat_shim_bind=self._cuda_compat_shim_bind(),
                )
            )
        else:
            server_script.append(
                SLURM_SCRIPT_TEMPLATE["activate_venv"].format(venv=self.params["venv"])
            )
            if unset_env_lines:
                server_script.append("\n".join(unset_env_lines))
            server_script.append(self.env_str)
        server_script.append(
            SLURM_SCRIPT_TEMPLATE["imports"].format(src_dir=self.params["src_dir"])
        )

        if self.is_multinode and self.engine == "vllm":
            template_key = (
                "multinode_vllm_common" if self.use_container else "multinode_vllm"
            )
            server_setup_str = "\n".join(
                SLURM_SCRIPT_TEMPLATE["server_setup"][template_key]
            ).format(gpus_per_node=self.params["gpus_per_node"])
            if self.use_container:
                image_path = self.params.get("image_path") or IMAGE_PATH[self.engine]
                server_setup_str = server_setup_str.replace(
                    "CONTAINER_PLACEHOLDER",
                    SLURM_SCRIPT_TEMPLATE["container_command"].format(
                        env_str=self.env_str,
                        image_path=image_path,
                    ),
                )
            else:
                server_setup_str = server_setup_str.replace(
                    "CONTAINER_PLACEHOLDER",
                    "\\",
                )
        elif self.is_multinode and self.engine == "sglang":
            server_setup_str = "\n".join(
                SLURM_SCRIPT_TEMPLATE["server_setup"]["multinode_sglang"]
            )
        else:
            server_setup_str = "\n".join(
                SLURM_SCRIPT_TEMPLATE["server_setup"]["single_node"]
            )
        server_script.append(server_setup_str)
        server_script.append("\n".join(SLURM_SCRIPT_TEMPLATE["find_server_port"]))
        server_script.append(
            "\n".join(SLURM_SCRIPT_TEMPLATE["write_to_json"]).format(
                log_dir=self.params["log_dir"], model_name=self.params["model_name"]
            )
        )
        return "\n".join(server_script)

    def _generate_launch_cmd(self) -> str:
        """Generate the inference server launch command.

        Creates the command to launch the inference server, handling different
        virtualization environments (venv or singularity/apptainer).

        Returns
        -------
        str
            Server launch command.
        """
        if self.is_multinode and self.engine == "sglang":
            return self._generate_multinode_sglang_launch_cmd()

        if self.is_multinode and self.engine == "vllm" and self.use_container:
            return self._generate_multinode_vllm_container_launch_cmd()

        if self.engine == "vllm":
            return self._generate_vllm_launch_cmd()

        launch_cmd = ["\n"]
        if self.use_container:
            image_path = self.params.get("image_path") or IMAGE_PATH[self.engine]
            launch_cmd.append(
                SLURM_SCRIPT_TEMPLATE["container_command"].format(
                    env_str=self.env_str,
                    image_path=image_path,
                )
            )

        launch_cmd.append(
            "\n".join(SLURM_SCRIPT_TEMPLATE["launch_cmd"][self.engine]).format(  # type: ignore[literal-required]
                model_weights_path=self.model_weights_path,
                model_name=self.params["model_name"],
            )
        )

        for arg, value in self.params["engine_args"].items():
            if isinstance(value, bool):
                launch_cmd.append(f"    {arg} \\")
            else:
                launch_cmd.append(
                    f"    {arg} {self._format_engine_arg_value(value)} \\"
                )

        # A known bug in vLLM requires setting backend to ray for multi-node
        # Remove this when the bug is fixed
        if (
            self.is_multinode
            and "--distributed-executor-backend" not in self.params["engine_args"]
        ):
            launch_cmd.append("    --distributed-executor-backend ray \\")

        return "\n".join(launch_cmd).rstrip(" \\")

    def _generate_vllm_launch_cmd(self) -> str:
        """Generate the vLLM launch command with optional API key auth.

        Returns
        -------
        str
            vLLM server launch command.
        """
        launch_cmd = ["\n"]
        if self.use_container:
            image_path = self.params.get("image_path") or IMAGE_PATH[self.engine]
            launch_cmd.append(
                SLURM_SCRIPT_TEMPLATE["container_command"].format(
                    env_str=self.env_str,
                    image_path=image_path,
                )
            )

        launch_cmd.append(
            "\n".join(SLURM_SCRIPT_TEMPLATE["launch_cmd"]["vllm"]).format(
                model_weights_path=self.model_weights_path,
                model_name=self.params["model_name"],
            )
        )

        for arg, value in self.params["engine_args"].items():
            if isinstance(value, bool):
                launch_cmd.append(f"    {arg} \\")
            else:
                launch_cmd.append(
                    f"    {arg} {self._format_engine_arg_value(value)} \\"
                )

        # If a key is already exported in the cluster job environment, enable
        # vLLM's bearer-token auth without writing the secret to generated files.
        launch_cmd.append(
            '    ${VEC_INF_API_KEY:+--api-key} ${VEC_INF_API_KEY:+"$VEC_INF_API_KEY"} \\'
        )

        # A known bug in vLLM requires setting backend to ray for multi-node
        # Remove this when the bug is fixed
        if (
            self.is_multinode
            and "--distributed-executor-backend" not in self.params["engine_args"]
        ):
            launch_cmd.append("    --distributed-executor-backend ray \\")

        return "\n".join(launch_cmd).rstrip(" \\")

    def _generate_multinode_vllm_container_launch_cmd(self) -> str:
        """Generate a multi-node vLLM launch with the driver on the Ray head.

        Ray 2.55 resolves the driver's local node by discovering a local raylet
        process. Start Ray head and vLLM inside the same head-node container so
        that discovery works while keeping ``--containall`` enabled.
        """
        image_path = self.params.get("image_path") or IMAGE_PATH[self.engine]
        container_cmd = SLURM_SCRIPT_TEMPLATE["container_command"].format(
            env_str=self.env_str,
            image_path=image_path,
        )
        work_dir = str(self.params.get("work_dir", str(Path.home())))
        runtime_dir = f"{work_dir}/.vec-inf-runtime-$SLURM_JOB_ID"
        head_script = f"{runtime_dir}/head_vllm.sh"

        vllm_lines = self._vllm_serve_command_lines()

        launch_cmd = [
            "\n# Start Ray head and vLLM in the same head-node container",
            f'vec_inf_runtime_dir="{runtime_dir}"',
            'mkdir -p "$vec_inf_runtime_dir"',
            'head_ready_file="$vec_inf_runtime_dir/ray-head-ready"',
            'workers_started_file="$vec_inf_runtime_dir/ray-workers-started"',
            'rm -f "$head_ready_file" "$workers_started_file"',
            f'head_launch_script="{head_script}"',
            'cat > "$head_launch_script" <<\'VEC_INF_HEAD_SCRIPT\'',
            "#!/bin/bash",
            "set -euo pipefail",
            'head_node_ip="$1"',
            'head_node_port="$2"',
            'cpus_per_task="$3"',
            'head_ready_file="$4"',
            'workers_started_file="$5"',
            'server_port_number="$6"',
            'export VLLM_HOST_IP="$head_node_ip"',
            'export HOST_IP="$head_node_ip"',
            'export RAY_NODE_IP_ADDRESS="$head_node_ip"',
            'export RAY_OVERRIDE_NODE_IP_ADDRESS="$head_node_ip"',
            'export RAY_ADDRESS="$head_node_ip:$head_node_port"',
            'echo "Starting Ray HEAD at $head_node_ip:$head_node_port"',
            (
                'ray start --head --node-ip-address="$head_node_ip" '
                '--port="$head_node_port" --num-cpus "$cpus_per_task" '
                f'--num-gpus {self.params["gpus_per_node"]}'
            ),
            'touch "$head_ready_file"',
            'echo "Waiting for Ray workers to be started..."',
            'while [ ! -f "$workers_started_file" ]; do sleep 1; done',
            'echo "Starting vLLM on Ray head"',
            *vllm_lines,
            "VEC_INF_HEAD_SCRIPT",
            'chmod +x "$head_launch_script"',
            'echo "Starting HEAD and vLLM at $head_node"',
            'srun --nodes=1 --ntasks=1 -w "$head_node" \\',
            f"    {container_cmd}",
            (
                '    bash "$head_launch_script" "$head_node_ip" "$head_node_port" '
                '"$SLURM_CPUS_PER_TASK" "$head_ready_file" '
                '"$workers_started_file" "$server_port_number" &'
            ),
            "head_vllm_pid=$!",
            'for _ in $(seq 1 120); do',
            '    if [ -f "$head_ready_file" ]; then break; fi',
            "    sleep 1",
            "done",
            'if [ ! -f "$head_ready_file" ]; then',
            '    echo "[ERROR] Ray head did not become ready." >&2',
            '    exit 1',
            "fi",
            "\n# Start Ray worker nodes",
            "worker_num=$((SLURM_JOB_NUM_NODES - 1))",
            "for ((i = 1; i <= worker_num; i++)); do",
            "    node_i=${nodes_array[$i]}",
            '    echo "Starting WORKER $i at $node_i"',
            '    node_ip=$(srun --nodes=1 --ntasks=1 -w "$node_i" hostname --ip-address)',
            '    echo "Node ip $node_ip"',
            '    srun --nodes=1 --ntasks=1 -w "$node_i" \\',
            f"        {container_cmd}",
            '        bash -c "export VLLM_HOST_IP=$node_ip && export HOST_IP=$node_ip && \\',
            '        export RAY_NODE_IP_ADDRESS=$node_ip && export RAY_OVERRIDE_NODE_IP_ADDRESS=$node_ip && \\',
            '        ray start --address "$ray_head" \\',
            '        --node-ip-address="$node_ip" \\',
            f'        --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus {self.params["gpus_per_node"]} --block" &',
            "    sleep 5",
            "done",
            'touch "$workers_started_file"',
            'wait "$head_vllm_pid"',
        ]

        return "\n".join(launch_cmd)

    def _vllm_serve_command_lines(self) -> list[str]:
        """Return shell lines for the vLLM serve command."""
        lines = [
            f"vllm serve {self._format_engine_arg_value(self.model_weights_path)} \\",
            (
                "    --served-model-name "
                f"{self._format_engine_arg_value(self.params['model_name'])} \\"
            ),
            '    --host "$head_node_ip" \\',
            '    --port "$server_port_number" \\',
        ]

        for arg, value in self.params["engine_args"].items():
            if isinstance(value, bool):
                lines.append(f"    {arg} \\")
            else:
                lines.append(
                    f"    {arg} {self._format_engine_arg_value(value)} \\"
                )

        lines.append(
            '    ${VEC_INF_API_KEY:+--api-key} ${VEC_INF_API_KEY:+"$VEC_INF_API_KEY"} \\'
        )

        if "--distributed-executor-backend" not in self.params["engine_args"]:
            lines.append("    --distributed-executor-backend ray \\")

        lines[-1] = lines[-1].rstrip(" \\")
        return lines

    def _generate_multinode_sglang_launch_cmd(self) -> str:
        """Generate the launch command for multi-node sglang setup.

        Returns
        -------
        str
            Multi-node sglang launch command.
        """
        launch_cmd = "\n" + "\n".join(
            SLURM_SCRIPT_TEMPLATE["launch_cmd"]["sglang_multinode"]
        ).format(
            num_nodes=self.params["num_nodes"],
            model_weights_path=self.model_weights_path,
            model_name=self.params["model_name"],
        )

        container_placeholder = "\\"
        if self.use_container:
            container_placeholder = SLURM_SCRIPT_TEMPLATE["container_command"].format(
                env_str=self.env_str,
                image_path=IMAGE_PATH[self.engine],
            )
        launch_cmd = launch_cmd.replace(
            "CONTAINER_PLACEHOLDER",
            container_placeholder,
        )

        engine_arg_str = ""
        for arg, value in self.params["engine_args"].items():
            if isinstance(value, bool):
                engine_arg_str += f"            {arg} \\\n"
            else:
                engine_arg_str += (
                    f"            {arg} {self._format_engine_arg_value(value)} \\\n"
                )

        return launch_cmd.replace(
            "SGLANG_ARGS_PLACEHOLDER", engine_arg_str.rstrip("\\\n")
        )

    def write_to_log_dir(self) -> Path:
        """Write the generated Slurm script to the log directory.

        Creates a timestamped script file in the configured log directory.

        Returns
        -------
        Path
            Path to the generated Slurm script file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path: Path = (
            Path(self.params["log_dir"])
            / f"launch_{self.params['model_name']}_{timestamp}.sbatch"
        )

        content = self._generate_script_content()
        script_path.write_text(content)
        return script_path


class BatchSlurmScriptGenerator:
    """A class to generate Slurm scripts for batch mode.

    This class handles the generation of Slurm scripts for batch mode, which
    launches multiple inference servers with different configurations in parallel.
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.script_paths: list[Path] = []
        self.use_container = self.params["venv"] == CONTAINER_MODULE_NAME
        for model_name in self.params["models"]:
            self.params["models"][model_name]["additional_binds"] = (
                f",{self.params['models'][model_name]['bind']}"
                if self.params["models"][model_name].get("bind")
                else ""
            )
            model_weights_dir_name = str(
                self.params["models"][model_name].get("model_weights_dir_name")
                or model_name
            )
            self.params["models"][model_name]["model_weights_path"] = str(
                Path(
                    self.params["models"][model_name]["model_weights_parent_dir"],
                    model_weights_dir_name,
                )
            )

    def _write_to_log_dir(self, script_content: list[str], script_name: str) -> Path:
        """Write the generated Slurm script to the log directory.

        Returns
        -------
        Path
            The Path object to the generated Slurm script file.
        """
        script_path = Path(self.params["log_dir"]) / script_name
        script_path.touch(exist_ok=True)
        script_path.write_text("\n".join(script_content))
        return script_path

    def _generate_model_launch_script(self, model_name: str) -> Path:
        """Generate the bash script for launching individual inference servers.

        Parameters
        ----------
        model_name : str
            The name of the model to launch.

        Returns
        -------
        Path
            The bash script path for launching the inference server.
        """
        # Generate the bash script content
        script_content = []
        model_params = self.params["models"][model_name]
        script_content.append(BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["shebang"])
        work_dir = str(self.params.get("work_dir", str(Path.home())))
        if self.use_container:
            script_content.append(BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["container_setup"])
            # Ensure bind source exists on each compute node before container launch.
            script_content.append(f'mkdir -p "{work_dir}/.vec-inf-cache"')
        script_content.append(
            BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["bind_path"].format(
                work_dir=work_dir,
                model_weights_path=model_params["model_weights_path"],
                additional_binds=model_params["additional_binds"],
            )
        )
        script_content.append(
            "\n".join(
                BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["server_address_setup"]
            ).format(src_dir=self.params["src_dir"])
        )
        script_content.append(
            "\n".join(BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["write_to_json"]).format(
                het_group_id=model_params["het_group_id"],
                log_dir=self.params["log_dir"],
                slurm_job_name=self.params["slurm_job_name"],
                model_name=model_name,
            )
        )
        if self.use_container:
            script_content.append(
                BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["container_command"].format(
                    image_path=IMAGE_PATH[model_params["engine"]],
                )
            )
        script_content.extend(self._generate_batch_model_launch_lines(model_name))
        # Write the bash script to the log directory
        launch_script_path = self._write_to_log_dir(
            script_content, f"launch_{model_name}.sh"
        )
        self.script_paths.append(launch_script_path)
        return launch_script_path

    def _generate_batch_model_launch_lines(self, model_name: str) -> list[str]:
        """Generate launch lines for an individual batch model script."""
        model_params = self.params["models"][model_name]
        launch_lines = [
            "\n".join(
                BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["launch_cmd"][model_params["engine"]]
            ).format(
                model_weights_path=model_params["model_weights_path"],
                model_name=model_name,
            )
        ]

        for arg, value in model_params["engine_args"].items():
            if isinstance(value, bool):
                launch_lines.append(f"    {arg} \\")
            else:
                launch_lines.append(
                    f"    {arg} {SlurmScriptGenerator._format_engine_arg_value(value)} \\"
                )

        if model_params["engine"] == "vllm":
            launch_lines.append(
                '    ${VEC_INF_API_KEY:+--api-key} ${VEC_INF_API_KEY:+"$VEC_INF_API_KEY"} \\'
            )

        launch_lines[-1] = launch_lines[-1].rstrip(" \\")
        return launch_lines

    def _generate_batch_slurm_script_shebang(self) -> str:
        """Generate the shebang for batch mode Slurm script.

        Returns
        -------
        str
            The shebang for batch mode Slurm script.
        """
        shebang = [BATCH_SLURM_SCRIPT_TEMPLATE["shebang"]]

        for arg, value in SLURM_JOB_CONFIG_ARGS.items():
            if self.params.get(value):
                shebang.append(f"#SBATCH --{arg}={self.params[value]}")
        shebang.append("#SBATCH --ntasks=1")
        shebang.append("\n")

        for model_name in self.params["models"]:
            shebang.append(f"# ===== Resource group for {model_name} =====")
            for arg, value in SLURM_JOB_CONFIG_ARGS.items():
                model_params = self.params["models"][model_name]
                if model_params.get(value) and value not in ["out_file", "err_file"]:
                    shebang.append(f"#SBATCH --{arg}={model_params[value]}")
                if value == "model_name":
                    shebang[-1] += "-vec-inf"
            shebang[-1] += "\n"
            shebang.append(BATCH_SLURM_SCRIPT_TEMPLATE["hetjob"])
        # Remove the last hetjob line
        shebang.pop()
        return "\n".join(shebang)

    def generate_batch_slurm_script(self) -> Path:
        """Generate the Slurm script for launching multiple inference servers in batch.

        Returns
        -------
        Path
            The Slurm script for launching multiple inference servers in batch.
        """
        script_content = []

        script_content.append(self._generate_batch_slurm_script_shebang())

        for model_name in self.params["models"]:
            model_params = self.params["models"][model_name]
            script_content.append(f"# ===== Launching {model_name} =====")
            launch_script_path = str(self._generate_model_launch_script(model_name))
            script_content.append(
                BATCH_SLURM_SCRIPT_TEMPLATE["permission_update"].format(
                    script_name=launch_script_path
                )
            )
            script_content.append(
                "\n".join(BATCH_SLURM_SCRIPT_TEMPLATE["launch_model_scripts"]).format(
                    het_group_id=model_params["het_group_id"],
                    out_file=model_params["out_file"],
                    err_file=model_params["err_file"],
                    script_name=launch_script_path,
                )
            )
        script_content.append("wait")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_name = f"{self.params['slurm_job_name']}_{timestamp}.sbatch"
        return self._write_to_log_dir(script_content, script_name)
