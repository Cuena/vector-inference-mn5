# Vector Inference: Easy inference on Slurm clusters

----------------------------------------------------

[![PyPI](https://img.shields.io/pypi/v/vec-inf)](https://pypi.org/project/vec-inf)
[![downloads](https://img.shields.io/pypi/dm/vec-inf)](https://pypistats.org/packages/vec-inf)
[![vLLM](https://img.shields.io/badge/vLLM-0.15.0-blue)](https://docs.vllm.ai/en/v0.15.0/)

This repository provides an easy-to-use solution to run inference servers on [Slurm](https://slurm.schedmd.com/overview.html)-managed computing clusters using open-source inference engines ([vLLM](https://docs.vllm.ai/en/v0.15.0/), [SGLang](https://docs.sglang.io/index.html)). **This package runs natively on the Vector Institute cluster environments**. To adapt to other environments, follow the instructions in [Installation](#installation).

**NOTE**: Supported models on Killarney are tracked [here](./MODEL_TRACKING.md)

## MareNostrum5 setup

This repository includes MN5-ready helper scripts and a cluster profile for public users.

### Fresh clone (step-by-step)

Use this flow if you just cloned the repo and want to avoid reusing paths from an older setup.

1. Clone and enter the repository:

```bash
git clone git@github.com:Cuena/vector-inference-mn5.git
cd vector-inference
```

2. Create your local launch config:

```bash
cp scripts/.launch.env.example scripts/.launch.env
```

3. Edit `scripts/.launch.env` with explicit v2-isolated paths (recommended):

```bash
REMOTE_USER="your_bsc_user"
RSYNC_ENABLED=1
RSYNC_DEST="/home/bsc/your_bsc_user/repos/vector-inference-v2"
VEC_INF_ENV="/home/bsc/your_bsc_user/repos/vector-inference-v2/.venv"
VEC_INF_CONFIG_DIR_REMOTE="/home/bsc/your_bsc_user/repos/vector-inference-v2/vec_inf/config/marenostrum5"
REMOTE_WORK_DIR="/home/bsc/your_bsc_user/repos/vector-inference-v2"
REMOTE_ACCOUNT="your_slurm_account"
VEC_INF_PROJECT_ROOT="/gpfs/projects/<project>/vec-inf"
VEC_INF_STORAGE_USER="your_storage_user"
```

4. Meaning of the key variables:
- `REMOTE_USER`: your BSC login username; used for SSH, rsync, and default `/home/bsc/<user>/...` paths.
- `RSYNC_DEST`: remote checkout path on MN5 where this repo is copied.
- `VEC_INF_ENV`: remote Python environment used to run `vec-inf launch`; normally `${RSYNC_DEST}/.venv`.
- `VEC_INF_CONFIG_DIR_REMOTE`: remote directory containing the MN5 `environment.yaml` and `models.yaml`.
- `REMOTE_WORK_DIR`: working directory passed to Slurm jobs; vec-inf uses it for runtime caches such as `.vec-inf-cache`.
- `REMOTE_ACCOUNT`: Slurm account to charge the job to.
- `VEC_INF_PROJECT_ROOT`: shared project directory containing `containers/`, `vec-inf-shared/`, and `models/`.
- `VEC_INF_STORAGE_USER`: optional override only when your `/gpfs/.../users/<owner>/...` storage owner differs from `REMOTE_USER`.

`models.yaml` resolves the compile YAMLs from this repo checkout automatically. Edit [`vec_inf/config/marenostrum5/environment.yaml`](vec_inf/config/marenostrum5/environment.yaml) only if your project layout differs from those defaults.

5. Run first-time remote setup (sync + remote `uv sync`):

```bash
./scripts/first_time_setup.sh
```

Optional dry-run preview (no remote writes):

```bash
./scripts/first_time_setup.sh --dry-run
```

6. Launch model and open local tunnel:

```bash
./scripts/launch_and_tunnel.sh Meta-Llama-3.1-8B-Instruct 5678
```

7. Use endpoint locally:

```text
http://localhost:5678/v1
```

### Why explicit paths matter

If `RSYNC_DEST`, `VEC_INF_ENV`, or `REMOTE_WORK_DIR` are left empty, scripts fall back to default locations and may reuse older data/environments. Setting explicit `-v2` paths keeps runs isolated. For the public MN5 profile, keeping `VEC_INF_PROJECT_ROOT` in `.launch.env` also avoids baking project-specific paths into tracked YAML files.

See [`docs/marenostrum5.md`](docs/marenostrum5.md) for full details.

## Installation
Choose the installation workflow that matches how you plan to run `vec-inf`:

If you only need the published package/CLI, install from PyPI:

```bash
pip install vec-inf
```

If you are working from this repo checkout or using the MN5 helper scripts, create the managed project environment:

```bash
uv sync --frozen
```

If you need a local backend environment instead of container images, install the vLLM backend plus the shared inference dependencies:

```bash
uv sync --group inference --extra vllm
```

We also provide [`vllm.Dockerfile`](vllm.Dockerfile) for container-based deployments.

If you'd like to use `vec-inf` on your own Slurm cluster, there are 3 configuration paths:
* Clone the repository and update the `environment.yaml` and `models.yaml` files in [`vec_inf/config`](vec_inf/config/). For a source checkout, prefer `uv sync` over `pip install .` so backend-specific dependency groups remain available.
* The package would try to look for cached configuration files in your environment before using the default configuration. The default cached configuration directory path points to `/model-weights/vec-inf-shared`, you would need to create an `environment.yaml` and a `models.yaml` following the format of these files in [`vec_inf/config`](vec_inf/config/).
* [OPTIONAL] The package can also look for an environment variable `VEC_INF_CONFIG_DIR`. You can put your `environment.yaml` and `models.yaml` in a directory of your choice and set `VEC_INF_CONFIG_DIR` to point to that location.

## Usage

Vector Inference provides 2 user interfaces, a CLI and an API

### CLI

The `launch` command allows users to deploy a model as a slurm job. If the job successfully launches, a URL endpoint is exposed for the user to send requests for inference.

We will use the Llama 3.1 model as example, to launch an OpenAI compatible inference server for Meta-Llama-3.1-8B-Instruct, run:

```bash
vec-inf launch Meta-Llama-3.1-8B-Instruct
```
You should see an output like the following:

<img width="720" alt="launch_image" src="./docs/assets/launch.png" />

**NOTE**: You can set the required fields in the environment configuration (`environment.yaml`), it's a mapping between required arguments and their corresponding environment variables. On the Vector **Killarney** Cluster environment, the required fields are:
  * `--account`, `-A`: The Slurm account, this argument can be set to default by setting environment variable `VEC_INF_ACCOUNT`.
  * `--work-dir`, `-D`: A working directory other than your home directory, this argument can be set to default by setting environment variable `VEC_INF_WORK_DIR`.

Models that are already supported by `vec-inf` are launched from the cached configuration (set in [`_slurm_vars.py`](vec_inf/client/_slurm_vars.py)) or the [default configuration](vec_inf/config/models.yaml). You can override these values by providing additional parameters. Use `vec-inf launch --help` to see the full list of parameters that can be overridden. You can also launch your own custom model as long as the model architecture is supported by the underlying inference engine. For detailed instructions on how to customize your model launch, check out the [`launch` command section in User Guide](https://vectorinstitute.github.io/vector-inference/latest/user_guide/#launch-command). During the launch process, relevant log files and scripts are written to a log directory (default `.vec-inf-logs` in your home directory), and a cache directory (`.vec-inf-cache`) is created in your working directory (defaults to your home directory if not specified or required) for torch compile cache.

#### Other commands

* `batch-launch`: Launch multiple model inference servers at once, currently ONLY single node models supported,
* `status`: Check the status of all `vec-inf` jobs, or a specific job by providing its job ID.
* `metrics`: Streams performance metrics to the console.
* `shutdown`: Shutdown a model by providing its Slurm job ID.
* `list`: List all available model names, or view the default/cached configuration of a specific model.
* `cleanup`: Remove old log directories, use `--help` to see the supported filters. Use `--dry-run` to preview what would be deleted.

For more details on the usage of these commands, refer to the [User Guide](https://vectorinstitute.github.io/vector-inference/user_guide/)

### API

Example:

```python
>>> from vec_inf.api import VecInfClient
>>> client = VecInfClient()
>>> # Assume VEC_INF_ACCOUNT and VEC_INF_WORK_DIR is set
>>> response = client.launch_model("Meta-Llama-3.1-8B-Instruct")
>>> job_id = response.slurm_job_id
>>> status = client.get_status(job_id)
>>> if status.status == ModelStatus.READY:
...     print(f"Model is ready at {status.base_url}")
>>> # Alternatively, use wait_until_ready which will either return a StatusResponse or throw a ServerError
>>> try:
>>>     status = wait_until_ready(job_id)
>>> except ServerError as e:
>>>     print(f"Model launch failed: {e}")
>>> client.shutdown_model(job_id)
```

For details on the usage of the API, refer to the [API Reference](https://vectorinstitute.github.io/vector-inference/api/)

## Check Job Configuration

With every model launch, a Slurm script will be generated dynamically based on the job and model configuration. Once the Slurm job is queued, the generated Slurm script will be moved to the log directory for reproducibility, located at `$log_dir/$model_family/$model_name.$slurm_job_id/$model_name.$slurm_job_id.slurm`. In the same directory you can also find a JSON file with the same name that captures the launch configuration, and will have an entry of server URL once the server is ready.

## Send inference requests

Once the inference server is ready, you can start sending in inference requests. We provide example scripts for sending inference requests in [`examples`](examples) folder. Make sure to update the model server URL and the model weights location in the scripts. For example, you can run `python examples/inference/llm/chat_completions.py`, and you should expect to see an output like the following:

```json
{
    "id":"chatcmpl-387c2579231948ffaf66cdda5439d3dc",
    "choices": [
        {
            "finish_reason":"stop",
            "index":0,
            "logprobs":null,
            "message": {
                "content":"Arrr, I be Captain Chatbeard, the scurviest chatbot on the seven seas! Ye be wantin' to know me identity, eh? Well, matey, I be a swashbucklin' AI, here to provide ye with answers and swappin' tales, savvy?",
                "role":"assistant",
                "function_call":null,
                "tool_calls":[],
                "reasoning_content":null
            },
            "stop_reason":null
        }
    ],
    "created":1742496683,
    "model":"Meta-Llama-3.1-8B-Instruct",
    "object":"chat.completion",
    "system_fingerprint":null,
    "usage": {
        "completion_tokens":66,
        "prompt_tokens":32,
        "total_tokens":98,
        "prompt_tokens_details":null
    },
    "prompt_logprobs":null
}

```
**NOTE**: Certain models don't adhere to OpenAI's chat template, e.g. Mistral family. For these models, you can either change your prompt to follow the model's default chat template or provide your own chat template via `--chat-template: TEMPLATE_PATH`.

## SSH tunnel from your local device
If you want to run inference from your local device, you can open a SSH tunnel to your cluster environment like the following:
```bash
ssh -L 8081:10.1.1.29:8081 username@v.vectorinstitute.ai -N
```
The example provided above is for the Vector Killarney cluster, change the variables accordingly for your environment. The IP address for the compute nodes on Killarney follow `10.1.1.XX` pattern, where `XX` is the GPU number (`kn029` -> `29` in this example).

## Reference
If you found Vector Inference useful in your research or applications, please cite using the following BibTeX template:
```
@software{vector_inference,
  title        = {Vector Inference: Efficient LLM inference on Slurm clusters},
  author       = {Wang, Marshall},
  organization = {Vector Institute},
  year         = {<YEAR_OF_RELEASE>},
  version      = {<VERSION_TAG>},
  url          = {https://github.com/Cuena/vector-inference-mn5}
}
```
