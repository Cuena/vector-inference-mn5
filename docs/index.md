# Vector Inference: Easy inference on Slurm clusters

This repository provides an easy-to-use solution to run inference servers on [Slurm](https://slurm.schedmd.com/overview.html)-managed computing clusters using open-source inference engines ([vLLM](https://docs.vllm.ai/en/stable/), [SGLang](https://docs.sglang.io/index.html)). **This package runs natively on the Vector Institute cluster environments**. To adapt to other environments, follow the instructions in [Installation](#installation).


**NOTE**: Supported models on Killarney are tracked in `MODEL_TRACKING.md` in the repository root.

## Installation

Choose the workflow that matches your setup:

```bash
pip install vec-inf
```

For a source checkout or the MN5 helper-script workflow:

```bash
uv sync --frozen
```

For a local backend environment without containers:

```bash
uv sync --group inference --extra vllm
```

If you'd like to use `vec-inf` on your own Slurm cluster, there are 3 configuration paths:

* Clone the repository and update the `environment.yaml` and `models.yaml` files in `vec_inf/config`. For a source checkout, prefer `uv sync` over `pip install .`.
* The package would try to look for cached configuration files in your environment before using the default configuration. The default cached configuration directory path points to `/model-weights/vec-inf-shared`, you would need to create an `environment.yaml` and a `models.yaml` following the format of the files in `vec_inf/config`.
* [OPTIONAL] The package also looks for an environment variable `VEC_INF_CONFIG_DIR`. You can put your `environment.yaml` and `models.yaml` in a directory of your choice and set `VEC_INF_CONFIG_DIR` to point to that location.


## MareNostrum5 (BSC)

Use the dedicated MN5 guide: [MareNostrum5 setup](marenostrum5.md).
