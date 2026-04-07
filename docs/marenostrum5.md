# MareNostrum5 (BSC) Setup

This page documents a local-to-MN5 workflow where `vec-inf` runs on MareNostrum5 and your local machine connects through an SSH tunnel.

## Quickstart

1. Clone and enter the repository:

```bash
git clone git@github.com:Cuena/vector-inference-mn5.git vector-inference-public
cd vector-inference-public
```

2. Run the setup wizard:

```bash
uv run vec-inf-mn5-wizard
```

The wizard writes `scripts/.launch.env`, explains each setting, and pre-fills defaults derived from your username and the current repo name. This setup path is vLLM-only for now. In the common case, you only need to set:

- `REMOTE_USER`: your BSC login username.
- `REMOTE_ACCOUNT`: defaults to `bsc70` in the wizard because that is the current team example in this repo.
- `VEC_INF_VLLM_IMAGE_PATH`: defaults to `/gpfs/scratch/bsc70/singularity/vllm_openai_0.18.0.sif`.
- `VEC_INF_MODEL_WEIGHTS_PARENT_DIR`: defaults to `/gpfs/scratch/bsc70/hpai/storage/projects/heka/models`.

Before running anything, the wizard prints the exact shell command it will execute and the effect that command will have. It can also immediately:
- run `./scripts/first_time_setup.sh`
- launch the lightweight 1-GPU smoke-test model `Llama-3.2-3B-Instruct`

3. If you skip the optional setup step in the wizard, run it manually:

```bash
./scripts/first_time_setup.sh
```

Dry-run preview (no remote writes):

```bash
./scripts/first_time_setup.sh --dry-run
```

4. Launch a model and open a local tunnel:

```bash
./scripts/launch_and_tunnel.sh Llama-3.2-3B-Instruct 5678
```

Server endpoint will be available at:

```text
http://localhost:5678/v1
```

To inspect active local tunnels, recover the exact served model id, and print canned validation calls:

```bash
./scripts/tunnel_tool.py status
./scripts/tunnel_tool.py show --port 5678
./scripts/tunnel_tool.py curls --port 5678
./scripts/tunnel_tui.py
```

## Manual configuration

If you prefer not to use the wizard:

1. Copy the example file and edit `scripts/.launch.env`.

```bash
cp scripts/.launch.env.example scripts/.launch.env
```

2. Set the key variables:

- `REMOTE_USER`, `RSYNC_DEST`, `VEC_INF_ENV`, `VEC_INF_CONFIG_DIR_REMOTE`, `REMOTE_WORK_DIR`
- `REMOTE_ACCOUNT`
- `VEC_INF_VLLM_IMAGE_PATH`
- `VEC_INF_MODEL_WEIGHTS_PARENT_DIR`

3. The tracked MN5 profile at [`vec_inf/config/marenostrum5/environment.yaml`](../vec_inf/config/marenostrum5/environment.yaml) reads those values from `.launch.env`, so most users do not need to modify the YAML directly.

4. Only edit [`vec_inf/config/marenostrum5/models.yaml`](../vec_inf/config/marenostrum5/models.yaml) when a specific model needs a different image path or runtime override.

## gpt-oss on MN5

For `gpt-oss-120b-0109`, make sure `VEC_INF_VLLM_IMAGE_PATH` points at the shared SIF (the wizard default is `vllm_openai_0.18.0.sif`). Then launch:

```bash
./scripts/launch_and_tunnel.sh gpt-oss-120b-0109 5678
```

## MN5 Config Profile

This repository includes an MN5 profile at:

- `vec_inf/config/marenostrum5/environment.yaml`
- `vec_inf/config/marenostrum5/models.yaml`

Helper scripts default to this profile by exporting `VEC_INF_CONFIG_DIR` on the remote side.

## Optional Repo Sync

To sync your local checkout to MN5 without launching:

```bash
./scripts/sync_repo.sh
```

## Existing Jobs: Print Tunnel Command

To print a tunnel command for an already-running job:

```bash
./scripts/print_tunnel_cmd.sh <SLURM_JOB_ID> [LOCAL_PORT]
```

## Notes

- The scripts enforce MN5 routing when using `*.bsc.es` hosts.
- Keep secrets and user-specific paths only in `scripts/.launch.env` (which is gitignored).
- Set explicit `RSYNC_DEST`/`VEC_INF_ENV`/`REMOTE_WORK_DIR` if you want side-by-side versions (for example, `-mn5`) without reusing old paths.
- The wizard-backed `.launch.env` values are exported into the tracked MN5 profile, which keeps the versioned YAML generic for the team.
