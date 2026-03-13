# MareNostrum5 (BSC) Setup

This page documents a local-to-MN5 workflow where `vec-inf` runs on MareNostrum5 and your local machine connects through an SSH tunnel.

## Quickstart

1. Configure helper scripts:

```bash
cp scripts/.launch.env.example scripts/.launch.env
# edit scripts/.launch.env
```

2. In `scripts/.launch.env`, set explicit isolated paths (recommended):

```bash
REMOTE_USER="your_bsc_user"
RSYNC_ENABLED=1
RSYNC_DEST="/home/bsc/your_bsc_user/repos/vector-inference-v2"
VEC_INF_ENV="/home/bsc/your_bsc_user/repos/vector-inference-v2/.venv"
VEC_INF_CONFIG_DIR_REMOTE="/home/bsc/your_bsc_user/repos/vector-inference-v2/vec_inf/config/marenostrum5"
REMOTE_WORK_DIR="/home/bsc/your_bsc_user/repos/vector-inference-v2"
REMOTE_ACCOUNT="your_slurm_account"
VEC_INF_PROJECT_ROOT="/gpfs/projects/<project>/vec-inf"
# Optional when storage owner differs from REMOTE_USER (defaults to REMOTE_USER)
VEC_INF_STORAGE_USER="your_storage_user"
```

The public MN5 profile resolves most tracked paths from these variables:

- `REMOTE_USER`: your BSC login username; used for SSH, rsync, and default `/home/bsc/<user>/...` paths.
- `RSYNC_DEST`: remote checkout path on MN5 where this repo is copied.
- `VEC_INF_ENV`: remote Python environment used to run `vec-inf launch`; normally `${RSYNC_DEST}/.venv`.
- `VEC_INF_CONFIG_DIR_REMOTE`: remote directory containing the MN5 `environment.yaml` and `models.yaml`.
- `REMOTE_WORK_DIR`: working directory passed to Slurm jobs; vec-inf uses it for runtime caches such as `.vec-inf-cache`.
- `REMOTE_ACCOUNT`: Slurm account to charge the job to.
- `VEC_INF_PROJECT_ROOT`: shared project directory containing `containers/`, `vec-inf-shared/`, and `models/`.
- `VEC_INF_STORAGE_USER`: optional override only when your storage owner differs from your login user.

Set `VEC_INF_STORAGE_USER` to the owner used in `/gpfs/.../users/<owner>/...` when it differs from `REMOTE_USER`.

3. Update [`vec_inf/config/marenostrum5/environment.yaml`](../vec_inf/config/marenostrum5/environment.yaml) only if your project layout differs from the default `containers/`, `vec-inf-shared/`, `models/` convention under `VEC_INF_PROJECT_ROOT`.

4. First-time setup (sync repo + create remote environment):

```bash
./scripts/first_time_setup.sh
```

Dry-run preview (no remote writes):

```bash
./scripts/first_time_setup.sh --dry-run
```

5. Launch a model and open a local tunnel:

```bash
./scripts/launch_and_tunnel.sh Meta-Llama-3.1-8B-Instruct 5678
```

Server endpoint will be available at:

```text
http://localhost:5678/v1
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
- Set explicit `RSYNC_DEST`/`VEC_INF_ENV`/`REMOTE_WORK_DIR` if you want side-by-side versions (for example, `-v2`) without reusing old paths.
- `launch_and_tunnel.sh` now exports `VEC_INF_PROJECT_ROOT` and `VEC_INF_STORAGE_USER` so the MN5 YAMLs can stay generic.
