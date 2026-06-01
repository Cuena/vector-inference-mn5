#!/usr/bin/env bash
# Usage: ./scripts/sync_repo.sh
#
# Sync this repo to the remote filesystem via rsync.
#
# Configuration:
#   Copy scripts/.launch.env.example to scripts/.launch.env and customize.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)

if [ ! -f "${SCRIPT_DIR}/.launch.env" ]; then
    echo "[ERROR] Configuration file ${SCRIPT_DIR}/.launch.env not found!"
    echo ""
    echo "Please copy and customize the example configuration:"
    echo "  cp ${SCRIPT_DIR}/.launch.env.example ${SCRIPT_DIR}/.launch.env"
    echo ""
    exit 1
fi

# Preserve one-off shell overrides so `.launch.env` acts as defaults.
_launch_env_vars=(
    REMOTE_LAUNCH_HOST REMOTE_TRANSFER_HOST REMOTE_INTERNET_HOST
    REMOTE_USER VEC_INF_PROJECT_ROOT
    VEC_INF_VLLM_IMAGE_PATH VEC_INF_CACHED_MODEL_CONFIG_PATH
    VEC_INF_MODEL_WEIGHTS_PARENT_DIR
    MODEL_NAME LOCAL_PORT AUTO_KILL_STALE_TUNNEL
    RSYNC_ENABLED RSYNC_SRC RSYNC_DEST VEC_INF_ENV VEC_INF_CONFIG_DIR_REMOTE
    REMOTE_WORK_DIR REMOTE_ACCOUNT REMOTE_QOS UV_SYNC_ARGS
    JOB_START_TIMEOUT SERVER_READY_TIMEOUT
)
_launch_env_override_names=()
_launch_env_override_values=()
for _name in "${_launch_env_vars[@]}"; do
    if [ "${!_name+x}" = "x" ]; then
        _launch_env_override_names+=("${_name}")
        _launch_env_override_values+=("${!_name}")
    fi
done

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/.launch.env"

for _idx in "${!_launch_env_override_names[@]}"; do
    _name="${_launch_env_override_names[$_idx]}"
    _value="${_launch_env_override_values[$_idx]}"
    printf -v "${_name}" '%s' "${_value}"
    export "${_name}"
done
unset _launch_env_vars _launch_env_override_names _launch_env_override_values
unset _idx _name _value

is_bsc_es_host() {
    case "${1:-}" in
        *.bsc.es) return 0 ;;
        *) return 1 ;;
    esac
}

# Always use the transfer node for local<->cluster data movement.
REMOTE_TRANSFER_HOST="${REMOTE_TRANSFER_HOST:-transfer1.bsc.es}"

# Enforce BSC MN5 routing rules when using bsc.es hostnames.
if is_bsc_es_host "${REMOTE_TRANSFER_HOST}" && [ "${REMOTE_TRANSFER_HOST}" != "transfer1.bsc.es" ]; then
    echo "[ERROR] REMOTE_TRANSFER_HOST must be transfer1.bsc.es for data transfers (got: ${REMOTE_TRANSFER_HOST})." >&2
    exit 2
fi

# Set your remote username here (or provide REMOTE_USER in scripts/.launch.env).
REMOTE_USER="${REMOTE_USER:-changeme}"
if [ "${REMOTE_USER}" = "changeme" ] || [ -z "${REMOTE_USER}" ]; then
    echo "[ERROR] REMOTE_USER is not set."
    echo "Set it in scripts/.launch.env (REMOTE_USER=...) or edit scripts/sync_repo.sh."
    exit 1
fi

REMOTE_SSH_TARGET="${REMOTE_USER}@${REMOTE_TRANSFER_HOST}"

RSYNC_SRC="${RSYNC_SRC:-.}"
RSYNC_DEST="${RSYNC_DEST:-}"
if [ -z "${RSYNC_DEST}" ]; then
    RSYNC_DEST="/home/bsc/${REMOTE_USER}/repos/vector-inference-mn5"
fi
REMOTE_WORK_DIR="${REMOTE_WORK_DIR:-}"
if [ "${REMOTE_WORK_DIR}" = "RSYNC_DEST" ]; then
    REMOTE_WORK_DIR="${RSYNC_DEST}"
fi

echo "======================================================================"
echo "  Vector Inference Repo Sync"
echo "======================================================================"
echo "Remote:           ${REMOTE_SSH_TARGET}"
echo "Sync local:      ${RSYNC_SRC} -> ${REMOTE_SSH_TARGET}:${RSYNC_DEST}"
echo "======================================================================"
echo ""

echo "--> Preparing remote destination: ${RSYNC_DEST}"
ssh "${REMOTE_SSH_TARGET}" "mkdir -p '${RSYNC_DEST}'" 2>/dev/null || true

if [ -n "${REMOTE_WORK_DIR:-}" ]; then
    echo "--> Preparing remote work dir: ${REMOTE_WORK_DIR}"
    ssh "${REMOTE_SSH_TARGET}" "mkdir -p '${REMOTE_WORK_DIR}'" 2>/dev/null || true
fi

echo "--> Syncing repo via rsync..."
rsync -rltDzv --filter=":- .gitignore" --exclude=".git" -e ssh "${RSYNC_SRC%/}/" "${REMOTE_SSH_TARGET}:${RSYNC_DEST%/}/"
echo "[OK] rsync complete."
