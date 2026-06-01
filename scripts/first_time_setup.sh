#!/usr/bin/env bash
# Usage: ./scripts/first_time_setup.sh [--dry-run]
#
# First-time setup helper for remote usage:
#  1) rsync this repo to the remote filesystem
#  2) run `uv sync` on a login node with internet access
#
# Configuration:
#   Copy scripts/.launch.env.example to scripts/.launch.env and customize.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)

show_help() {
    cat <<'EOF'
Usage:
  ./scripts/first_time_setup.sh [--dry-run]

Options:
  --dry-run   Preview actions without remote writes.
  -h, --help  Show this help.
EOF
}

DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            show_help >&2
            exit 2
            ;;
    esac
done

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

# MN5 (BSC) routing:
# - Local<->cluster transfers: transfer1.bsc.es
# - Internet downloads (e.g. uv sync): alogin4.bsc.es
REMOTE_TRANSFER_HOST="${REMOTE_TRANSFER_HOST:-transfer1.bsc.es}"
REMOTE_INTERNET_HOST="${REMOTE_INTERNET_HOST:-alogin4.bsc.es}"

# Enforce BSC MN5 routing rules when using bsc.es hostnames.
if is_bsc_es_host "${REMOTE_TRANSFER_HOST}" && [ "${REMOTE_TRANSFER_HOST}" != "transfer1.bsc.es" ]; then
    echo "[ERROR] REMOTE_TRANSFER_HOST must be transfer1.bsc.es for data transfers (got: ${REMOTE_TRANSFER_HOST})." >&2
    exit 2
fi
if is_bsc_es_host "${REMOTE_INTERNET_HOST}" && [ "${REMOTE_INTERNET_HOST}" != "alogin4.bsc.es" ]; then
    echo "[ERROR] REMOTE_INTERNET_HOST must be alogin4.bsc.es for internet downloads (got: ${REMOTE_INTERNET_HOST})." >&2
    exit 2
fi

# Set your remote username here (or provide REMOTE_USER in scripts/.launch.env).
REMOTE_USER="${REMOTE_USER:-changeme}"
if [ "${REMOTE_USER}" = "changeme" ] || [ -z "${REMOTE_USER}" ]; then
    echo "[ERROR] REMOTE_USER is not set."
    echo "Set it in scripts/.launch.env (REMOTE_USER=...) or edit scripts/first_time_setup.sh."
    exit 1
fi

REMOTE_TRANSFER_SSH_TARGET="${REMOTE_USER}@${REMOTE_TRANSFER_HOST}"
REMOTE_INTERNET_SSH_TARGET="${REMOTE_USER}@${REMOTE_INTERNET_HOST}"

RSYNC_SRC="${RSYNC_SRC:-.}"
RSYNC_DEST="${RSYNC_DEST:-}"
if [ -z "${RSYNC_DEST}" ]; then
    RSYNC_DEST="/home/bsc/${REMOTE_USER}/repos/vector-inference-mn5"
fi

UV_SYNC_ARGS="${UV_SYNC_ARGS:---frozen}"

echo "======================================================================"
echo "  Vector Inference Remote Setup"
echo "======================================================================"
echo "Remote (rsync):  ${REMOTE_TRANSFER_SSH_TARGET}"
echo "Remote (uv):     ${REMOTE_INTERNET_SSH_TARGET}"
echo "Sync local:      ${RSYNC_SRC} -> ${REMOTE_TRANSFER_SSH_TARGET}:${RSYNC_DEST}"
echo "uv sync args:    ${UV_SYNC_ARGS}"
if [ "${DRY_RUN}" = "1" ]; then
    echo "Mode:            dry-run"
fi
echo "======================================================================"
echo ""

if [ "${DRY_RUN}" = "1" ]; then
    echo "--> [DRY-RUN] Would prepare remote destination: ${RSYNC_DEST}"
    echo "    ssh ${REMOTE_TRANSFER_SSH_TARGET} \"mkdir -p '${RSYNC_DEST}'\""
else
    echo "--> Preparing remote destination: ${RSYNC_DEST}"
    ssh "${REMOTE_TRANSFER_SSH_TARGET}" "mkdir -p '${RSYNC_DEST}'" 2>/dev/null || true
fi

echo "--> Syncing repo via rsync..."
if [ "${DRY_RUN}" = "1" ]; then
    rsync -rltDzvn --itemize-changes --filter=":- .gitignore" --exclude=".git" -e ssh "${RSYNC_SRC%/}/" "${REMOTE_TRANSFER_SSH_TARGET}:${RSYNC_DEST%/}/" || true
    echo "[OK] rsync dry-run complete."
else
    rsync -rltDzv --filter=":- .gitignore" --exclude=".git" -e ssh "${RSYNC_SRC%/}/" "${REMOTE_TRANSFER_SSH_TARGET}:${RSYNC_DEST%/}/"
    echo "[OK] rsync complete."
fi

echo "--> Running uv sync on ${REMOTE_INTERNET_HOST} (internet-enabled login node)..."
if [ "${DRY_RUN}" = "1" ]; then
    if ssh "${REMOTE_INTERNET_SSH_TARGET}" "[ -d '${RSYNC_DEST}' ]" 2>/dev/null; then
        ssh "${REMOTE_INTERNET_SSH_TARGET}" "set -euo pipefail; cd '${RSYNC_DEST}'; command -v uv >/dev/null 2>&1 || { echo '[ERROR] uv not found on remote. Install uv first (only needs to be done once).'; exit 1; }; uv sync ${UV_SYNC_ARGS} --dry-run"
    else
        echo "--> [DRY-RUN] Remote path does not exist yet: ${RSYNC_DEST}"
        echo "    Run without --dry-run first (or create path + sync) to preview uv sync there."
    fi
else
    ssh "${REMOTE_INTERNET_SSH_TARGET}" "set -euo pipefail; cd '${RSYNC_DEST}'; command -v uv >/dev/null 2>&1 || { echo '[ERROR] uv not found on remote. Install uv first (only needs to be done once).'; exit 1; }; uv sync ${UV_SYNC_ARGS}"
fi

if [ "${DRY_RUN}" = "1" ]; then
    echo "[OK] Dry-run complete."
    exit 0
fi

echo "[OK] Remote setup complete."
echo ""
echo "Next: configure scripts/.launch.env VEC_INF_ENV to point to the created venv:"
echo "  VEC_INF_ENV=\"${RSYNC_DEST}/.venv\""
