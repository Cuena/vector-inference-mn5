#!/usr/bin/env bash
# Usage: ./scripts/print_tunnel_cmd.sh [JOB_ID] [LOCAL_PORT]
#
# Prints an SSH command that creates a local tunnel to a vec-inf job that is
# already running on the remote cluster.
#
# It locates the job's JSON metadata under ~/.vec-inf-logs* on the remote host,
# extracts the server endpoint, and prints an `ssh -L ...` command.
#
# Examples:
#   ./scripts/print_tunnel_cmd.sh 123456
#   ./scripts/print_tunnel_cmd.sh 123456 5679
#
# Notes:
#   - This script does not create the tunnel; it only prints the command.
#   - It uses scripts/.launch.env to determine REMOTE_USER and REMOTE_LAUNCH_HOST.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)

show_help() {
    cat <<'EOF'
Usage:
  ./scripts/print_tunnel_cmd.sh [JOB_ID] [LOCAL_PORT]

Args:
  JOB_ID       Required. SLURM job id (numeric).
  LOCAL_PORT   Optional. Local port to bind (default from scripts/.launch.env, else 5678).

Output:
  Prints a single ssh command to stdout, e.g.:
    ssh -o ExitOnForwardFailure=yes -N -L 5678:<SERVER_IP>:<SERVER_PORT> <USER>@<HOST>

Tips:
  - After running the printed command, access the server at:
      http://localhost:<LOCAL_PORT>/v1
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    show_help
    exit 0
fi

JOB_ID="${1:-}"
if [ -z "${JOB_ID}" ]; then
    echo "[ERROR] JOB_ID is required." >&2
    show_help >&2
    exit 2
fi
if ! [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] JOB_ID must be numeric (got: ${JOB_ID})." >&2
    exit 2
fi

# Load configuration from scripts/.launch.env
if [ ! -f "${SCRIPT_DIR}/.launch.env" ]; then
    echo "[ERROR] Configuration file ${SCRIPT_DIR}/.launch.env not found!" >&2
    echo "Please copy and customize the example configuration:" >&2
    echo "  cp ${SCRIPT_DIR}/.launch.env.example ${SCRIPT_DIR}/.launch.env" >&2
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

REMOTE_LAUNCH_HOST="${REMOTE_LAUNCH_HOST:-alogin1.bsc.es}"

if is_bsc_es_host "${REMOTE_LAUNCH_HOST}" && ! [[ "${REMOTE_LAUNCH_HOST}" =~ ^alogin[1-4]\.bsc\.es$ ]]; then
    echo "[ERROR] REMOTE_LAUNCH_HOST must be one of alogin1-4.bsc.es (got: ${REMOTE_LAUNCH_HOST})." >&2
    exit 2
fi

REMOTE_USER="${REMOTE_USER:-changeme}"
if [ "${REMOTE_USER}" = "changeme" ] || [ -z "${REMOTE_USER}" ]; then
    echo "[ERROR] REMOTE_USER is not set." >&2
    echo "Set it in scripts/.launch.env (REMOTE_USER=...)." >&2
    exit 1
fi

REMOTE_SSH_TARGET="${REMOTE_USER}@${REMOTE_LAUNCH_HOST}"

LOCAL_PORT="${2:-${LOCAL_PORT:-5678}}"
if ! [[ "${LOCAL_PORT}" =~ ^[0-9]+$ ]] || [ "${LOCAL_PORT}" -lt 1 ] || [ "${LOCAL_PORT}" -gt 65535 ]; then
    echo "[ERROR] Invalid LOCAL_PORT: ${LOCAL_PORT}" >&2
    exit 2
fi

is_port_listening() {
    local p="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"${p}" -sTCP:LISTEN >/dev/null 2>&1
        return $?
    fi
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "( sport = :${p} )" 2>/dev/null | grep -q LISTEN
        return $?
    fi
    return 1
}

next_free_port() {
    local start="$1"
    local p=""
    for p in $(seq $((start + 1)) $((start + 200))); do
        if ! is_port_listening "${p}"; then
            echo "${p}"
            return 0
        fi
    done
    return 1
}

if is_port_listening "${LOCAL_PORT}"; then
    if [ -t 0 ]; then
        suggested=$(next_free_port "${LOCAL_PORT}" || true)
        if [ -n "${suggested:-}" ]; then
            echo "[WARN] Local port ${LOCAL_PORT} is in use." >&2
            echo "--> Press Enter to use ${suggested}, type another port number, or 'q' to abort:" >&2
            read -r answer
            answer="${answer:-${suggested}}"
            if [ "${answer}" = "q" ] || [ "${answer}" = "Q" ]; then
                echo "[ERROR] Aborted due to local port conflict." >&2
                exit 2
            fi
            if ! [[ "${answer}" =~ ^[0-9]+$ ]] || [ "${answer}" -lt 1 ] || [ "${answer}" -gt 65535 ]; then
                echo "[ERROR] Invalid port: ${answer}" >&2
                exit 2
            fi
            LOCAL_PORT="${answer}"
        fi
    fi
fi

REMOTE_JSON_PATH=$(
    ssh "${REMOTE_SSH_TARGET}" "
set -e
for d in ~/.vec-inf-logs-v2 ~/.vec-inf-logs; do
  [ -d \"\$d\" ] || continue
  p=\$(find \"\$d\" -maxdepth 6 -name '*.${JOB_ID}.json' -print -quit 2>/dev/null || true)
  if [ -n \"\$p\" ]; then
    echo \"\$p\"
    break
  fi
done
" || true
)
REMOTE_JSON_PATH=$(echo "${REMOTE_JSON_PATH}" | head -n1 | tr -d '\r')
if [ -z "${REMOTE_JSON_PATH}" ]; then
    echo "[ERROR] Could not find a JSON log for job ${JOB_ID} on ${REMOTE_SSH_TARGET} under ~/.vec-inf-logs-v2 or ~/.vec-inf-logs." >&2
    echo "Try:" >&2
    echo "  ssh ${REMOTE_SSH_TARGET} \"squeue -j ${JOB_ID}\"" >&2
    echo "  ssh ${REMOTE_SSH_TARGET} \"find ~/.vec-inf-logs-v2 ~/.vec-inf-logs -name '*.${JOB_ID}.json' -maxdepth 6 -print\"" >&2
    exit 1
fi

JSON_CONTENT=$(ssh "${REMOTE_SSH_TARGET}" "cat '${REMOTE_JSON_PATH}' 2>/dev/null" || true)
SERVER_INFO=$(echo "$JSON_CONTENT" | python3 -c "
import sys, json
from urllib.parse import urlparse
try:
    data = json.load(sys.stdin)
    addr = data.get('server_address')
    if addr and 'http' in addr:
        p = urlparse(addr)
        if p.hostname and p.port:
            print(f'{p.hostname} {p.port}')
except Exception:
    pass
" || true)

read -r SERVER_HOST SERVER_PORT <<< "${SERVER_INFO:-}"
if [ -z "${SERVER_HOST:-}" ] || [ -z "${SERVER_PORT:-}" ]; then
    echo "[ERROR] Could not extract server endpoint from ${REMOTE_JSON_PATH}." >&2
    echo "The job may still be initializing. Check:" >&2
    echo "  ssh ${REMOTE_SSH_TARGET} \"tail -n 50 '${REMOTE_JSON_PATH}'\"" >&2
    exit 1
fi

SERVER_IP="${SERVER_HOST}"
ACTUAL_IP=$(ssh "${REMOTE_SSH_TARGET}" "getent hosts ${SERVER_HOST} 2>/dev/null | awk '{print \$1}' | head -1" || true)
ACTUAL_IP=$(echo "${ACTUAL_IP}" | tr -d '\r' | head -n1)
if [ -n "${ACTUAL_IP}" ]; then
    SERVER_IP="${ACTUAL_IP}"
fi

echo "ssh -o ExitOnForwardFailure=yes -N -L ${LOCAL_PORT}:${SERVER_IP}:${SERVER_PORT} ${REMOTE_SSH_TARGET}"
