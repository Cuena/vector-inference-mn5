#!/usr/bin/env bash
# Usage: ./scripts/launch_and_tunnel.sh [--launch-only] [MODEL_NAME] [LOCAL_PORT]
#
# Launches a model on a remote SLURM cluster using vec-inf and opens an SSH tunnel
# so the remote model server is accessible locally.
#
# Configuration:
#   Copy scripts/.launch.env.example to scripts/.launch.env and customize.
#   The script always loads config from its own directory.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)

show_help() {
    cat <<'EOF'
Usage:
  ./scripts/launch_and_tunnel.sh [--launch-only] [MODEL_NAME] [LOCAL_PORT]

Options:
  --launch-only   Only submit the SLURM job and print the job id (no tunnel; does not cancel on exit).
  -h, --help      Show this help.

Notes:
  - MODEL_NAME can also come from scripts/.launch.env (MODEL_NAME=...).
  - LOCAL_PORT is only used when creating the SSH tunnel.
EOF
}

# --- CLI option parsing (keeps positional compatibility) ---
LAUNCH_ONLY=0
POSITIONAL_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --launch-only)
            LAUNCH_ONLY=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "[ERROR] Unknown option: $1" >&2
            show_help >&2
            exit 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL_ARGS[@]}" "$@"

# Load configuration from scripts/.launch.env
if [ ! -f "${SCRIPT_DIR}/.launch.env" ]; then
    echo "[ERROR] Configuration file ${SCRIPT_DIR}/.launch.env not found!"
    echo ""
    echo "Please copy and customize the example configuration:"
    echo "  cp ${SCRIPT_DIR}/.launch.env.example ${SCRIPT_DIR}/.launch.env"
    echo ""
    exit 1
fi

# Preserve one-off shell overrides (e.g. `export REMOTE_LAUNCH_HOST=...`) so
# `.launch.env` acts as defaults.
_launch_env_vars=(
    REMOTE_LAUNCH_HOST REMOTE_TRANSFER_HOST REMOTE_INTERNET_HOST
    REMOTE_USER VEC_INF_STORAGE_USER VEC_INF_PROJECT_ROOT
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

# --- Apply defaults for optional settings ---
is_bsc_es_host() {
    case "${1:-}" in
        *.bsc.es) return 0 ;;
        *) return 1 ;;
    esac
}

# MN5 (BSC) routing:
# - Job submission + tunneling: alogin1-4.bsc.es
# - Local<->cluster transfers: transfer1.bsc.es
REMOTE_LAUNCH_HOST="${REMOTE_LAUNCH_HOST:-alogin1.bsc.es}"
REMOTE_TRANSFER_HOST="${REMOTE_TRANSFER_HOST:-transfer1.bsc.es}"

# Enforce BSC MN5 routing rules when using bsc.es hostnames.
if is_bsc_es_host "${REMOTE_LAUNCH_HOST}" && ! [[ "${REMOTE_LAUNCH_HOST}" =~ ^alogin[1-4]\.bsc\.es$ ]]; then
    echo "[ERROR] REMOTE_LAUNCH_HOST must be one of alogin1-4.bsc.es for job submission/tunneling (got: ${REMOTE_LAUNCH_HOST})." >&2
    exit 2
fi
if is_bsc_es_host "${REMOTE_TRANSFER_HOST}" && [ "${REMOTE_TRANSFER_HOST}" != "transfer1.bsc.es" ]; then
    echo "[ERROR] REMOTE_TRANSFER_HOST must be transfer1.bsc.es for data transfers (got: ${REMOTE_TRANSFER_HOST})." >&2
    exit 2
fi

# Set your remote username here (or provide REMOTE_USER in scripts/.launch.env).
REMOTE_USER="${REMOTE_USER:-changeme}"
if [ "${REMOTE_USER}" = "changeme" ] || [ -z "${REMOTE_USER}" ]; then
    echo "[ERROR] REMOTE_USER is not set."
    echo "Set it in scripts/.launch.env (REMOTE_USER=...) or edit scripts/launch_and_tunnel.sh."
    exit 1
fi

REMOTE_SSH_TARGET="${REMOTE_USER}@${REMOTE_LAUNCH_HOST}"
REMOTE_TRANSFER_SSH_TARGET="${REMOTE_USER}@${REMOTE_TRANSFER_HOST}"

# LOCAL_PORT: only needed when tunneling
if [ "${LAUNCH_ONLY}" != "1" ]; then
    if [ -n "${2:-}" ]; then
        LOCAL_PORT="$2"
        echo "[INFO] Using local port from command-line argument: $LOCAL_PORT"
    else
        LOCAL_PORT="${LOCAL_PORT:-5678}"
    fi
fi

# Minimal rsync settings (optional)
RSYNC_ENABLED="${RSYNC_ENABLED:-0}"
RSYNC_SRC="${RSYNC_SRC:-.}"
RSYNC_DEST="${RSYNC_DEST:-}"

if [ -z "${RSYNC_DEST}" ]; then
    RSYNC_DEST="/home/bsc/${REMOTE_USER}/repos/vector-inference"
fi

# Optional: override vec-inf work-dir (SBATCH --chdir)
REMOTE_WORK_DIR="${REMOTE_WORK_DIR:-}"
if [ "${REMOTE_WORK_DIR}" = "RSYNC_DEST" ]; then
    REMOTE_WORK_DIR="${RSYNC_DEST}"
fi

# Optional: pass Slurm account explicitly to vec-inf launch.
# This avoids requiring VEC_INF_ACCOUNT to be set in remote shell profiles.
REMOTE_ACCOUNT="${REMOTE_ACCOUNT:-}"

# Optional: pass Slurm QoS explicitly to vec-inf launch.
# Leave empty to use vec-inf/environment defaults.
# Set REMOTE_QOS=NONE to disable passing --qos explicitly.
REMOTE_QOS="${REMOTE_QOS:-}"
if [ "${REMOTE_QOS}" = "NONE" ]; then
    REMOTE_QOS=""
fi

# Default venv lives in the remote checkout by default (created by `uv sync`)
VEC_INF_ENV="${VEC_INF_ENV:-}"
if [ -z "${VEC_INF_ENV}" ]; then
    VEC_INF_ENV="${RSYNC_DEST}/.venv"
fi

# By default use the bundled MN5 profile on the remote checkout.
# Set VEC_INF_CONFIG_DIR_REMOTE=NONE to disable exporting VEC_INF_CONFIG_DIR.
VEC_INF_CONFIG_DIR_REMOTE="${VEC_INF_CONFIG_DIR_REMOTE:-}"
if [ -z "${VEC_INF_CONFIG_DIR_REMOTE}" ]; then
    VEC_INF_CONFIG_DIR_REMOTE="${RSYNC_DEST}/vec_inf/config/marenostrum5"
fi

# Directory name used in MN5 paths under /gpfs/.../users/<name>/...
# Defaults to REMOTE_USER, but can be overridden when that path segment differs.
VEC_INF_STORAGE_USER="${VEC_INF_STORAGE_USER:-$REMOTE_USER}"

# Shared project root used by the public MN5 environment profile.
VEC_INF_PROJECT_ROOT="${VEC_INF_PROJECT_ROOT:-}"
if [ -z "${VEC_INF_PROJECT_ROOT}" ]; then
    case "${VEC_INF_CONFIG_DIR_REMOTE:-}" in
        */vec_inf/config/marenostrum5)
            VEC_INF_PROJECT_ROOT="${VEC_INF_CONFIG_DIR_REMOTE%/vec_inf/config/marenostrum5}"
            ;;
        *)
            if [ -n "${RSYNC_DEST:-}" ]; then
                VEC_INF_PROJECT_ROOT="${RSYNC_DEST}"
            fi
            ;;
    esac
fi

# MODEL_NAME: use command-line argument if provided, otherwise from .launch.env
if [ -n "${1:-}" ]; then
    MODEL_NAME="$1"
    echo "[INFO] Using model name from command-line argument: $MODEL_NAME"
elif [ -z "${MODEL_NAME:-}" ]; then
    echo "[ERROR] MODEL_NAME not set. Provide it as an argument or set it in ${SCRIPT_DIR}/.launch.env"
    echo "Usage: $0 [--launch-only] [MODEL_NAME] [LOCAL_PORT]"
    exit 1
fi

# Timeout settings
JOB_START_TIMEOUT="${JOB_START_TIMEOUT:-900}"          # 15 minutes
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"   # 20 minutes

# Use same signature as vec-inf client uses
MODEL_READY_SIGNATURE="INFO:     Application startup complete."
MODEL_READY_SIGNATURE_ESCAPED=$(printf '%q' "$MODEL_READY_SIGNATURE")

# Globals determined after launch
JOB_ID=""
REMOTE_LOG_DIR=""
REMOTE_JSON_PATH=""
REMOTE_ERR_PATH=""
REMOTE_JOB_DIR=""

# Ensure the chosen LOCAL_PORT is free. Safe-by-default: do NOT kill unless
# AUTO_KILL_STALE_TUNNEL=1 is set. When killing, only target ssh -L listeners
# that match this script's ${REMOTE_SSH_TARGET} and the same LOCAL_PORT.
ensure_local_port_free() {
    local port="$1"

    if ! command -v lsof >/dev/null 2>&1 && ! command -v ss >/dev/null 2>&1; then
        echo "--> Warning: cannot check local port availability (missing lsof/ss). Proceeding with LOCAL_PORT=${port}."
        LOCAL_PORT="${port}"
        return 0
    fi

    is_port_listening() {
        local p="$1"
        if command -v lsof >/dev/null 2>&1; then
            lsof -nP -iTCP:"${p}" -sTCP:LISTEN >/dev/null 2>&1
            return $?
        fi
        ss -ltn "( sport = :${p} )" 2>/dev/null | grep -q LISTEN
        return $?
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

    while true; do
        local pids=""
        if command -v lsof >/dev/null 2>&1; then
            pids=$(lsof -nP -iTCP:"${port}" -sTCP:LISTEN -t 2>/dev/null || true)
        else
            pids=$(ss -ltnp "( sport = :${port} )" 2>/dev/null | awk 'NR>1 {print $7}' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u || true)
        fi

        if [ -z "${pids}" ]; then
            LOCAL_PORT="${port}"
            return 0
        fi

        local matching_ssh_pid=""
        for pid in ${pids}; do
            cmd=$(ps -o comm= -p "${pid}" 2>/dev/null || true)
            args=$(ps -o args= -p "${pid}" 2>/dev/null || true)
            if [ "${cmd}" = "ssh" ] \
               && echo "${args}" | grep -q -- "-L[[:space:]]*${port}:" \
               && echo "${args}" | grep -q -- "[[:space:]]${REMOTE_SSH_TARGET}"; then
                matching_ssh_pid="${pid}"
                break
            fi
        done

        if [ -n "${matching_ssh_pid}" ]; then
            if [ "${AUTO_KILL_STALE_TUNNEL:-}" = "1" ] || [ "${AUTO_KILL_STALE_TUNNEL:-}" = "true" ]; then
                echo "--> Reclaiming local port ${port} from matching ssh pid ${matching_ssh_pid} (${REMOTE_SSH_TARGET})..."
                kill "${matching_ssh_pid}" 2>/dev/null || true
                sleep 0.3
                if kill -0 "${matching_ssh_pid}" 2>/dev/null; then
                    kill -9 "${matching_ssh_pid}" 2>/dev/null || true
                fi
                continue
            fi
            echo "--> Port ${port} is in use by an ssh tunnel to ${REMOTE_SSH_TARGET} (pid ${matching_ssh_pid})."
        else
            echo "--> Port ${port} is already in use by another local process."
        fi

        if [ -t 0 ]; then
            local suggested=""
            suggested=$(next_free_port "${port}" || true)
            if [ -z "${suggested}" ]; then
                echo "[ERROR] Could not find a free local port near ${port}. Please set LOCAL_PORT manually."
                exit 2
            fi
            echo "--> Press Enter to use ${suggested}, type another port number, or 'q' to abort:"
            read -r answer
            answer="${answer:-${suggested}}"
            if [ "${answer}" = "q" ] || [ "${answer}" = "Q" ]; then
                echo "[ERROR] Aborted due to local port conflict."
                exit 2
            fi
            if ! [[ "${answer}" =~ ^[0-9]+$ ]] || [ "${answer}" -lt 1 ] || [ "${answer}" -gt 65535 ]; then
                echo "[ERROR] Invalid port: ${answer}"
                exit 2
            fi
            port="${answer}"
            continue
        fi

        echo "--> Refusing to continue non-interactively with a busy port."
        echo "    Set LOCAL_PORT in ${SCRIPT_DIR}/.launch.env (or pass it as an argument) to use a different port."
        exit 2
    done
}

cleanup() {
    if [ "${CANCEL_JOB:-1}" = "1" ] && [ -n "${JOB_ID:-}" ]; then
        echo ""
        echo "--> Cancelling remote SLURM job ${JOB_ID}..."
        ssh "${REMOTE_SSH_TARGET}" "scancel ${JOB_ID}" 2>/dev/null || true
    fi
    echo "--> Cleanup complete."
}

# Default behavior (tunnel mode): cancel the job when the script exits.
# Launch-only mode: never cancel automatically.
CANCEL_JOB="${CANCEL_JOB:-1}"
if [ "${LAUNCH_ONLY}" = "1" ]; then
    CANCEL_JOB=0
fi
trap cleanup SIGINT SIGTERM EXIT

ensure_log_paths_available() {
    if [ -n "${REMOTE_JSON_PATH:-}" ] && [ -n "${REMOTE_ERR_PATH:-}" ]; then
        return
    fi

    if [ -n "${REMOTE_LOG_DIR:-}" ] && [ -n "${JOB_ID:-}" ] && [ -n "${MODEL_NAME:-}" ]; then
        REMOTE_JOB_DIR="${REMOTE_LOG_DIR}/${MODEL_NAME}.${JOB_ID}"
        REMOTE_JSON_PATH="${REMOTE_JOB_DIR}/${MODEL_NAME}.${JOB_ID}.json"
        REMOTE_ERR_PATH="${REMOTE_JOB_DIR}/${MODEL_NAME}.${JOB_ID}.err"
        return
    fi

    # Fallback: locate the job JSON under ~/.vec-inf-logs-v2 or ~/.vec-inf-logs
    local found_json_path
    found_json_path=$(
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
    found_json_path=$(echo "$found_json_path" | head -n1 | tr -d '\r')

    if [ -n "$found_json_path" ]; then
        REMOTE_JSON_PATH="$found_json_path"
        REMOTE_ERR_PATH="${found_json_path%.json}.err"
        REMOTE_LOG_DIR=$(dirname "$found_json_path")
        REMOTE_JOB_DIR="$REMOTE_LOG_DIR"
    fi
}

get_remote_job_state() {
    local state=""
    if [ -n "${JOB_ID:-}" ]; then
        state=$(ssh "${REMOTE_SSH_TARGET}" "squeue -j $JOB_ID -h -o %T" 2>/dev/null || true)
        state=$(echo "$state" | head -n1 | tr -d '\r')
        if [ -z "$state" ]; then
            state=$(ssh "${REMOTE_SSH_TARGET}" "sacct -j $JOB_ID --format=State -n -P 2>/dev/null | head -n1 | cut -d'|' -f1" || true)
            state=$(echo "$state" | tr -d '\r' | awk '{print $1}')
        fi
    fi
    echo "$state"
}

print_remote_err_tail() {
    ensure_log_paths_available
    if [ -n "${REMOTE_ERR_PATH:-}" ]; then
        ssh "${REMOTE_SSH_TARGET}" "if [ -r '${REMOTE_ERR_PATH}' ]; then echo '--- Last 80 lines of ${REMOTE_ERR_PATH} ---'; tail -n 80 '${REMOTE_ERR_PATH}'; fi" 2>/dev/null || true
    fi
}

abort_if_job_ended() {
    local context="$1"
    local state=""
    state=$(get_remote_job_state)
    if [ -z "$state" ]; then
        return 0
    fi

    case "$state" in
        PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|RESIZING)
            return 0
            ;;
        *)
            echo ""
            echo "[ERROR] Job ${JOB_ID} ended while ${context}. Final state: ${state}"
            echo "[ERROR] Check SLURM output on the remote host for details."
            print_remote_err_tail
            exit 1
            ;;
    esac
}

# Check local port availability
if [ "${LAUNCH_ONLY}" != "1" ]; then
    ensure_local_port_free "${LOCAL_PORT}"
fi

echo "======================================================================"
echo "  Vector Inference Launch & Tunnel"
echo "======================================================================"
echo "Model:        $MODEL_NAME"
echo "Launch host:  ${REMOTE_SSH_TARGET}"
if [ "${LAUNCH_ONLY}" != "1" ]; then
    echo "Local port:   $LOCAL_PORT"
else
    echo "Mode:         launch-only (no tunnel)"
fi
echo "Venv:         $VEC_INF_ENV"
if [ "${VEC_INF_CONFIG_DIR_REMOTE}" != "NONE" ]; then
    echo "Config dir:   $VEC_INF_CONFIG_DIR_REMOTE"
fi
if [ -n "${VEC_INF_STORAGE_USER:-}" ]; then
    echo "Storage user: $VEC_INF_STORAGE_USER"
fi
if [ -n "${VEC_INF_VLLM_IMAGE_PATH:-}" ]; then
    echo "vLLM image:   $VEC_INF_VLLM_IMAGE_PATH"
fi
if [ -n "${VEC_INF_MODEL_WEIGHTS_PARENT_DIR:-}" ]; then
    echo "Weights root: $VEC_INF_MODEL_WEIGHTS_PARENT_DIR"
fi
if [ -n "${VEC_INF_PROJECT_ROOT:-}" ]; then
    echo "Project root: $VEC_INF_PROJECT_ROOT"
else
    echo "Project root: <unset>"
fi
if [ -n "${REMOTE_WORK_DIR:-}" ]; then
    echo "Work dir:     $REMOTE_WORK_DIR"
fi
if [ -n "${REMOTE_ACCOUNT:-}" ]; then
    echo "Account:      $REMOTE_ACCOUNT"
fi
if [ -n "${REMOTE_QOS:-}" ]; then
    echo "QoS:          $REMOTE_QOS"
fi
if [ "${RSYNC_ENABLED}" = "1" ] || [ "${RSYNC_ENABLED}" = "true" ]; then
    echo "Transfer:     ${REMOTE_TRANSFER_SSH_TARGET}"
    echo "Sync local:   ${RSYNC_SRC} -> ${REMOTE_TRANSFER_SSH_TARGET}:${RSYNC_DEST}"
fi
echo "======================================================================"
echo ""

# Optionally sync code to remote before launch (minimal)
if [ "${RSYNC_ENABLED}" = "1" ] || [ "${RSYNC_ENABLED}" = "true" ]; then
    echo "--> Preparing remote path for rsync: ${RSYNC_DEST}"
    ssh "${REMOTE_TRANSFER_SSH_TARGET}" "mkdir -p '${RSYNC_DEST}'" 2>/dev/null || true
    echo "--> Syncing code via rsync..."
    rsync -rltDzv --filter=":- .gitignore" --exclude=".git" -e ssh "${RSYNC_SRC%/}/" "${REMOTE_TRANSFER_SSH_TARGET}:${RSYNC_DEST%/}/"
    echo "[OK] rsync complete."
fi

echo "--> Launching $MODEL_NAME on cluster..."

LAUNCH_CMD="vec-inf launch '$MODEL_NAME' --json-mode"
if [ -n "${REMOTE_WORK_DIR:-}" ]; then
    LAUNCH_CMD+=" --work-dir '$REMOTE_WORK_DIR'"
fi
if [ -n "${REMOTE_ACCOUNT:-}" ]; then
    LAUNCH_CMD+=" --account '$REMOTE_ACCOUNT'"
fi
if [ -n "${REMOTE_QOS:-}" ]; then
    LAUNCH_CMD+=" --qos '$REMOTE_QOS'"
fi

REMOTE_CMD="source '$VEC_INF_ENV/bin/activate'"
if [ "${VEC_INF_CONFIG_DIR_REMOTE}" != "NONE" ]; then
    REMOTE_CMD+=" && export VEC_INF_CONFIG_DIR='$VEC_INF_CONFIG_DIR_REMOTE'"
fi
if [ -n "${VEC_INF_STORAGE_USER:-}" ]; then
    REMOTE_CMD+=" && export VEC_INF_STORAGE_USER='$VEC_INF_STORAGE_USER'"
fi
if [ -n "${VEC_INF_VLLM_IMAGE_PATH:-}" ]; then
    REMOTE_CMD+=" && export VEC_INF_VLLM_IMAGE_PATH='$VEC_INF_VLLM_IMAGE_PATH'"
fi
if [ -n "${VEC_INF_CACHED_MODEL_CONFIG_PATH:-}" ]; then
    REMOTE_CMD+=" && export VEC_INF_CACHED_MODEL_CONFIG_PATH='$VEC_INF_CACHED_MODEL_CONFIG_PATH'"
fi
if [ -n "${VEC_INF_MODEL_WEIGHTS_PARENT_DIR:-}" ]; then
    REMOTE_CMD+=" && export VEC_INF_MODEL_WEIGHTS_PARENT_DIR='$VEC_INF_MODEL_WEIGHTS_PARENT_DIR'"
fi
if [ -n "${VEC_INF_PROJECT_ROOT:-}" ]; then
    REMOTE_CMD+=" && export VEC_INF_PROJECT_ROOT='$VEC_INF_PROJECT_ROOT'"
fi
REMOTE_CMD+=" && ${LAUNCH_CMD}"

if ! LAUNCH_OUTPUT=$(ssh "${REMOTE_SSH_TARGET}" "${REMOTE_CMD}"); then
    echo "[ERROR] Failed to launch model"
    exit 1
fi

echo "$LAUNCH_OUTPUT"

JOB_ID=$(echo "$LAUNCH_OUTPUT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('slurm_job_id',''))")
REMOTE_LOG_DIR=$(echo "$LAUNCH_OUTPUT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('log_dir',''))")

if ! [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Failed to extract valid job ID from launch output"
    exit 1
fi

if [ -n "${REMOTE_LOG_DIR:-}" ]; then
    REMOTE_LOG_DIR=$(ssh "${REMOTE_SSH_TARGET}" "python3 -c \"import os,sys; print(os.path.expanduser(os.path.expandvars(sys.argv[1])))\" '${REMOTE_LOG_DIR}'" 2>/dev/null || true)
fi

ensure_log_paths_available

echo "[OK] Model launched! Job ID: $JOB_ID"
echo ""

if [ "${LAUNCH_ONLY}" = "1" ]; then
    echo "Remote commands:"
    echo "  ssh ${REMOTE_SSH_TARGET} \"squeue -j ${JOB_ID}\""
    echo "  ssh ${REMOTE_SSH_TARGET} \"sacct -j ${JOB_ID} --format=JobID,State%20,Elapsed,MaxRSS --units=G -n -P\""
    if [ -n "${REMOTE_JOB_DIR:-}" ]; then
        echo "Logs:"
        echo "  ssh ${REMOTE_SSH_TARGET} \"ls -lah '${REMOTE_JOB_DIR}'\""
    fi
    trap - SIGINT SIGTERM EXIT
    exit 0
fi

echo "--> Waiting for job to start... (timeout: ${JOB_START_TIMEOUT}s)"
elapsed=0
while true; do
    STATUS=$(ssh "${REMOTE_SSH_TARGET}" "squeue -j $JOB_ID -h -o %T" 2>/dev/null || true)
    if [ "${STATUS}" == "RUNNING" ]; then
        echo "[OK] Job is now RUNNING."
        break
    elif [ -z "${STATUS}" ] || [[ "${STATUS}" == "FAILED" || "${STATUS}" == "CANCELLED" || "${STATUS}" == "COMPLETED" ]]; then
        echo "[ERROR] Job ${JOB_ID} ended unexpectedly. Status: ${STATUS:-'UNKNOWN'}"
        echo "[ERROR] Check SLURM output on the remote host for details."
        exit 1
    fi

    if [ $elapsed -ge $JOB_START_TIMEOUT ]; then
        echo "[ERROR] Timeout waiting for job to start (${JOB_START_TIMEOUT}s)"
        exit 1
    fi

    printf "."
    sleep 5
    elapsed=$((elapsed + 5))
done

echo ""

echo "--> Waiting for server endpoint... (timeout: ${SERVER_READY_TIMEOUT}s)"
elapsed=0
SERVER_IP=""
SERVER_PORT=""
SERVER_HOST_LABEL=""

while true; do
    JSON_CONTENT=""
    abort_if_job_ended "waiting for server endpoint"
    ensure_log_paths_available

    if [ -n "${REMOTE_JSON_PATH:-}" ]; then
        JSON_CONTENT=$(ssh "${REMOTE_SSH_TARGET}" "if [ -r '${REMOTE_JSON_PATH}' ]; then cat '${REMOTE_JSON_PATH}'; fi" 2>/dev/null || true)
    fi

    if [ -n "$JSON_CONTENT" ]; then
        SERVER_INFO=$(echo "$JSON_CONTENT" | python3 -c "
import sys, json
from urllib.parse import urlparse
try:
    data = json.load(sys.stdin)
    addr = data.get('server_address')
    if addr and 'http' in addr:
        p = urlparse(addr)
        print(f'{p.hostname} {p.port}')
except Exception:
    pass
" || true)
        read -r SERVER_IP SERVER_PORT <<< "$SERVER_INFO"

        if [ -n "$SERVER_IP" ] && [ -n "$SERVER_PORT" ]; then
            SERVER_HOST_LABEL="$SERVER_IP"
            ACTUAL_IP=$(ssh "${REMOTE_SSH_TARGET}" "getent hosts $SERVER_IP 2>/dev/null | awk '{print \$1}' | head -1" || true)

            if [ -n "$ACTUAL_IP" ]; then
                SERVER_IP="$ACTUAL_IP"
                echo "[INFO] Server endpoint assigned at ${SERVER_HOST_LABEL}:$SERVER_PORT (resolves to: $SERVER_IP)"
            else
                echo "[INFO] Server endpoint assigned at $SERVER_IP:$SERVER_PORT (using hostname)"
            fi
            echo "--> Waiting for model readiness signal (server may still be initializing)..."
            break
        fi
    fi

    if [ $elapsed -ge $SERVER_READY_TIMEOUT ]; then
        echo "[ERROR] Timeout waiting for server endpoint (${SERVER_READY_TIMEOUT}s)"
        echo "[ERROR] The job may still be initializing. Check logs manually:"
        echo "        ssh ${REMOTE_SSH_TARGET} 'find ~/.vec-inf-logs -name \"*.${JOB_ID}.json\" -print -quit'"
        exit 1
    fi

    printf "."
    sleep 30
    elapsed=$((elapsed + 30))
done

echo ""

READY_CHECK_INTERVAL=15
while true; do
    ensure_log_paths_available
    abort_if_job_ended "waiting for model readiness"
    if [ -n "${REMOTE_ERR_PATH:-}" ]; then
        READY_FLAG=$(ssh "${REMOTE_SSH_TARGET}" "if [ -r '${REMOTE_ERR_PATH}' ]; then grep -F -m1 -- ${MODEL_READY_SIGNATURE_ESCAPED} '${REMOTE_ERR_PATH}' >/dev/null && echo READY; fi" 2>/dev/null || true)
        if [ "${READY_FLAG}" = "READY" ]; then
            echo "[OK] Model initialization complete (ready signal detected)."
            break
        fi
    fi

    if [ $elapsed -ge $SERVER_READY_TIMEOUT ]; then
        echo "[ERROR] Timeout waiting for model readiness (${SERVER_READY_TIMEOUT}s)"
        echo "[ERROR] Check logs manually for progress:"
        if [ -n "${REMOTE_JOB_DIR:-}" ]; then
            echo "        ssh ${REMOTE_SSH_TARGET} 'ls ${REMOTE_JOB_DIR}'"
        else
            echo "        ssh ${REMOTE_SSH_TARGET} 'find ~/.vec-inf-logs -name \"*.${JOB_ID}.err\" -print -quit'"
        fi
        exit 1
    fi

    printf "."
    sleep "${READY_CHECK_INTERVAL}"
    elapsed=$((elapsed + READY_CHECK_INTERVAL))
done

echo ""

echo "--> Establishing SSH tunnel..."
echo ""
echo "======================================================================"
echo "  SERVER ACCESSIBLE AT: http://localhost:${LOCAL_PORT}/v1"
echo "======================================================================"
echo ""
echo "Job ID:        $JOB_ID"
echo "Remote server: ${SERVER_IP}:${SERVER_PORT}"
echo ""
echo "Press Ctrl+C to close the tunnel and cancel the remote job."
echo ""

ssh -o ExitOnForwardFailure=yes -N -L "${LOCAL_PORT}:${SERVER_IP}:${SERVER_PORT}" "${REMOTE_SSH_TARGET}"

echo "--> Tunnel closed."
