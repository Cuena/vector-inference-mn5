#!/usr/bin/env bash
# Diagnostics for FlashInfer CUDA header/toolchain issues (e.g. missing <cuda/ptx>).
#
# Run:
#   bash scripts/diag_flashinfer_cuda.sh | tee diag.txt
#
# If your inference environment uses a specific venv:
#   bash scripts/diag_flashinfer_cuda.sh --venv /path/to/.venv | tee diag.txt
#
# To probe available CUDA modules (does not modify parent shell):
#   bash scripts/diag_flashinfer_cuda.sh --probe-cuda-modules | tee diag.txt
#
# Then paste the output (or attach diag.txt).

set -euo pipefail

show_help() {
    cat <<'EOF'
Usage:
  bash scripts/diag_flashinfer_cuda.sh [--venv PATH] [--probe-cuda-modules]

Purpose:
  Collect environment + CUDA toolkit + Python package details needed to debug
  FlashInfer/vLLM JIT build failures on HPC systems.

Tip:
  bash scripts/diag_flashinfer_cuda.sh | tee diag.txt

Options:
  --venv PATH              Source PATH/bin/activate before Python checks.
  --probe-cuda-modules     Try to locate a CUDA module that provides nvcc and
                           the <cuda/ptx> header (runs module loads in a
                           subshell so your current shell is unchanged).
EOF
}

VENV_TO_SOURCE=""
PROBE_CUDA_MODULES=0
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        --venv)
            if [ $# -lt 2 ]; then
                echo "[ERROR] --venv requires a PATH argument" >&2
                exit 2
            fi
            VENV_TO_SOURCE="$2"
            shift 2
            ;;
        --probe-cuda-modules)
            PROBE_CUDA_MODULES=1
            shift
            ;;
        *)
            echo "[ERROR] Unexpected argument: $1" >&2
            show_help >&2
            exit 2
            ;;
    esac
done

hr() {
    echo ""
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
}

run() {
    echo ""
    echo "+ $*"
    set +e
    "$@" 2>&1
    local ec=$?
    set -e
    if [ $ec -ne 0 ]; then
        echo "[exit=$ec]"
    fi
    return 0
}

maybe_run() {
    local bin="$1"
    shift || true
    if command -v "$bin" >/dev/null 2>&1; then
        run "$bin" "$@"
    else
        echo ""
        echo "+ $bin $*"
        echo "[missing] $bin"
    fi
}

try_enable_modules() {
    # Some systems define `module` only after sourcing an init script.
    if type module >/dev/null 2>&1; then
        return 0
    fi
    local candidates=(
        "/etc/profile.d/modules.sh"
        "/usr/share/Modules/init/bash"
        "/usr/share/modules/init/bash"
        "/usr/share/lmod/lmod/init/bash"
        "/opt/apps/lmod/lmod/init/bash"
    )
    local f
    for f in "${candidates[@]}"; do
        if [ -r "$f" ]; then
            # shellcheck disable=SC1090
            source "$f" >/dev/null 2>&1 || true
            if type module >/dev/null 2>&1; then
                return 0
            fi
        fi
    done
    return 0
}

probe_cuda_modules() {
    try_enable_modules
    if ! type module >/dev/null 2>&1; then
        echo "module command not available in this shell."
        return 0
    fi

    echo "+ module -t avail cuda"
    local avail
    avail="$(module -t avail cuda 2>&1 | tr -s ' \t' '\n' | sed '/^$/d' | sort -u)"
    if [ -z "$avail" ]; then
        echo "[no output] module -t avail cuda"
        return 0
    fi

    local mods
    mods="$(printf '%s\n' "$avail" | grep -E '^(cuda|CUDA)/' || true)"
    if [ -z "$mods" ]; then
        echo "[no matches] No modules matching ^(cuda|CUDA)/ found in output."
        echo "Raw module output (first 120 lines):"
        printf '%s\n' "$avail" | head -n 120
        return 0
    fi

    echo ""
    echo "Candidate CUDA modules:"
    printf '%s\n' "$mods" | head -n 120
    echo ""
    echo "Probing modules for nvcc + <cuda/ptx> (subshell; does not affect current shell):"
    echo ""

    local m
    while IFS= read -r m; do
        [ -n "$m" ] || continue
        echo "----------------------------------------------------------------------"
        echo "module: $m"
        bash -lc "
            set -euo pipefail
            if type module >/dev/null 2>&1; then :; else
                for f in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/modules/init/bash /usr/share/lmod/lmod/init/bash /opt/apps/lmod/lmod/init/bash; do
                    [ -r \"\$f\" ] && source \"\$f\" >/dev/null 2>&1 || true
                done
            fi
            if ! type module >/dev/null 2>&1; then
                echo \"module command unavailable in subshell\"
                exit 0
            fi
            module load \"$m\" >/dev/null 2>&1 || { echo \"FAILED: module load $m\"; exit 0; }
            if command -v nvcc >/dev/null 2>&1; then
                nvcc_path=\"\$(command -v nvcc)\"
                cuda_root=\"\$(dirname \"\$(dirname \"\$(readlink -f \"\$nvcc_path\")\")\")\"
                echo \"nvcc=\$nvcc_path\"
                nvcc --version | sed -n '1,8p'
                if test -e \"\$cuda_root/include/cuda/ptx\"; then
                    echo \"cuda/ptx=FOUND (\$cuda_root/include/cuda/ptx)\"
                else
                    echo \"cuda/ptx=MISSING (\$cuda_root/include/cuda/ptx)\"
                fi
                if test -d \"\$cuda_root/include/cccl\"; then
                    echo \"cccl=FOUND (\$cuda_root/include/cccl)\"
                else
                    echo \"cccl=MISSING (\$cuda_root/include/cccl)\"
                fi
            else
                echo \"nvcc=MISSING (module did not provide nvcc)\"
            fi
        " 2>&1 || true
    done <<<"$mods"
    echo "----------------------------------------------------------------------"
    return 0
}

detect_cuda_root() {
    # Preference order:
    #  1) CUDA_HOME/CUDA_PATH if they look valid
    #  2) derive from nvcc path
    #  3) /usr/local/cuda if present
    local root=""
    if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME:-}/include" ]; then
        root="$CUDA_HOME"
    elif [ -n "${CUDA_PATH:-}" ] && [ -d "${CUDA_PATH:-}/include" ]; then
        root="$CUDA_PATH"
    elif command -v nvcc >/dev/null 2>&1; then
        root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
        if [ ! -d "$root/include" ]; then
            root=""
        fi
    fi
    if [ -z "$root" ] && [ -d "/usr/local/cuda/include" ]; then
        root="/usr/local/cuda"
    fi
    echo "$root"
}

hr "Basic Environment"
run date
run hostname || true
run uname -a
echo ""
echo "+ id"
id 2>/dev/null || true
echo ""
echo "+ pwd"
pwd

hr "Key Environment Variables (trimmed)"
for v in \
    PATH \
    LD_LIBRARY_PATH \
    LIBRARY_PATH \
    CPATH \
    C_INCLUDE_PATH \
    CPLUS_INCLUDE_PATH \
    CUDA_HOME \
    CUDA_PATH \
    CUDA_ROOT \
    CUDACXX \
    NVCC \
    CC \
    CXX \
    VIRTUAL_ENV \
    CONDA_PREFIX \
    LOADEDMODULES \
    LMOD_SYSTEM_NAME \
    MODULEPATH \
    FLASHINFER_CACHE_DIR \
    TORCH_CUDA_ARCH_LIST \
    VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE \
    VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM \
    RAY_CGRAPH_get_timeout \
; do
    if [ -n "${!v:-}" ]; then
        val="${!v}"
        if [ ${#val} -gt 400 ]; then
            val="${val:0:400}...[truncated]"
        fi
        printf '%s=%q\n' "$v" "$val"
    fi
done

hr "Modules (if available)"
try_enable_modules
if type module >/dev/null 2>&1; then
    run module --version || true
    # Lmod supports both `module list` and `module -t list`. Environment Modules varies.
    run module list || true
else
    echo "module command not available in this shell."
fi

if [ "$PROBE_CUDA_MODULES" = "1" ]; then
    hr "CUDA Module Probe"
    probe_cuda_modules
fi

if [ -n "$VENV_TO_SOURCE" ]; then
    hr "Virtualenv Activation"
    echo "+ source \"$VENV_TO_SOURCE/bin/activate\""
    if [ -f "$VENV_TO_SOURCE/bin/activate" ]; then
        # shellcheck disable=SC1090
        source "$VENV_TO_SOURCE/bin/activate"
        echo "VENV_ACTIVE=1"
    else
        echo "[missing] $VENV_TO_SOURCE/bin/activate"
        echo "VENV_ACTIVE=0"
    fi
fi

hr "GPU / Driver"
maybe_run nvidia-smi -L
maybe_run nvidia-smi

hr "Build Toolchain"
maybe_run gcc --version
maybe_run g++ --version
maybe_run ninja --version
maybe_run cmake --version
maybe_run make --version

hr "CUDA Toolkit Detection"
maybe_run nvcc --version
maybe_run which nvcc
CUDA_ROOT_DETECTED="$(detect_cuda_root)"
echo ""
echo "CUDA_ROOT_DETECTED=${CUDA_ROOT_DETECTED:-<empty>}"
if [ -n "${CUDA_ROOT_DETECTED:-}" ]; then
    echo ""
    echo "+ ls -la \"${CUDA_ROOT_DETECTED}/include\" | head -n 50"
    ls -la "${CUDA_ROOT_DETECTED}/include" 2>&1 | head -n 50 || true

    echo ""
    echo "+ test -e \"${CUDA_ROOT_DETECTED}/include/cuda/ptx\" && echo FOUND || echo MISSING"
    if test -e "${CUDA_ROOT_DETECTED}/include/cuda/ptx"; then
        echo "FOUND: ${CUDA_ROOT_DETECTED}/include/cuda/ptx"
        echo ""
        echo "+ ls -la \"${CUDA_ROOT_DETECTED}/include/cuda/ptx\""
        ls -la "${CUDA_ROOT_DETECTED}/include/cuda/ptx" 2>&1 || true
    else
        echo "MISSING: ${CUDA_ROOT_DETECTED}/include/cuda/ptx"
        echo ""
        echo "+ find \"${CUDA_ROOT_DETECTED}/include\" -maxdepth 5 -path \"*/cuda/ptx*\" -print | head -n 50"
        find "${CUDA_ROOT_DETECTED}/include" -maxdepth 5 -path "*/cuda/ptx*" -print 2>&1 | head -n 50 || true
    fi

    echo ""
    echo "+ test -d \"${CUDA_ROOT_DETECTED}/include/cccl\" && echo FOUND || echo MISSING"
    if test -d "${CUDA_ROOT_DETECTED}/include/cccl"; then
        echo "FOUND: ${CUDA_ROOT_DETECTED}/include/cccl"
    else
        echo "MISSING: ${CUDA_ROOT_DETECTED}/include/cccl"
    fi

    if command -v nvcc >/dev/null 2>&1; then
        hr "nvcc Include Search Paths (preprocessor)"
        echo "+ echo | nvcc -x cu -E -v -"
        set +e
        echo | nvcc -x cu -E -v - 2>&1 | sed -n '/#include <...> search starts here:/,/End of search list./p'
        set -e
    fi
else
    echo "Could not detect a CUDA toolkit root with headers (no nvcc and no CUDA_HOME/CUDA_PATH)."
fi

hr "Python / Packages"
maybe_run python -V
maybe_run which python
maybe_run python -c "import sys; print('executable=', sys.executable); print('version=', sys.version.replace('\\n',' '))"
maybe_run python -c "import torch; print('torch=', torch.__version__); print('torch.cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('arch=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('capability=', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
maybe_run python -c "import vllm; print('vllm=', getattr(vllm, '__version__', '<no __version__>'))"
maybe_run python -c "import flashinfer; print('flashinfer=', getattr(flashinfer, '__version__', '<no __version__>')); import flashinfer.jit as fj; print('flashinfer.jit=', fj.__file__)"
maybe_run python -m pip --version
maybe_run python -m pip show flashinfer-python
maybe_run python -m pip show flashinfer-cubin
maybe_run python -m pip show vllm

hr "FlashInfer Cache (if present)"
echo "+ echo \"FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-<unset>}\""
echo "FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-<unset>}"
echo ""
echo "+ ls -la \"${FLASHINFER_CACHE_DIR:-$HOME/.cache/flashinfer}\" | head -n 80"
ls -la "${FLASHINFER_CACHE_DIR:-$HOME/.cache/flashinfer}" 2>&1 | head -n 80 || true
echo ""
echo "+ find \"${FLASHINFER_CACHE_DIR:-$HOME/.cache/flashinfer}\" -maxdepth 4 -type d \\( -name \"gdn_prefill*\" -o -name \"cached_ops\" -o -name \"generated\" \\) -print | head -n 80"
find "${FLASHINFER_CACHE_DIR:-$HOME/.cache/flashinfer}" -maxdepth 4 -type d \( -name "gdn_prefill*" -o -name "cached_ops" -o -name "generated" \) -print 2>&1 | head -n 80 || true

hr "Done"
if command -v python >/dev/null 2>&1; then
    set +e
    python -c "import torch, vllm, flashinfer" >/dev/null 2>&1
    py_pkgs_ec=$?
    set -e
    if [ $py_pkgs_ec -ne 0 ]; then
        echo "NOTE: Python packages (torch/vllm/flashinfer) are not importable in this environment."
        if [ -z "$VENV_TO_SOURCE" ]; then
            echo "      Re-run with: --venv /path/to/venv"
        fi
    fi
fi
echo "If you hit a FlashInfer build error, the most important line is whether cuda/ptx is FOUND under CUDA_ROOT_DETECTED (or in the CUDA Module Probe)."
