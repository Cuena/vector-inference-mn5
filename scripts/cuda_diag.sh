#!/usr/bin/env bash
set -euo pipefail

IMG="${1:-}"
if [[ -z "$IMG" ]]; then
  echo "Usage: $0 /path/to/image.sif"
  exit 2
fi
if [[ ! -f "$IMG" ]]; then
  echo "ERROR: file not found: $IMG"
  exit 2
fi

# Avoid inheriting stale experiments
unset LD_PRELOAD || true

echo "== SIF diag =="
echo "Image: $IMG"
echo

singularity exec --nv "$IMG" bash --noprofile --norc -lc '
set -euo pipefail

echo "== Host kernel driver (from /proc inside container) =="
cat /proc/driver/nvidia/version 2>/dev/null | head -n 2 || echo "No /proc/driver/nvidia/version"
echo

echo "== GPU device nodes visible? =="
ls -l /dev/nvidia* 2>/dev/null | head -n 12 || echo "No /dev/nvidia* (no --nv or no GPU allocation)"
echo

PYBIN="$(command -v python3 || command -v python || true)"
echo "== Python binary =="
if [[ -z "$PYBIN" ]]; then
  echo "Python not found in image. Skipping torch/loader checks."
  exit 0
fi
echo "PYBIN=$PYBIN"
echo

echo "== Torch sanity (if installed) =="
"$PYBIN" - <<PY 2>/dev/null || echo "torch not importable in this image (ok)"
try:
    import torch
    print("torch", torch.__version__)
    print("torch.version.cuda", torch.version.cuda)
    print("cuda.is_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device0", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch import failed:", e)
PY
echo

echo "== Locate compat directories =="
find /usr/local -maxdepth 5 -type d -name compat 2>/dev/null | sed "s/^/  /" || true
echo

echo "== Locate PTX JIT + libcuda inside image (top hits) =="
PTX_HITS="$(find /usr/local -maxdepth 8 -type f -name "libnvidia-ptxjitcompiler.so.1" 2>/dev/null | head -n 20 || true)"
CUDA_HITS="$(find /usr/local -maxdepth 8 -type f -name "libcuda.so.1" 2>/dev/null | head -n 20 || true)"
echo "PTXJIT:"
echo "${PTX_HITS:-  <none found under /usr/local>}" | sed "s/^/  /"
echo "libcuda:"
echo "${CUDA_HITS:-  <none found under /usr/local>}" | sed "s/^/  /"
echo

# Helper: run loader probe and print where init comes from
probe() {
  LD_DEBUG=libs "$PYBIN" -c "import ctypes; ctypes.CDLL(\"libnvidia-ptxjitcompiler.so.1\"); ctypes.CDLL(\"libcuda.so.1\"); print(\"ok\")" 2>&1 \
  | grep -E "calling init: .*libnvidia-ptxjitcompiler|calling init: .*libcuda|\\.singularity\\.d/libs|compat" \
  | head -n 120 || true
}

echo "== Loader probe: default environment =="
probe
echo

# Choose best compat dir for LD_LIBRARY_PATH (prefer compat/lib.real, then compat)
BEST_COMPAT=""
for cand in \
  /usr/local/cuda-*/compat/lib.real \
  /usr/local/cuda/compat/lib.real \
  /usr/local/cuda-*/compat \
  /usr/local/cuda/compat
do
  if [[ -d "$cand" ]] && ls "$cand"/libnvidia-ptxjitcompiler.so.1 >/dev/null 2>&1; then
    BEST_COMPAT="$cand"
    break
  fi
done

if [[ -n "$BEST_COMPAT" ]]; then
  echo "== Try fix A: LD_LIBRARY_PATH prepend =="
  echo "Candidate: $BEST_COMPAT"
  export LD_LIBRARY_PATH="$BEST_COMPAT:${LD_LIBRARY_PATH:-}"
  probe
  echo
else
  echo "== Try fix A: LD_LIBRARY_PATH prepend =="
  echo "No compat dir with libnvidia-ptxjitcompiler.so.1 found (skipping)"
  echo
fi

# Pick a specific PTXJIT to preload (first hit if any)
PTX_ONE="$(echo "$PTX_HITS" | head -n 1 || true)"
if [[ -n "$PTX_ONE" ]]; then
  echo "== Try fix B: LD_PRELOAD PTXJIT only =="
  echo "Candidate: $PTX_ONE"
  export LD_PRELOAD="$PTX_ONE"
  probe
  echo
else
  echo "== Try fix B: LD_PRELOAD PTXJIT only =="
  echo "No libnvidia-ptxjitcompiler.so.1 found under /usr/local (skipping)"
  echo
fi

echo "== Recommendation (what to set for future runs) =="
# Heuristic: prefer LD_LIBRARY_PATH to BEST_COMPAT if it yields init from compat; else prefer LD_PRELOAD if it yields init from compat.
# We can’t perfectly parse here without extra tooling, so we print both options with guidance.

if [[ -n "$BEST_COMPAT" ]]; then
  echo "Option A (preferred if it makes init come from compat):"
  echo "  export SINGULARITYENV_LD_LIBRARY_PATH='$BEST_COMPAT:\$LD_LIBRARY_PATH'"
fi

if [[ -n "$PTX_ONE" ]]; then
  echo "Option B (force PTXJIT; often enough even if libcuda stays host):"
  echo "  export SINGULARITYENV_LD_PRELOAD='$PTX_ONE'"
fi

echo
echo "Then run:"
echo "  singularity exec --nv \"'"$IMG"'\" <your command>"
'