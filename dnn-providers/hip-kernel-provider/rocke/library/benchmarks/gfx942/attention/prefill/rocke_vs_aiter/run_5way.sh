#!/usr/bin/env bash
set -Eeuo pipefail

GPU_BENCH_ROOT="${GPU_BENCH_ROOT:-/root/gpu-bench}"
ROCM_REPO="${ROCM_REPO:-$GPU_BENCH_ROOT/rocm-libraries}"
ROCKE_ROOT="${ROCKE_ROOT:-$ROCM_REPO/dnn-providers/hip-kernel-provider/rocke}"
ROCKE_VENV="${ROCKE_VENV:-$GPU_BENCH_ROOT/venvs/rocke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/5way_specs}"

[[ -f "$ROCKE_ROOT/library/kernels/gfx942/attention_dense.py" ]] || {
  echo "Missing $ROCKE_ROOT/library/kernels/gfx942/attention_dense.py" >&2; exit 2;
}
[[ -f "$ROCKE_VENV/bin/activate" ]] || {
  echo "Missing Python virtualenv $ROCKE_VENV" >&2; exit 2;
}
source "$ROCKE_VENV/bin/activate"

SITE="$(python - <<'PY'
import site
packages = [p for p in site.getsitepackages() if p.endswith('site-packages')]
if not packages:
    raise SystemExit('Could not find Python site-packages')
print(packages[0])
PY
)"
CORE_LIB="$SITE/_rocm_sdk_core/lib"
[[ -f "$CORE_LIB/libamd_comgr.so.3" && -f "$CORE_LIB/libamdhip64.so.7" ]] || {
  echo "Expected wheel ROCm shared libraries not found in $CORE_LIB" >&2; exit 2;
}
export ROCKE_ROOT
export ROCKE_BACKEND=python
export ROCKE_LLVM_FLAVOR="${ROCKE_LLVM_FLAVOR:-llvm23}"
export ROCKE_COMGR_LIB="$CORE_LIB/libamd_comgr.so.3"
export ROCKE_HIP_LIB="$CORE_LIB/libamdhip64.so.7"
export LD_LIBRARY_PATH="$CORE_LIB"
unset LD_PRELOAD
export PYTHONPATH="$ROCKE_ROOT/library:$ROCKE_ROOT/platform/Python:$ROCKE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# One shell launch -> one Python process -> one current HIP stream.
exec python "$SCRIPT_DIR/bench_rocke_5way_specs.py" \
  --rocke-root "$ROCKE_ROOT" \
  --out-dir "$OUT_DIR" \
  "$@"