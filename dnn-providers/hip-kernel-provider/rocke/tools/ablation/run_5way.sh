#!/usr/bin/env bash
set -Eeuo pipefail

# ONE shell launch -> ONE Python process -> ONE HIP stream for all five variants.
GPU_BENCH_ROOT="${GPU_BENCH_ROOT:-/root/gpu-bench}"
ROCM_REPO="${ROCM_REPO:-$GPU_BENCH_ROOT/rocm-libraries}"
ROCKE_ROOT="${ROCKE_ROOT:-$ROCM_REPO/dnn-providers/hip-kernel-provider/rocke}"
VENV="${ROCKE_VENV:-$GPU_BENCH_ROOT/venvs/rocke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/5way}"

[[ -f "$ROCKE_ROOT/library/kernels/gfx942/attention_dense.py" ]] || {
  echo "ERROR: invalid ROCKE_ROOT: $ROCKE_ROOT" >&2; exit 2;
}
[[ -f "$VENV/bin/activate" ]] || {
  echo "ERROR: venv not found: $VENV" >&2; exit 2;
}
[[ -f "$SCRIPT_DIR/attention_dense_ablation.py" ]] || {
  echo "ERROR: missing $SCRIPT_DIR/attention_dense_ablation.py" >&2; exit 2;
}
[[ -f "$SCRIPT_DIR/bench_rocke_5way.py" ]] || {
  echo "ERROR: missing $SCRIPT_DIR/bench_rocke_5way.py" >&2; exit 2;
}

source "$VENV/bin/activate"

# Keep PyTorch + ROCKE on the same wheel ROCm stack.  This avoids the
# "spirv-expand-step registered more than once" abort from loading two LLVM stacks.
SITE="$(python - <<'PY'
import site
ps = [p for p in site.getsitepackages() if p.endswith('site-packages')]
print(ps[0])
PY
)"
CORE_LIB="$SITE/_rocm_sdk_core/lib"

[[ -f "$CORE_LIB/libamd_comgr.so.3" ]] || {
  echo "ERROR: missing $CORE_LIB/libamd_comgr.so.3" >&2; exit 2;
}
[[ -f "$CORE_LIB/libamdhip64.so.7" ]] || {
  echo "ERROR: missing $CORE_LIB/libamdhip64.so.7" >&2; exit 2;
}

export ROCKE_BACKEND=python
export ROCKE_LLVM_FLAVOR="${ROCKE_LLVM_FLAVOR:-llvm23}"
export ROCKE_COMGR_LIB="$CORE_LIB/libamd_comgr.so.3"
export ROCKE_HIP_LIB="$CORE_LIB/libamdhip64.so.7"
export LD_LIBRARY_PATH="$CORE_LIB"
unset LD_PRELOAD

export ROCKE_ROOT
export PYTHONPATH="$ROCKE_ROOT/library:$ROCKE_ROOT/platform/Python:$ROCKE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT_DIR"

echo "ROCKE_ROOT=$ROCKE_ROOT"
echo "ROCKE_BACKEND=$ROCKE_BACKEND"
echo "ROCKE_LLVM_FLAVOR=$ROCKE_LLVM_FLAVOR"
echo "ROCKE_COMGR_LIB=$ROCKE_COMGR_LIB"
echo "ROCKE_HIP_LIB=$ROCKE_HIP_LIB"
echo "OUT_DIR=$OUT_DIR"

# Intentionally exactly ONE Python invocation.
python "$SCRIPT_DIR/bench_rocke_5way.py" \
  --rocke-root "$ROCKE_ROOT" \
  --shape "${SHAPE:-1,4096,32,8,128,bf16}" \
  --persistent "${PERSISTENT:-auto}" \
  --num-persistent "${NUM_PERSISTENT:-304}" \
  --warmup "${WARMUP:-10}" \
  --rounds "${ROUNDS:-5}" \
  --repeat "${REPEAT:-50}" \
  --out-dir "$OUT_DIR" \
  "$@"