#!/usr/bin/env bash
set -Eeuo pipefail

# One shell launch -> one Python process -> one HIP context/stream for all five variants.
# Defaults match the user's MI300X workspace; override any path with env vars.
GPU_BENCH_ROOT="${GPU_BENCH_ROOT:-/root/gpu-bench}"
ROCM_REPO="${ROCM_REPO:-$GPU_BENCH_ROOT/rocm-libraries}"
ROCKE_ROOT="${ROCKE_ROOT:-$ROCM_REPO/dnn-providers/hip-kernel-provider/rocke}"
VENV="${ROCKE_VENV:-$GPU_BENCH_ROOT/venvs/rocke}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/ablation}"

if [[ ! -f "$ROCKE_ROOT/library/kernels/gfx942/attention_dense.py" ]]; then
  echo "ERROR: ROCKE_ROOT does not look valid: $ROCKE_ROOT" >&2
  exit 2
fi
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "ERROR: venv not found: $VENV" >&2
  exit 2
fi

source "$VENV/bin/activate"

# IMPORTANT: PyTorch in this venv is built against the pip ROCm SDK. ROCKE must
# load HIP + COMGR from that SAME SDK. Mixing these with /opt/rocm in one Python
# process aborts LLVM with e.g. "spirv-expand-step registered more than once".
SITE="$VENV/lib/python3.12/site-packages"
CORE_LIB="$SITE/_rocm_sdk_core/lib"

if [[ ! -f "$CORE_LIB/libamd_comgr.so.3" ]]; then
  echo "ERROR: wheel ROCm COMGR not found: $CORE_LIB/libamd_comgr.so.3" >&2
  exit 2
fi
if [[ ! -f "$CORE_LIB/libamdhip64.so.7" ]]; then
  echo "ERROR: wheel ROCm HIP runtime not found: $CORE_LIB/libamdhip64.so.7" >&2
  exit 2
fi

export ROCKE_BACKEND=python
export ROCKE_LLVM_FLAVOR=llvm23
export ROCKE_COMGR_LIB="$CORE_LIB/libamd_comgr.so.3"
export ROCKE_HIP_LIB="$CORE_LIB/libamdhip64.so.7"

# Deliberately do NOT append a pre-existing /opt/rocm LD_LIBRARY_PATH here.
# The whole process must see one ROCm/LLVM stack.
export LD_LIBRARY_PATH="$CORE_LIB"
unset LD_PRELOAD

# Keep imports explicit and reproducible.
export ROCKE_ROOT
export PYTHONPATH="$ROCKE_ROOT/library:$ROCKE_ROOT/platform/Python:$ROCKE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "ROCKE_BACKEND=$ROCKE_BACKEND"
echo "ROCKE_LLVM_FLAVOR=$ROCKE_LLVM_FLAVOR"
echo "ROCKE_COMGR_LIB=$ROCKE_COMGR_LIB"
echo "ROCKE_HIP_LIB=$ROCKE_HIP_LIB"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

mkdir -p "$OUT_DIR"

# IMPORTANT: this is intentionally ONE Python invocation. Do not split variants into
# separate python commands; that would violate the same-process/same-stream evidence rule.
python "$SCRIPT_DIR/bench_rocke_ablation.py" \
  --rocke-root "$ROCKE_ROOT" \
  --shape "${SHAPE:-1,4096,32,8,128,bf16}" \
  --persistent "${PERSISTENT:-auto}" \
  --num-persistent "${NUM_PERSISTENT:-304}" \
  --warmup "${WARMUP:-10}" \
  --rounds "${ROUNDS:-5}" \
  --repeat "${REPEAT:-50}" \
  --out-dir "$OUT_DIR" \
  "$@"