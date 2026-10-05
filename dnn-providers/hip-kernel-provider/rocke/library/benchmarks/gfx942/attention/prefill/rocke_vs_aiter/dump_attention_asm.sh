#!/usr/bin/env bash
set -Eeuo pipefail

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ROOT="/root/gpu-bench"
ROCKE_VENV="$ROOT/venvs/rocke"
ROCKE_ENV="$ROOT/env/rocke_env.sh"

BENCH_DIR="$ROOT/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter"

AITER_CO="$ROOT/aiter/hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16_causal_rtne.co"

LLVM_OBJDUMP="/opt/rocm/llvm/bin/llvm-objdump"
LLVM_NM="/opt/rocm/llvm/bin/llvm-nm"

ROCKE_SYMBOL="rocke_attention_dense_d128_hq32_kv8_bn64_bf16_sq4096_sk4096_causal_lazyrs_persist304_gfx942_b1_wpe2_kdbvpf1"

ROCKE_ASM="$BENCH_DIR/rocke_kdbvpf1.s"
AITER_ASM="$BENCH_DIR/aiter_fmha_rtne.s"


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

source "$ROCKE_VENV/bin/activate"
source "$ROCKE_ENV"

cd "$BENCH_DIR"


# ------------------------------------------------------------
# Clean old HIP code-object dumps
# ------------------------------------------------------------

rm -f ./_code_object*.o


# ------------------------------------------------------------
# Run ROCKE and dump JIT code objects
# ------------------------------------------------------------

echo "=== Running ROCKE and dumping code objects ==="

GPU_DUMP_CODE_OBJECT=1 \
AMD_COMGR_SAVE_TEMPS=1 \
python bench_rocke.py


# ------------------------------------------------------------
# Find exact ROCKE code object
# ------------------------------------------------------------

echo
echo "=== Searching for exact ROCKE kernel ==="

ROCKE_OBJ=""

for f in ./_code_object*.o; do
    [ -e "$f" ] || continue

    if "$LLVM_NM" "$f" 2>/dev/null | grep -Fq "$ROCKE_SYMBOL"; then
        ROCKE_OBJ="$f"
        break
    fi
done

if [ -z "$ROCKE_OBJ" ]; then
    echo "ERROR: could not find code object containing:"
    echo "$ROCKE_SYMBOL"
    exit 1
fi

echo "Found:"
echo "  $ROCKE_OBJ"

"$LLVM_NM" "$ROCKE_OBJ" | grep -F "$ROCKE_SYMBOL"


# ------------------------------------------------------------
# Disassemble ROCKE
# ------------------------------------------------------------

echo
echo "=== Disassembling ROCKE ==="

"$LLVM_OBJDUMP" \
    --disassemble-all \
    --mcpu=gfx942 \
    --no-show-raw-insn \
    "$ROCKE_OBJ" \
    > "$ROCKE_ASM"


# ------------------------------------------------------------
# Disassemble AITER
# ------------------------------------------------------------

echo
echo "=== Disassembling AITER ==="

if [ ! -f "$AITER_CO" ]; then
    echo "ERROR: AITER code object not found:"
    echo "$AITER_CO"
    exit 1
fi

"$LLVM_OBJDUMP" \
    --disassemble-all \
    --mcpu=gfx942 \
    --no-show-raw-insn \
    "$AITER_CO" \
    > "$AITER_ASM"


# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------

echo
echo "=== Verification ==="

if ! grep -Fq "$ROCKE_SYMBOL" "$ROCKE_ASM"; then
    echo "ERROR: ROCKE symbol not found in generated assembly"
    exit 1
fi

echo
ls -lh "$ROCKE_ASM" "$AITER_ASM"

echo
echo "ROCKE kernel:"
grep -n -F "$ROCKE_SYMBOL" "$ROCKE_ASM" | head -1

echo
echo "AITER kernel:"
grep -n 'fmha_fwd_hd128_bf16_causal_rtne' "$AITER_ASM" | head -1 || true

echo
echo "Done."
echo "ROCKE: $ROCKE_ASM"
echo "AITER: $AITER_ASM"