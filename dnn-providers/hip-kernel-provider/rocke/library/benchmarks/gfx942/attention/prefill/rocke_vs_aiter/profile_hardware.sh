#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/gpu-bench
REPO="$ROOT/rocm-libraries"
AITER="$ROOT/aiter"
VENV="$ROOT/venvs"
ENV="$ROOT/env"

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$HERE/profiling/first_shape"
ROCKE_HELPER="$BASE/rocke_first_shape.py"
OUT="$BASE/hardware_counters"

AITER_ROCPROF="$VENV/aiter/bin/rocprofv3"
ROCKE_ROCPROF="$VENV/rocke/bin/rocprofv3"

[ -x "$AITER_ROCPROF" ] || { echo "missing $AITER_ROCPROF"; exit 1; }
[ -x "$ROCKE_ROCPROF" ] || { echo "missing $ROCKE_ROCPROF"; exit 1; }
[ -x "$AITER/op_tests/cpp/mha/fwd.exe" ] || { echo "missing AITER fwd.exe"; exit 1; }
[ -f "$ROCKE_HELPER" ] || { echo "missing $ROCKE_HELPER; run the single-dispatch profile.sh first"; exit 1; }

mkdir -p "$OUT/aiter" "$OUT/rocke"

# Focused bottleneck set for MI300X attention.
# Each counter is its own rocprofv3 job/pass. This avoids counter-group
# compatibility failures and makes pass_N -> counter mapping unambiguous.
COUNTERS=(
  MfmaUtil
  MfmaFlopsBF16
  OccupancyPercent
  MeanOccupancyPerActiveCU
  SQ_WAVES
  SQ_WAVE_CYCLES
  SQ_BUSY_CU_CYCLES

  LdsUtil
  LdsBankConflict
  LdsLatency
  SQ_INSTS_LDS
  SQ_WAIT_INST_LDS
  SQ_LDS_BANK_CONFLICT

  MemUnitStalled
  VmemLatency
  SQ_INSTS_VMEM_RD
  SQ_INSTS_VMEM_WR
  SQ_INST_CYCLES_VMEM_RD
  SQ_INST_CYCLES_VMEM_WR

  SQ_INSTS_MFMA
  SQ_INSTS_VALU_MFMA_BF16
  SQ_VALU_MFMA_BUSY_CYCLES
  SQ_INSTS_VALU
  SQ_INSTS_SALU
  SQ_INSTS_SMEM

  TCC_HIT_sum
  TCC_MISS_sum
  TCC_EA0_RDREQ_DRAM_sum
  TCC_EA0_WRREQ_DRAM_sum

  FetchSize
  WriteSize
  BANDWIDTH_EA
)

write_yaml() {
  local path="$1"
  local regex="$2"

  {
    echo "jobs:"
    for c in "${COUNTERS[@]}"; do
      echo "  - pmc: [\"$c\"]"
      echo "    kernel_include_regex: \"$regex\""
    done
  } > "$path"
}

write_map() {
  local path="$1"
  {
    printf "pass,counter\n"
    local i=1
    for c in "${COUNTERS[@]}"; do
      printf "pass_%d,%s\n" "$i" "$c"
      i=$((i + 1))
    done
  } > "$path"
}

AITER_YAML="$OUT/aiter_counters.yaml"
ROCKE_YAML="$OUT/rocke_counters.yaml"

write_yaml "$AITER_YAML" "fmha_fwd_hd128_bf16_causal_rtz"
write_yaml "$ROCKE_YAML" "rocke_attention_dense"
write_map "$OUT/pass_map.csv"

rm -rf "$OUT/aiter"/pass_* "$OUT/rocke"/pass_*

echo "============================================================"
echo "AITER hardware counters"
echo "============================================================"

bash -lc "
source '$ENV/aiter_env.sh'
cd '$AITER/op_tests/cpp/mha'
'$AITER_ROCPROF' \
  -i '$AITER_YAML' \
  --output-format csv \
  --output-directory '$OUT/aiter' \
  -- \
  ./fwd.exe \
    -prec=bf16 \
    -b=1 -h=32 -h_k=8 \
    -d=128 -d_v=128 \
    -s=4096 -s_k=4096 \
    -iperm=0 -operm=0 \
    -mask=1 -lse=0 \
    -fwd_v3=1 \
    -v3_bf16_cvt=2 \
    -mode=0 \
    -timer=gpu \
    -kname=1 \
    -v=0 \
    -warmup=0 \
    -repeat=1
" 2>&1 | tee "$OUT/aiter/rocprofv3_counters.log"

echo "============================================================"
echo "ROCKE hardware counters"
echo "============================================================"

bash -lc "
source '$ENV/rocke_env.sh'
cd '$REPO/dnn-providers/hip-kernel-provider'
'$ROCKE_ROCPROF' \
  -i '$ROCKE_YAML' \
  --output-format csv \
  --output-directory '$OUT/rocke' \
  -- \
  '$VENV/rocke/bin/python' '$ROCKE_HELPER'
" 2>&1 | tee "$OUT/rocke/rocprofv3_counters.log"

cat > "$OUT/README.txt" <<EOF
Hardware-counter collection for first attention shape.

Shape:
B=1
S=4096
Hq=32
Hkv=8
D=128
dtype=BF16
causal=true
layout=BSHD

Each PMC is collected as its own rocprofv3 YAML job/pass.
This intentionally avoids mixing incompatible counters in one hardware pass.

AITER kernel filter:
fmha_fwd_hd128_bf16_causal_rtz

ROCKE kernel filter:
rocke_attention_dense

See pass_map.csv for pass_N -> counter mapping.
EOF

echo
echo "============================================================"
echo "COUNTER COLLECTION FILES"
echo "============================================================"
find "$OUT" -name '*counter_collection.csv' -type f | sort

echo
echo "Pass mapping:"
cat "$OUT/pass_map.csv"
