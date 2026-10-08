#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/gpu-bench
REPO="$ROOT/rocm-libraries"
AITER="$ROOT/aiter"
ENV="$ROOT/env"
VENV="$ROOT/venvs"

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/profiling/eight_shapes"

AITER_ROCPROF="$VENV/aiter/bin/rocprofv3"
ROCKE_ROCPROF="$VENV/rocke/bin/rocprofv3"
AVAIL="$VENV/rocke/bin/rocprofv3-avail"

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
PROFILE="${PROFILE:-1}"

# B  S  Hkv
SHAPES=(
  "1 4096 8"
  "1 4096 16"
  "1 8192 8"
  "1 8192 16"
  "1 16384 8"
  "16 4096 8"
  "16 4096 16"
  "16 8192 8"
)

HQ=32
D=128

[ -x "$AITER_ROCPROF" ] || { echo "missing $AITER_ROCPROF"; exit 1; }
[ -x "$ROCKE_ROCPROF" ] || { echo "missing $ROCKE_ROCPROF"; exit 1; }
[ -x "$VENV/rocke/bin/python" ] || { echo "missing $VENV/rocke/bin/python"; exit 1; }
[ -x "$AITER/op_tests/cpp/mha/fwd.exe" ] || { echo "missing AITER fwd.exe"; exit 1; }
[ -f "$ENV/aiter_env.sh" ] || { echo "missing $ENV/aiter_env.sh"; exit 1; }
[ -f "$ENV/rocke_env.sh" ] || { echo "missing $ENV/rocke_env.sh"; exit 1; }

mkdir -p "$OUT"
rm -f "$OUT/summary.csv"

if [ -x "$AVAIL" ]; then
    "$AVAIL" list --pmc > "$OUT/available_counters.txt"
fi

cat > "$OUT/rocke_shape.py" <<'PY'
#!/usr/bin/env python3
import argparse
import math
import torch

from builders.gfx942.attention.prefill.attention_dense_prefill import (
    dense_request,
    resolve_dense_spec,
    describe_dense_spec,
    _make_launcher,
    _launch_config,
)
from kernels.gfx942.attention_dense import supports_attention_dense
from rocke.runtime import time_launches


cli = argparse.ArgumentParser()
cli.add_argument("--batch", type=int, required=True)
cli.add_argument("--seqlen", type=int, required=True)
cli.add_argument("--hkv", type=int, required=True)
cli.add_argument("--run-mode", choices=("benchmark", "single"), required=True)
cli.add_argument("--warmup", type=int, default=10)
cli.add_argument("--repeat", type=int, default=50)
opt = cli.parse_args()

B = opt.batch
SQ = opt.seqlen
SK = opt.seqlen
HQ = 32
HKV = opt.hkv
D = 128

args = argparse.Namespace(
    persistent=None,
    num_persistent=None,
    persist_decode=None,
    block_n=None,
    waves_per_eu=None,
    interleave=None,
    lds_k_group_pad=None,
    sliding_window=None,
)

req = dense_request(
    args,
    batch=B,
    seqlen_q=SQ,
    seqlen_kv=SK,
    num_query_heads=HQ,
    num_kv_heads=HKV,
    head_size=D,
    causal=True,
    dtype="bf16",
)

# Use dispatch-resolved BN64. Do not override block_n.
# K double buffering + V-next VGPR prefetch are implementation policy
# inside the modified gfx942 attention_dense.py.
spec = resolve_dense_spec(req, {})

if spec.block_n != 64:
    raise SystemExit(
        f"ERROR: expected dispatch block_n=64, got block_n={spec.block_n} "
        f"for B={B} S={SQ} Hkv={HKV}"
    )

if not hasattr(spec, "resolved_use_cfvst"):
    raise SystemExit(
        "ERROR: resolved spec is not the expected Gfx942AttentionDenseSpec"
    )

if not spec.resolved_use_cfvst():
    raise SystemExit(
        "ERROR: CFVST path is not active; "
        "K-double-buffer/V-prefetch pipeline would not be selected"
    )

ok, why = supports_attention_dense(spec, arch="gfx942")
if not ok:
    raise SystemExit(f"unsupported: {why}")

name = describe_dense_spec(spec)

if "_bn64_" not in name:
    raise SystemExit(f"ERROR: expected _bn64_ in kernel name, got: {name}")

if "_kdbvpf1" not in name:
    raise SystemExit(
        f"ERROR: expected _kdbvpf1 pipeline tag in kernel name, got: {name}"
    )

print(
    f"ROCKE shape: B={B} S={SQ} Hq={HQ} Hkv={HKV} D={D} "
    "dtype=BF16 causal=true"
)
print("ROCKE kernel:", name)
print(f"ROCKE experiment: block_n={spec.block_n}")
print("ROCKE experiment: K LDS buffers=2")
print("ROCKE experiment: V LDS buffers=1")
print("ROCKE experiment: V[j+1] prefetch=VGPR")

torch.manual_seed(0)
dt = torch.bfloat16

q = (torch.randn(B, SQ, HQ, D, dtype=dt, device="cuda") * 0.2).contiguous()
k = (torch.randn(B, SK, HKV, D, dtype=dt, device="cuda") * 0.2).contiguous()
v = (torch.randn(B, SK, HKV, D, dtype=dt, device="cuda") * 0.2).contiguous()
out = torch.zeros(B, SQ, HQ, D, dtype=dt, device="cuda")

scale = 1.0 / math.sqrt(D)
launcher = _make_launcher(spec)
stream = int(torch.cuda.current_stream().cuda_stream)
cfg = _launch_config(spec, stream)

vals = {
    "q_ptr": q,
    "k_ptr": k,
    "v_ptr": v,
    "o_ptr": out,
    "scale": scale,
}

def call():
    launcher(vals, config=cfg)

if opt.run_mode == "benchmark":
    ms = time_launches(
        call,
        warmup=opt.warmup,
        iters=opt.repeat,
        stream=stream,
    )
    torch.cuda.synchronize()
    print(
        f"ROCKE benchmark: warmup={opt.warmup} "
        f"repeat={opt.repeat} avg={ms:.6f} ms"
    )
else:
    # Exactly one target attention dispatch for rocprofv3.
    call()
    torch.cuda.synchronize()
PY

echo "B,S,Hq,Hkv,D,aiter_ms,rocke_ms,rocke_over_aiter" > "$OUT/summary.csv"

echo "============================================================"
echo "STABLE UNPROFILED PERFORMANCE"
echo "AITER and ROCKE: ${WARMUP} warmup + ${REPEAT} repeat"
echo "============================================================"

for shape in "${SHAPES[@]}"; do
    read -r B S HKV <<< "$shape"
    TAG="b${B}_s${S}_hq${HQ}_hkv${HKV}_d${D}"
    SDIR="$OUT/$TAG"
    mkdir -p "$SDIR/aiter" "$SDIR/rocke"

    echo
    echo "============================================================"
    echo "SHAPE: B=$B S=$S Hq=$HQ Hkv=$HKV D=$D BF16 causal"
    echo "============================================================"

    echo "--- AITER benchmark ---"
    bash -lc "
        source '$ENV/aiter_env.sh'
        cd '$AITER/op_tests/cpp/mha'
        ./fwd.exe \
          -prec=bf16 \
          -b=$B -h=$HQ -h_k=$HKV \
          -d=$D -d_v=$D \
          -s=$S -s_k=$S \
          -iperm=0 -operm=0 \
          -mask=1 -lse=0 \
          -fwd_v3=1 \
          -v3_bf16_cvt=0 \
          -mode=0 \
          -timer=gpu \
          -kname=1 \
          -v=0 \
          -warmup=$WARMUP \
          -repeat=$REPEAT
    " 2>&1 | tee "$SDIR/aiter/benchmark.log"

    AITER_MS="$(
        grep -Eo ', [0-9]+([.][0-9]+)? ms' "$SDIR/aiter/benchmark.log" \
        | tail -n 1 \
        | awk '{print $2}'
    )"

    [ -n "$AITER_MS" ] || {
        echo "ERROR: could not parse AITER latency for $TAG"
        exit 1
    }

    echo
    echo "--- ROCKE benchmark ---"
    bash -lc "
        source '$ENV/rocke_env.sh'
        cd '$REPO/dnn-providers/hip-kernel-provider'
        '$VENV/rocke/bin/python' \
          '$OUT/rocke_shape.py' \
          --batch $B \
          --seqlen $S \
          --hkv $HKV \
          --run-mode benchmark \
          --warmup $WARMUP \
          --repeat $REPEAT
    " 2>&1 | tee "$SDIR/rocke/benchmark.log"

    ROCKE_MS="$(
        grep -Eo 'avg=[0-9]+([.][0-9]+)? ms' "$SDIR/rocke/benchmark.log" \
        | tail -n 1 \
        | sed -E 's/^avg=//; s/ ms$//'
    )"

    [ -n "$ROCKE_MS" ] || {
        echo "ERROR: could not parse ROCKE latency for $TAG"
        exit 1
    }

    RATIO="$(awk -v r="$ROCKE_MS" -v a="$AITER_MS" 'BEGIN { printf "%.4f", r/a }')"
    echo "$B,$S,$HQ,$HKV,$D,$AITER_MS,$ROCKE_MS,$RATIO" >> "$OUT/summary.csv"

    printf '\nRESULT %-28s AITER=%9s ms  ROCKE=%9s ms  ROCKE/AITER=%s x\n' \
        "$TAG" "$AITER_MS" "$ROCKE_MS" "$RATIO"
done

echo
echo "============================================================"
echo "PERFORMANCE SUMMARY"
echo "============================================================"
column -s, -t "$OUT/summary.csv" 2>/dev/null || cat "$OUT/summary.csv"

if [ "$PROFILE" = "0" ]; then
    echo
    echo "PROFILE=0: skipping rocprofv3 single-dispatch profiling."
    echo "Results: $OUT/summary.csv"
    exit 0
fi

echo
echo "============================================================"
echo "SINGLE-DISPATCH PROFILING FOR ALL 8 SHAPES"
echo "============================================================"

for shape in "${SHAPES[@]}"; do
    read -r B S HKV <<< "$shape"
    TAG="b${B}_s${S}_hq${HQ}_hkv${HKV}_d${D}"
    SDIR="$OUT/$TAG"

    rm -f "$SDIR/aiter/"*kernel_trace.csv \
          "$SDIR/aiter/"*kernel_stats.csv \
          "$SDIR/rocke/"*kernel_trace.csv \
          "$SDIR/rocke/"*kernel_stats.csv 2>/dev/null || true

    echo
    echo "============================================================"
    echo "PROFILE SHAPE: B=$B S=$S Hq=$HQ Hkv=$HKV D=$D"
    echo "============================================================"

    echo "--- AITER: exactly one measured attention dispatch ---"
    bash -lc "
        source '$ENV/aiter_env.sh'
        cd '$AITER/op_tests/cpp/mha'
        '$AITER_ROCPROF' \
          --kernel-trace \
          --stats \
          --output-format csv \
          --output-directory '$SDIR/aiter' \
          --output-file aiter \
          -- \
          ./fwd.exe \
            -prec=bf16 \
            -b=$B -h=$HQ -h_k=$HKV \
            -d=$D -d_v=$D \
            -s=$S -s_k=$S \
            -iperm=0 -operm=0 \
            -mask=1 -lse=0 \
            -fwd_v3=1 \
            -v3_bf16_cvt=0 \
            -mode=0 \
            -timer=gpu \
            -kname=1 \
            -v=0 \
            -warmup=0 \
            -repeat=1
    " 2>&1 | tee "$SDIR/aiter/rocprofv3.log"

    echo
    echo "--- ROCKE: exactly one target attention dispatch ---"
    bash -lc "
        source '$ENV/rocke_env.sh'
        cd '$REPO/dnn-providers/hip-kernel-provider'
        '$ROCKE_ROCPROF' \
          --kernel-trace \
          --stats \
          --output-format csv \
          --output-directory '$SDIR/rocke' \
          --output-file rocke \
          -- \
          '$VENV/rocke/bin/python' \
            '$OUT/rocke_shape.py' \
            --batch $B \
            --seqlen $S \
            --hkv $HKV \
            --run-mode single
    " 2>&1 | tee "$SDIR/rocke/rocprofv3.log"

    echo
    echo "AITER target row:"
    grep -h \
      'fmha_fwd_hd128_bf16_causal_rtne' \
      "$SDIR/aiter/"*kernel_trace.csv \
      || {
          echo "ERROR: target AITER kernel not found for $TAG"
          exit 1
      }

    echo
    echo "ROCKE target row:"
    ROCKE_ROW="$(
        grep -h \
          'rocke_attention_dense.*bn64.*kdbvpf1' \
          "$SDIR/rocke/"*kernel_trace.csv \
          | head -n 1 \
          || true
    )"

    if [ -z "$ROCKE_ROW" ]; then
        echo "ERROR: BN64 kdbvpf1 ROCKE dispatch not found for $TAG"
        exit 1
    fi

    echo "$ROCKE_ROW"

    if echo "$ROCKE_ROW" | grep -q 'bn32'; then
        echo "ERROR: accidentally profiled BN32 for $TAG"
        exit 1
    fi
done

cat > "$OUT/README.txt" <<EOF
Eight-shape AITER vs ROCKE benchmark/profile cohort.

Common:
Hq=32
D=128
dtype=BF16
causal=true
layout=BSHD
Sq=Sk

Shapes:
B=1  S=4096   Hkv=8
B=1  S=4096   Hkv=16
B=1  S=8192   Hkv=8
B=1  S=8192   Hkv=16
B=1  S=16384  Hkv=8
B=16 S=4096   Hkv=8
B=16 S=4096   Hkv=16
B=16 S=8192   Hkv=8

Stable timing:
warmup=$WARMUP
repeat=$REPEAT
unprofiled for both AITER and ROCKE

AITER:
FMHA-v3 ASM
rounding=RTNE
-fwd_v3=1
-v3_bf16_cvt=0

ROCKE:
attention_dense prefill
block_n=64
K LDS buffers=2
V LDS buffers=1
V[j+1] prefetched into VGPRs
expected kernel tag: _kdbvpf1

Profiling:
PROFILE=$PROFILE
rocprofv3 uses exactly one target attention dispatch per implementation
for each shape. Stable 10/50 timing is kept separate from profiling.

Summary:
$OUT/summary.csv
EOF

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "Summary: $OUT/summary.csv"
echo "Per-shape logs/profiles: $OUT/<shape>/"