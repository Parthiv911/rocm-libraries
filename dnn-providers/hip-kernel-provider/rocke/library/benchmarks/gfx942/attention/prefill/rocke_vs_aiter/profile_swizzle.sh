#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/gpu-bench
REPO="$ROOT/rocm-libraries"
AITER="$ROOT/aiter"
ENV="$ROOT/env"
VENV="$ROOT/venvs"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/profiling/first_shape"
AITER_ROCPROF="$VENV/aiter/bin/rocprofv3"
ROCKE_ROCPROF="$VENV/rocke/bin/rocprofv3"
AVAIL="$VENV/rocke/bin/rocprofv3-avail"
WARMUP=10
REPEAT=50
[ -x "$AITER_ROCPROF" ] || { echo "missing $AITER_ROCPROF"; exit 1; }
[ -x "$ROCKE_ROCPROF" ] || { echo "missing $ROCKE_ROCPROF"; exit 1; }
[ -x "$VENV/rocke/bin/python" ] || { echo "missing $VENV/rocke/bin/python"; exit 1; }
[ -x "$AITER/op_tests/cpp/mha/fwd.exe" ] || { echo "missing AITER fwd.exe"; exit 1; }
[ -f "$ENV/aiter_env.sh" ] || { echo "missing $ENV/aiter_env.sh"; exit 1; }
[ -f "$ENV/rocke_env.sh" ] || { echo "missing $ENV/rocke_env.sh"; exit 1; }
mkdir -p "$OUT/aiter" "$OUT/rocke"
rm -f \
    "$OUT/aiter"/*.csv \
    "$OUT/aiter"/*.log \
    "$OUT/rocke"/*.csv \
    "$OUT/rocke"/*.log
if [ -x "$AVAIL" ]; then
    "$AVAIL" list --pmc > "$OUT/available_counters.txt"
fi
# ============================================================
# ROCKE helper
#
# --benchmark:
#   compile/load once, then 10 warmups + 50 timed launches.
#
# --single:
#   compile/load once, then exactly one target attention launch.
#   This is what rocprofv3 runs, so the trace stays clean.
# ============================================================
cat > "$OUT/rocke_first_shape.py" <<'PY'
#!/usr/bin/env python3
import argparse
import math
from dataclasses import replace
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
B = 1
SQ = 4096
SK = 4096
HQ = 32
HKV = 8
D = 128
cli = argparse.ArgumentParser()
cli.add_argument(
    "--run-mode",
    choices=("benchmark", "single"),
    required=True,
)
cli.add_argument("--warmup", type=int, default=10)
cli.add_argument("--repeat", type=int, default=50)
opt = cli.parse_args()
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
spec = replace(
    spec,
    lds_row_pad=0,
    use_k_swizzle=True,
)
if spec.lds_row_pad != 0:
    raise SystemExit(
        f"ERROR: expected lds_row_pad=0, got lds_row_pad={spec.lds_row_pad}"
    )
if not getattr(spec, "use_k_swizzle", False):
    raise SystemExit("ERROR: expected use_k_swizzle=True")
if spec.block_n != 64:
    raise SystemExit(
        f"ERROR: expected dispatch block_n=64, got block_n={spec.block_n}"
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
    raise SystemExit(
        f"ERROR: expected _bn64_ in kernel name, got: {name}"
    )
for tag in ("_krowpad0", "_kswz1", "_kdbvpf1"):
    if tag not in name:
        raise SystemExit(
            f"ERROR: expected {tag} in kernel name, got: {name}"
        )
print("ROCKE kernel:", name)
print(f"ROCKE experiment: block_n={spec.block_n}")
print(f"ROCKE experiment: K LDS row pad={spec.lds_row_pad}")
print(f"ROCKE experiment: K XOR swizzle={spec.use_k_swizzle}")
print("ROCKE experiment: K LDS buffers=2")
print("ROCKE experiment: V LDS buffers=1")
print("ROCKE experiment: V[j+1] prefetch=VGPR")
print("ROCKE experiment: buffering policy is implemented in attention_dense.py")
torch.manual_seed(0)
dt = torch.bfloat16
q = (
    torch.randn(B, SQ, HQ, D, dtype=dt, device="cuda") * 0.2
).contiguous()
k = (
    torch.randn(B, SK, HKV, D, dtype=dt, device="cuda") * 0.2
).contiguous()
v = (
    torch.randn(B, SK, HKV, D, dtype=dt, device="cuda") * 0.2
).contiguous()
out = torch.zeros(
    B,
    SQ,
    HQ,
    D,
    dtype=dt,
    device="cuda",
)
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
    # Exactly one target ROCKE attention dispatch for rocprofv3.
    call()
    torch.cuda.synchronize()
PY
# ============================================================
# STABLE UNPROFILED PERFORMANCE
# 10 warmups + 50 repeats for both implementations.
# ============================================================
echo "============================================================"
echo "AITER performance: ${WARMUP} warmup + ${REPEAT} repeat"
echo "NOT profiled"
echo "============================================================"
bash -lc "
source '$ENV/aiter_env.sh'
cd '$AITER/op_tests/cpp/mha'
./fwd.exe \
  -prec=bf16 \
  -b=1 -h=32 -h_k=8 \
  -d=128 -d_v=128 \
  -s=4096 -s_k=4096 \
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
"
echo
echo "============================================================"
echo "ROCKE performance: ${WARMUP} warmup + ${REPEAT} repeat"
echo "NOT profiled"
echo "============================================================"
bash -lc "
source '$ENV/rocke_env.sh'
cd '$REPO/dnn-providers/hip-kernel-provider'
'$VENV/rocke/bin/python' \
  '$OUT/rocke_first_shape.py' \
  --run-mode benchmark \
  --warmup $WARMUP \
  --repeat $REPEAT
"
# ============================================================
# SINGLE-DISPATCH PROFILING
#
# Keep the profiler run at one target dispatch. This avoids
# contaminating the trace/stats with 60 attention dispatches.
# The stable performance numbers were already collected above.
# ============================================================
echo
echo "============================================================"
echo "AITER rocprofv3: exactly one measured ASM dispatch"
echo "============================================================"
bash -lc "
source '$ENV/aiter_env.sh'
cd '$AITER/op_tests/cpp/mha'
'$AITER_ROCPROF' \
  --kernel-trace \
  --stats \
  --output-format csv \
  --output-directory '$OUT/aiter' \
  --output-file aiter_first_shape \
  -- \
  ./fwd.exe \
    -prec=bf16 \
    -b=1 -h=32 -h_k=8 \
    -d=128 -d_v=128 \
    -s=4096 -s_k=4096 \
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
" 2>&1 | tee "$OUT/aiter/rocprofv3.log"
echo
echo "============================================================"
echo "ROCKE rocprofv3: exactly one target attention dispatch"
echo "============================================================"
bash -lc "
source '$ENV/rocke_env.sh'
cd '$REPO/dnn-providers/hip-kernel-provider'
'$ROCKE_ROCPROF' \
  --kernel-trace \
  --stats \
  --output-format csv \
  --output-directory '$OUT/rocke' \
  --output-file rocke_first_shape \
  -- \
  '$VENV/rocke/bin/python' \
    '$OUT/rocke_first_shape.py' \
    --run-mode single
" 2>&1 | tee "$OUT/rocke/rocprofv3.log"
# ============================================================
# README
# ============================================================
cat > "$OUT/README.txt" <<EOF2
Shape:
B=1
S=4096
Hq=32
Hkv=8
D=128
dtype=BF16
causal=true
layout=BSHD
Stable performance timing:
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
Experimental pipeline:
K LDS buffers=2
V LDS buffers=1
V[j+1] prefetched into VGPRs
The K double-buffer/V-prefetch policy is implemented inside
kernels/gfx942/attention_dense.py.
There is intentionally no n_buffers spec override.
Expected experimental kernel suffix:
_krowpad0_kswz1_kdbvpf1
Expected BN64 LDS footprint:
49152 bytes
Profiling:
The stable 10/50 benchmark is separate from rocprofv3.
rocprofv3 runs exactly ONE target attention dispatch for AITER
and exactly ONE target attention dispatch for ROCKE.
rocprofv3=$ROCKE_ROCPROF
Collection:
rocprofv3 --kernel-trace --stats --output-format csv
EOF2
# ============================================================
# Show target rows
# ============================================================
echo
echo "============================================================"
echo "TARGET KERNEL ROWS"
echo "============================================================"
echo
echo "AITER:"
grep -h \
  'fmha_fwd_hd128_bf16_causal_rtne' \
  "$OUT/aiter"/*kernel_trace.csv \
  || true
echo
echo "ROCKE:"
grep -h \
  'rocke_attention_dense.*bn64.*krowpad0.*kswz1.*kdbvpf1' \
  "$OUT/rocke"/*kernel_trace.csv \
  || true
# ============================================================
# ROCKE sanity check
# ============================================================
echo
echo "============================================================"
echo "ROCKE SANITY CHECK"
echo "============================================================"
ROCKE_ROW="$(
    grep -h \
      'rocke_attention_dense.*bn64.*krowpad0.*kswz1.*kdbvpf1' \
      "$OUT/rocke"/*kernel_trace.csv \
      | head -n 1 \
      || true
)"
if [ -z "$ROCKE_ROW" ]; then
    echo "ERROR: did not find BN64 krowpad0 kswz1 kdbvpf1 ROCKE dispatch in trace."
    exit 1
fi
echo "Found expected ROCKE dispatch:"
echo "$ROCKE_ROW"
if echo "$ROCKE_ROW" | grep -q 'bn32'; then
    echo "ERROR: accidentally profiled BN32."
    exit 1
fi
if ! echo "$ROCKE_ROW" | grep -q ',49152,0,'; then
    echo "ERROR: expected LDS/group segment 49152 bytes and private segment 0."
    echo "Actual row:"
    echo "$ROCKE_ROW"
    exit 1
fi
echo
echo "Expected resource signature for modified BN64 kernel:"
echo "  LDS/group segment:       49152 bytes"
echo "  private segment:         0 bytes"
echo "  total VGPR allocation:   256"
# ============================================================
# Output files
# ============================================================
echo
echo "============================================================"
echo "FILES"
echo "============================================================"
find "$OUT" -maxdepth 2 -type f | sort