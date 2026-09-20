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

[ -x "$AITER_ROCPROF" ] || { echo "missing $AITER_ROCPROF"; exit 1; }
[ -x "$ROCKE_ROCPROF" ] || { echo "missing $ROCKE_ROCPROF"; exit 1; }
[ -x "$AITER/op_tests/cpp/mha/fwd.exe" ] || { echo "missing AITER fwd.exe"; exit 1; }
[ -f "$ENV/aiter_env.sh" ] || { echo "missing $ENV/aiter_env.sh"; exit 1; }
[ -f "$ENV/rocke_env.sh" ] || { echo "missing $ENV/rocke_env.sh"; exit 1; }

mkdir -p "$OUT/aiter" "$OUT/rocke"
rm -f "$OUT/aiter"/*.csv "$OUT/aiter"/*.log \
      "$OUT/rocke"/*.csv "$OUT/rocke"/*.log

# Save counters available on this MI300X.
if [ -x "$AVAIL" ]; then
    "$AVAIL" list --pmc > "$OUT/available_counters.txt"
fi

# ---------------------------------------------------------------------------
# ROCKE helper:
# exact first benchmark shape, and EXACTLY ONE target attention dispatch.
# ---------------------------------------------------------------------------
cat > "$OUT/rocke_first_shape.py" <<'PY'
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

B = 1
SQ = 4096
SK = 4096
HQ = 32
HKV = 8
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

spec = resolve_dense_spec(req, {})

ok, why = supports_attention_dense(spec, arch="gfx942")
if not ok:
    raise SystemExit(f"unsupported: {why}")

print("ROCKE kernel:", describe_dense_spec(spec))

torch.manual_seed(0)
dt = torch.bfloat16

q = (torch.randn(B, SQ, HQ, D, dtype=dt, device="cuda") * 0.2).contiguous()
k = (torch.randn(B, SK, HKV, D, dtype=dt, device="cuda") * 0.2).contiguous()
v = (torch.randn(B, SK, HKV, D, dtype=dt, device="cuda") * 0.2).contiguous()
out = torch.zeros(B, SQ, HQ, D, dtype=dt, device="cuda")
scale = 1.0 / math.sqrt(D)

launcher = _make_launcher(spec)
stream = torch.cuda.current_stream().cuda_stream
cfg = _launch_config(spec, stream)

vals = {
    "q_ptr": q,
    "k_ptr": k,
    "v_ptr": v,
    "o_ptr": out,
    "scale": scale,
}

# Exactly one ROCKE attention-kernel launch in this process.
launcher(vals, config=cfg)
torch.cuda.synchronize()
PY

# ---------------------------------------------------------------------------
# AITER
# ---------------------------------------------------------------------------
echo "============================================================"
echo "AITER warmup (not profiled)"
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
  -v3_bf16_cvt=2 \
  -mode=0 \
  -timer=gpu \
  -kname=1 \
  -v=0 \
  -warmup=1 \
  -repeat=1
"

echo "============================================================"
echo "AITER rocprofv3: one measured ASM dispatch"
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
    -v3_bf16_cvt=2 \
    -mode=0 \
    -timer=gpu \
    -kname=1 \
    -v=0 \
    -warmup=0 \
    -repeat=1
" 2>&1 | tee "$OUT/aiter/rocprofv3.log"

# ---------------------------------------------------------------------------
# ROCKE
# ---------------------------------------------------------------------------
echo "============================================================"
echo "ROCKE warmup (not profiled)"
echo "============================================================"

bash -lc "
source '$ENV/rocke_env.sh'
cd '$REPO/dnn-providers/hip-kernel-provider'
'$VENV/rocke/bin/python' '$OUT/rocke_first_shape.py'
"

echo "============================================================"
echo "ROCKE rocprofv3: exactly one attention dispatch"
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
  '$VENV/rocke/bin/python' '$OUT/rocke_first_shape.py'
" 2>&1 | tee "$OUT/rocke/rocprofv3.log"

# ---------------------------------------------------------------------------
# Metadata + quick verification
# ---------------------------------------------------------------------------
cat > "$OUT/README.txt" <<EOF
Shape:
B=1
S=4096
Hq=32
Hkv=8
D=128
dtype=BF16
causal=true
layout=BSHD

AITER:
FMHA-v3 ASM
-fwd_v3=1
-v3_bf16_cvt=2
profiled warmup=0 repeat=1
rocprofv3=$AITER_ROCPROF

ROCKE:
attention_dense prefill
profiling helper bypasses run()
profiling helper calls KernelLauncher exactly once
rocprofv3=$ROCKE_ROCPROF

Collection:
rocprofv3 --kernel-trace --stats --output-format csv
EOF

echo
echo "============================================================"
echo "TARGET KERNEL ROWS"
echo "============================================================"

echo
echo "AITER:"
grep -h 'fmha_fwd_hd128_bf16_causal_rtz' \
    "$OUT/aiter"/*kernel_trace.csv || true

echo
echo "ROCKE:"
grep -h 'rocke_attention_dense' \
    "$OUT/rocke"/*kernel_trace.csv || true

echo
echo "============================================================"
echo "FILES"
echo "============================================================"
find "$OUT" -maxdepth 2 -type f | sort
