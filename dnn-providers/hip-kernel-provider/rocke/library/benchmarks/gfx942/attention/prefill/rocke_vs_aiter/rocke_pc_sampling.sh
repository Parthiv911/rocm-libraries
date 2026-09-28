#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/gpu-bench
REPO="$ROOT/rocm-libraries"
ENV="$ROOT/env"
VENV="$ROOT/venvs"

ROCPROF_VENV="$VENV/rocprof"
ROCPROF_COMPUTE="$VENV/rocke/bin/rocprof-compute"
ROCPROFV3="$ROCPROF_VENV/bin/rocprofv3"
ROCPROFV3_AVAIL="$VENV/rocke/bin/rocprofv3-avail"

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_ROOT="$HERE/profiling"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$PROFILE_ROOT/rocke_pc_sampling_$STAMP"
HELPER="$PROFILE_ROOT/rocke_pc_target.py"

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-1000}"

# rocprof-compute's documented minimum stochastic interval is 65536 cycles.
# Keep it a power of two.
PC_INTERVAL="${PC_INTERVAL:-65536}"

KERNEL_FILTER="rocke_attention_dense_d128_hq32_kv8_bn64"

[ -x "$ROCPROF_COMPUTE" ] || {
    echo "ERROR: missing $ROCPROF_COMPUTE"
    exit 1
}
[ -x "$VENV/rocke/bin/python" ] || {
    echo "ERROR: missing $VENV/rocke/bin/python"
    exit 1
}
[ -f "$ENV/rocke_env.sh" ] || {
    echo "ERROR: missing $ENV/rocke_env.sh"
    exit 1
}

mkdir -p "$PROFILE_ROOT"

cat > "$HELPER" <<'PY'
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


B = 1
SQ = 4096
SK = 4096
HQ = 32
HKV = 8
D = 128


cli = argparse.ArgumentParser()
cli.add_argument("--warmup", type=int, default=10)
cli.add_argument("--repeat", type=int, default=1000)
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

spec = resolve_dense_spec(req, {})

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
        "ERROR: CFVST path is not active; expected modified BN64 pipeline"
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

print("ROCKE kernel:", name)
print("PC sampling target:")
print("  B=1 SQ=4096 SK=4096 HQ=32 HKV=8 D=128 BF16 causal")
print("  block_n=64")
print("  K LDS buffers=2")
print("  V LDS buffers=1")
print("  V[j+1] prefetch=VGPR")
print(f"  warmup={opt.warmup}")
print(f"  sampled launches={opt.repeat}")


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


# Prime the launcher, then warm up before the repeated measurement batch.
# rocprof-compute is active for the whole process, so these early target
# dispatches can also contribute samples; with 1000 repeated launches their
# contribution is negligible. The kernel-name filter excludes other kernels.
call()
torch.cuda.synchronize()

for _ in range(opt.warmup):
    call()
torch.cuda.synchronize()

for _ in range(opt.repeat):
    call()
torch.cuda.synchronize()

print("ROCKE PC-sampling target launches complete.")
PY

chmod +x "$HELPER"

echo "============================================================"
echo "PC SAMPLING SUPPORT (non-fatal preflight)"
echo "============================================================"

if [ -x "$ROCPROFV3_AVAIL" ]; then
    if ! "$ROCPROFV3_AVAIL" info --pc-sampling \
        > "$PROFILE_ROOT/pc_sampling_support.txt" 2>&1; then
        echo "WARNING: rocprofv3-avail PC-sampling query failed."
        echo "This does NOT stop the profiling run."
        echo "Details saved to: $PROFILE_ROOT/pc_sampling_support.txt"
    else
        cat "$PROFILE_ROOT/pc_sampling_support.txt"
    fi
elif [ -x "$ROCPROFV3" ]; then
    if ! "$ROCPROFV3" -L \
        > "$PROFILE_ROOT/pc_sampling_support.txt" 2>&1; then
        echo "WARNING: rocprofv3 capability query failed; continuing."
    else
        cat "$PROFILE_ROOT/pc_sampling_support.txt"
    fi
else
    echo "WARNING: no rocprofv3-avail/rocprofv3 found for capability listing."
    echo "Continuing directly to rocprof-compute PC sampling."
fi

echo
echo "============================================================"
echo "ROCKE STOCHASTIC PC SAMPLING"
echo "============================================================"
echo "Output:        $OUT"
echo "Kernel filter: $KERNEL_FILTER"
echo "Interval:      $PC_INTERVAL cycles"
echo "Warmup:        $WARMUP"
echo "Launches:      $REPEAT"
echo

"$ROCPROF_COMPUTE" profile \
    --output-directory "$OUT" \
    --no-roof \
    --experimental \
    --pc-sampling \
    --pc-sampling-method stochastic \
    --pc-sampling-interval "$PC_INTERVAL" \
    -k "$KERNEL_FILTER" \
    -- \
    bash -lc "
        source '$ENV/rocke_env.sh'
        cd '$REPO/dnn-providers/hip-kernel-provider'
        exec '$VENV/rocke/bin/python' \
            '$HELPER' \
            --warmup '$WARMUP' \
            --repeat '$REPEAT'
    "

echo
echo "============================================================"
echo "PC SAMPLING ANALYSIS — SORTED BY SAMPLE COUNT"
echo "============================================================"

"$ROCPROF_COMPUTE" analyze \
    -p "$OUT" \
    -k 0 \
    --pc-sampling-sorting-type count \
    | tee "$OUT/pc_sampling_count.txt"

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "Workload: $OUT"
echo "Analysis: $OUT/pc_sampling_count.txt"
echo
echo "To inspect by ISA offset instead:"
echo "  $ROCPROF_COMPUTE analyze -p '$OUT' -k 0 --pc-sampling-sorting-type offset"