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
# BF16 D128 may legitimately use the non-CFVST implementation.
# Benchmark the kernel selected by this source tree rather than requiring
# an experimental CFVST-only pipeline.
cfvst_active = spec.resolved_use_cfvst()
ok, why = supports_attention_dense(spec, arch="gfx942")
if not ok:
    raise SystemExit(f"unsupported: {why}")
name = describe_dense_spec(spec)
if "_bn64_" not in name:
    raise SystemExit(f"ERROR: expected _bn64_ in kernel name, got: {name}")
# The kdbvpf1 tag belongs to a particular experimental pipeline; absence
# of that tag is valid for the stock/non-CFVST code path.
print(
    f"ROCKE shape: B={B} S={SQ} Hq={HQ} Hkv={HKV} D={D} "
    "dtype=BF16 causal=true"
)
print("ROCKE kernel:", name)
print(f"ROCKE experiment: block_n={spec.block_n}")
print(f"ROCKE experiment: CFVST active={cfvst_active}")
print("ROCKE experiment: pipeline inferred from compiled kernel; not forced by harness")
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
