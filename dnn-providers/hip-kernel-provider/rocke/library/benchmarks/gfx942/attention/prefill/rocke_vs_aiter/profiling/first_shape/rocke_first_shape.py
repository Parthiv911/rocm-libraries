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
