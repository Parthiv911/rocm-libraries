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
warmup=10
repeat=50
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
_kdbvpf1
Expected BN64 LDS footprint:
51200 bytes
Profiling:
The stable 10/50 benchmark is separate from rocprofv3.
rocprofv3 runs exactly ONE target attention dispatch for AITER
and exactly ONE target attention dispatch for ROCKE.
rocprofv3=/root/gpu-bench/venvs/rocke/bin/rocprofv3
Collection:
rocprofv3 --kernel-trace --stats --output-format csv
