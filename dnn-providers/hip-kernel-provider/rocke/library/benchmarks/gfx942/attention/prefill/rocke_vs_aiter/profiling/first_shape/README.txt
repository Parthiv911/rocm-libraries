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
rocprofv3=/root/gpu-bench/venvs/aiter/bin/rocprofv3

ROCKE:
attention_dense prefill
profiling helper bypasses run()
profiling helper calls KernelLauncher exactly once
rocprofv3=/root/gpu-bench/venvs/rocke/bin/rocprofv3

Collection:
rocprofv3 --kernel-trace --stats --output-format csv
