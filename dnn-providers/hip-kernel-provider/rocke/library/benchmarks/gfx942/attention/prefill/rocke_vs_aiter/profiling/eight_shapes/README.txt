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
CFVST and pipeline selection follow the compiled ROCKE implementation.
No _kdbvpf1 tag is required.
Profiling:
PROFILE=1
rocprofv3 uses exactly one target attention dispatch per implementation
for each shape. Stable 10/50 timing is kept separate from profiling.
Summary:
/root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/profiling/eight_shapes/summary.csv
