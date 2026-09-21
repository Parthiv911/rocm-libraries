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
