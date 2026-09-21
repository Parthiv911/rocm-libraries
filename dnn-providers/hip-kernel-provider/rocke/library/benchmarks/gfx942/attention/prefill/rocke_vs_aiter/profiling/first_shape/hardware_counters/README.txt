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

| Resource          |    AITER |    ROCKE |
| ----------------- | -------: | -------: |
| Workgroup size    |      512 |      512 |
| Grid size         |  131,072 |  155,648 |
| LDS per block     | 65,536 B | 34,816 B |
| VGPRs             |       32 |       52 |
| Accumulator VGPRs |      224 |      132 |
| SGPRs             |      112 |      112 |

| Metric                    |   AITER |   ROCKE | ROCKE / AITER | Observation                                                   |
| ------------------------- | ------: | ------: | ------------: | ------------------------------------------------------------- |
| MFMA utilization          |  64.08% |  21.05% |         0.33× | AITER achieves ~3.0× higher MFMA utilization                  |
| BF16 MFMA FLOPs           | 146.03B | 146.03B |         1.00× | Same matrix-compute workload                                  |
| MFMA instructions         |   8.91M |   8.91M |         1.00× | Same MFMA instruction count                                   |
| Occupancy                 |  19.04% |  15.53% |         0.82× | AITER has moderately higher occupancy                         |
| Wave cycles               | 168.73M | 421.14M |         2.50× | ROCKE requires ~2.5× more wave cycles                         |
| CU busy cycles            |  84.21M | 212.23M |         2.52× | ROCKE keeps the CUs active much longer                        |
| LDS instructions          |   7.70M |  20.33M |         2.64× | ROCKE performs substantially more LDS operations              |
| LDS bank conflicts        |       0 |  17.83M |             — | AITER reports no LDS bank conflicts                           |
| Memory-unit stalled       |  0.0416 |  0.1262 |         3.03× | ROCKE has a substantially higher memory-stall metric          |
| VMEM latency              |  340.08 |  679.58 |         2.00× | ROCKE has roughly 2× higher VMEM latency                      |
| VALU instructions         |  50.61M |  83.91M |         1.66× | ROCKE executes considerably more non-MFMA vector instructions |
| TCC hits                  |   4.00M |   4.39M |         1.10× | Similar number of L2 hits                                     |
| TCC misses                |   1.28M |   3.76M |         2.94× | ROCKE generates almost 3× more L2 misses                      |
| Approx. TCC hit rate      |   75.7% |   53.8% |             — | AITER has substantially better cache efficiency               |
| DRAM read requests        |   1.02M |   3.50M |         3.43× | ROCKE generates ~3.4× more DRAM reads                         |
| Fetch size                | 127.43K | 437.50K |         3.43× | ROCKE transfers substantially more read-side data             |
| DRAM write requests       | 524.29K | 539.47K |         1.03× | Write traffic is approximately equal                          |
| External-memory bandwidth |  448.56 |  431.61 |         0.96× | Both kernels reach similar bandwidth                          |

