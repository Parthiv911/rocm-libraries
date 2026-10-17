# gfx942 BF16 D128 causal prefill: five-way ablation results

**LOCAL DRAFT — review software-version disclosure before publishing.**
Performance evidence below is relative-only. Internal CSV files hold absolute
GPU timings and should not be copied into public reports.

Measured scope: 12 BF16 D128 causal GQA shapes, gfx942; device=AMD Instinct MI300X VF, CUs=304; five compile-time variants of the current `attention_dense.py`.
Source revision: `8dec2cd5ce81d6a05fbecf5cc84952daf386f02a`; working tree dirty: `True`. Production dispatch baseline for all shapes: `True`.

## Benchmark identity and replay

- Harness: `/root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/bench_rocke_5way_specs.py`
- Harness SHA256: `eeca67b56c2396357784a3b61d49eb951e64203de41391e104d4611ca4736fa5`
- Kernel SHA256: `ab70315fe84d058a4ee2a497aadfa804ff845a41e40c69f1a522bd1bd4f446c2`
- Dispatch: `attention_tuning_spec(req, 'gfx942_dense').kernel_spec`
- Each non-baseline arm changes only `use_cfvst`, `lds_num_buffers`, and `v_reg_prefetch`.
- Compile: backend=`python`, ROCKE_LLVM_FLAVOR=`llvm23`; COMGR library=`/root/gpu-bench/venvs/rocke/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamd_comgr.so.3`
- Python=3.12.3, PyTorch=2.13.0+rocm10.0.0, PyTorch HIP=7.15.26333, Triton=3.8.0+git4cff872c.rocm10.0.0
- Exact Python replay (including flags and environment): `replay_command.sh`
- Complete dispatch specs, including resolved `persistent`, `exp2_fast`, block sizes, WPE, and kernel names: `resolved_specs.json`
- Full environment details and Git state: `provenance.json`

## Correctness and measurement protocol

- Independent reference: chunked FP32 causal GQA SDPA; shared Q/K/V across all arms.
- All 60 arm/shape outputs passed `torch.testing.assert_close` against FP32 reference: **True**, atol=0.01, rtol=0.01, input dtype=BF16.
- Same Python process, same current HIP stream, five independent compiled kernels.
- Warmup: 10 round-robin launches/arm; measurement: 5 randomized rounds, 50 direct KernelLauncher calls/arm/round; GPU events and synchronization.
- Reported performance: ratio of each arm's **median round latency** to the
  baseline median (not median of per-round ratios).
- Spread: per-arm minimum and maximum round latency expressed as percentage
  deviations from that arm's median. These are relative spread figures, not absolute latency.

## Five compile-time arms

| Arm | use_cfvst | lds_num_buffers (K) | v_reg_prefetch |
|---|---:|---:|---:|
| baseline | False | 1 | False |
| cfvst | True | 1 | False |
| cfvst_klds | True | 2 | False |
| cfvst_vpf | True | 1 | True |
| cfvst_klds_vpf | True | 2 | True |

## Full pipeline vs production baseline

| B | S | Hq | Hkv | D | Median speedup | Lower latency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 32 | 8 | 128 | 1.1231x | 10.96% |
| 1 | 1024 | 32 | 16 | 128 | 1.1286x | 11.40% |
| 1 | 2048 | 32 | 8 | 128 | 1.1394x | 12.23% |
| 1 | 2048 | 32 | 16 | 128 | 1.1410x | 12.36% |
| 1 | 4096 | 32 | 8 | 128 | 1.2248x | 18.35% |
| 1 | 4096 | 32 | 16 | 128 | 1.2235x | 18.27% |
| 1 | 8192 | 32 | 8 | 128 | 1.2389x | 19.28% |
| 1 | 8192 | 32 | 16 | 128 | 1.2354x | 19.05% |
| 1 | 16384 | 32 | 8 | 128 | 1.2453x | 19.70% |
| 16 | 4096 | 32 | 8 | 128 | 1.2248x | 18.35% |
| 16 | 4096 | 32 | 16 | 128 | 1.2256x | 18.40% |
| 16 | 8192 | 32 | 8 | 128 | 1.2395x | 19.32% |

## Five-way ablation by shape (relative values only)

| Shape | Arm | Median speedup vs baseline | Latency reduction vs baseline | Round min..max deviation from arm median |
|---|---|---:|---:|---:|
| b1_s1024_hq32_hkv8_d128_bf16 | baseline | 1.0000x | +0.00% | -0.56% .. +0.28% |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst | 1.0479x | +4.57% | -0.25% .. +0.20% |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst_klds | 1.0552x | +5.23% | -0.74% .. +0.15% |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst_vpf | 1.0646x | +6.06% | -0.52% .. +0.67% |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst_klds_vpf | 1.1231x | +10.96% | -0.35% .. +0.31% |
| b1_s1024_hq32_hkv16_d128_bf16 | baseline | 1.0000x | +0.00% | -0.32% .. +0.37% |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst | 1.0456x | +4.36% | -0.38% .. +1.11% |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst_klds | 1.0489x | +4.66% | -0.95% .. +2.64% |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst_vpf | 1.0708x | +6.61% | -0.15% .. +0.48% |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst_klds_vpf | 1.1286x | +11.40% | -0.23% .. +9.12% |
| b1_s2048_hq32_hkv8_d128_bf16 | baseline | 1.0000x | +0.00% | -0.16% .. +0.09% |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst | 1.0585x | +5.53% | -0.11% .. +0.30% |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst_klds | 1.0612x | +5.76% | -0.30% .. +0.19% |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst_vpf | 1.0788x | +7.31% | -0.13% .. +0.14% |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst_klds_vpf | 1.1394x | +12.23% | -0.18% .. +0.22% |
| b1_s2048_hq32_hkv16_d128_bf16 | baseline | 1.0000x | +0.00% | -0.29% .. +0.16% |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst | 1.0521x | +4.95% | -0.30% .. +0.38% |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst_klds | 1.0571x | +5.40% | -0.31% .. +3.84% |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst_vpf | 1.0785x | +7.28% | -0.10% .. +0.30% |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst_klds_vpf | 1.1410x | +12.36% | -0.10% .. +0.37% |
| b1_s4096_hq32_hkv8_d128_bf16 | baseline | 1.0000x | +0.00% | -0.06% .. +0.18% |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst | 1.1539x | +13.34% | -0.11% .. +0.13% |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst_klds | 1.1616x | +13.91% | -0.06% .. +0.13% |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst_vpf | 1.1658x | +14.22% | -0.08% .. +0.37% |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst_klds_vpf | 1.2248x | +18.35% | -0.31% .. +0.56% |
| b1_s4096_hq32_hkv16_d128_bf16 | baseline | 1.0000x | +0.00% | -0.12% .. +0.15% |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst | 1.1496x | +13.01% | -0.13% .. +0.63% |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst_klds | 1.1607x | +13.85% | -0.08% .. +0.10% |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst_vpf | 1.1587x | +13.69% | -0.01% .. +0.22% |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst_klds_vpf | 1.2235x | +18.27% | -0.06% .. +0.34% |
| b1_s8192_hq32_hkv8_d128_bf16 | baseline | 1.0000x | +0.00% | -0.11% .. +0.24% |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst | 1.1609x | +13.86% | -0.08% .. +0.11% |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst_klds | 1.1699x | +14.52% | -0.09% .. +0.43% |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst_vpf | 1.1659x | +14.23% | -0.01% .. +0.08% |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst_klds_vpf | 1.2389x | +19.28% | -0.08% .. +0.03% |
| b1_s8192_hq32_hkv16_d128_bf16 | baseline | 1.0000x | +0.00% | -0.04% .. +0.17% |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst | 1.1569x | +13.56% | -0.15% .. +0.06% |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst_klds | 1.1673x | +14.34% | -0.04% .. +0.08% |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst_vpf | 1.1634x | +14.04% | -0.05% .. +0.11% |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst_klds_vpf | 1.2354x | +19.05% | -0.11% .. +0.18% |
| b1_s16384_hq32_hkv8_d128_bf16 | baseline | 1.0000x | +0.00% | -0.03% .. +0.07% |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst | 1.1634x | +14.04% | -0.05% .. +0.04% |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst_klds | 1.1723x | +14.70% | -0.06% .. +0.03% |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst_vpf | 1.1693x | +14.48% | -0.05% .. +0.02% |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst_klds_vpf | 1.2453x | +19.70% | -0.04% .. +0.08% |
| b16_s4096_hq32_hkv8_d128_bf16 | baseline | 1.0000x | +0.00% | -0.01% .. +0.06% |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst | 1.1559x | +13.49% | -0.03% .. +0.03% |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst_klds | 1.1574x | +13.60% | -0.05% .. +0.04% |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst_vpf | 1.1576x | +13.62% | -0.03% .. +0.05% |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst_klds_vpf | 1.2248x | +18.35% | -0.07% .. +0.07% |
| b16_s4096_hq32_hkv16_d128_bf16 | baseline | 1.0000x | +0.00% | -0.02% .. +0.01% |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst | 1.1505x | +13.08% | -0.04% .. +0.09% |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst_klds | 1.1668x | +14.29% | -0.04% .. +0.04% |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst_vpf | 1.1544x | +13.37% | -0.05% .. +0.03% |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst_klds_vpf | 1.2256x | +18.40% | -0.04% .. +0.03% |
| b16_s8192_hq32_hkv8_d128_bf16 | baseline | 1.0000x | +0.00% | -0.02% .. +0.01% |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst | 1.1636x | +14.06% | -0.01% .. +0.05% |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst_klds | 1.1631x | +14.02% | -0.02% .. +0.06% |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst_vpf | 1.1670x | +14.31% | -0.01% .. +0.03% |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst_klds_vpf | 1.2395x | +19.32% | -0.05% .. +0.03% |

## Same-session incremental comparisons

Per comparison below: ratio of medians from the same session, with the
range of **paired within-round** ratios (round N vs round N).

| Shape | Compared arms (A/B) | Ratio of medians | Paired-round ratio min..max |
|---|---|---:|---:|
| b1_s1024_hq32_hkv8_d128_bf16 | baseline / cfvst | 1.0479x | 1.0420x .. 1.0522x |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst / cfvst_klds | 1.0069x | 1.0041x .. 1.0119x |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst / cfvst_vpf | 1.0159x | 1.0111x .. 1.0194x |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0643x | 1.0531x .. 1.0692x |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0549x | 1.0462x .. 1.0657x |
| b1_s1024_hq32_hkv8_d128_bf16 | baseline / cfvst_klds_vpf | 1.1231x | 1.1155x .. 1.1270x |
| b1_s1024_hq32_hkv16_d128_bf16 | baseline / cfvst | 1.0456x | 1.0307x .. 1.0505x |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst / cfvst_klds | 1.0032x | 0.9788x .. 1.0193x |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst / cfvst_vpf | 1.0242x | 1.0208x .. 1.0356x |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0760x | 1.0121x .. 1.0785x |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0540x | 0.9705x .. 1.0557x |
| b1_s1024_hq32_hkv16_d128_bf16 | baseline / cfvst_klds_vpf | 1.1286x | 1.0381x .. 1.1313x |
| b1_s2048_hq32_hkv8_d128_bf16 | baseline / cfvst | 1.0585x | 1.0553x .. 1.0592x |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst / cfvst_klds | 1.0025x | 1.0019x .. 1.0044x |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst / cfvst_vpf | 1.0192x | 1.0188x .. 1.0209x |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0737x | 1.0682x .. 1.0770x |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0561x | 1.0531x .. 1.0595x |
| b1_s2048_hq32_hkv8_d128_bf16 | baseline / cfvst_klds_vpf | 1.1394x | 1.1358x .. 1.1415x |
| b1_s2048_hq32_hkv16_d128_bf16 | baseline / cfvst | 1.0521x | 1.0477x .. 1.0553x |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst / cfvst_klds | 1.0048x | 0.9677x .. 1.0086x |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst / cfvst_vpf | 1.0251x | 1.0220x .. 1.0301x |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0793x | 1.0735x .. 1.1218x |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0579x | 1.0558x .. 1.0622x |
| b1_s2048_hq32_hkv16_d128_bf16 | baseline / cfvst_klds_vpf | 1.1410x | 1.1345x .. 1.1417x |
| b1_s4096_hq32_hkv8_d128_bf16 | baseline / cfvst | 1.1539x | 1.1523x .. 1.1561x |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst / cfvst_klds | 1.0067x | 1.0054x .. 1.0077x |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst / cfvst_vpf | 1.0103x | 1.0058x .. 1.0116x |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0544x | 1.0481x .. 1.0576x |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0505x | 1.0485x .. 1.0538x |
| b1_s4096_hq32_hkv8_d128_bf16 | baseline / cfvst_klds_vpf | 1.2248x | 1.2193x .. 1.2286x |
| b1_s4096_hq32_hkv16_d128_bf16 | baseline / cfvst | 1.1496x | 1.1433x .. 1.1516x |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst / cfvst_klds | 1.0097x | 1.0073x .. 1.0162x |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst / cfvst_vpf | 1.0079x | 1.0057x .. 1.0140x |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0541x | 1.0513x .. 1.0552x |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0560x | 1.0524x .. 1.0568x |
| b1_s4096_hq32_hkv16_d128_bf16 | baseline / cfvst_klds_vpf | 1.2235x | 1.2193x .. 1.2261x |
| b1_s8192_hq32_hkv8_d128_bf16 | baseline / cfvst | 1.1609x | 1.1604x .. 1.1624x |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst / cfvst_klds | 1.0077x | 1.0044x .. 1.0085x |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst / cfvst_vpf | 1.0042x | 1.0031x .. 1.0048x |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0590x | 1.0581x .. 1.0634x |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0627x | 1.0626x .. 1.0635x |
| b1_s8192_hq32_hkv8_d128_bf16 | baseline / cfvst_klds_vpf | 1.2389x | 1.2376x .. 1.2416x |
| b1_s8192_hq32_hkv16_d128_bf16 | baseline / cfvst | 1.1569x | 1.1563x .. 1.1588x |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst / cfvst_klds | 1.0091x | 1.0076x .. 1.0099x |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst / cfvst_vpf | 1.0056x | 1.0033x .. 1.0065x |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0583x | 1.0565x .. 1.0601x |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0619x | 1.0599x .. 1.0631x |
| b1_s8192_hq32_hkv16_d128_bf16 | baseline / cfvst_klds_vpf | 1.2354x | 1.2332x .. 1.2386x |
| b1_s16384_hq32_hkv8_d128_bf16 | baseline / cfvst | 1.1634x | 1.1632x .. 1.1641x |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst / cfvst_klds | 1.0076x | 1.0074x .. 1.0079x |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst / cfvst_vpf | 1.0051x | 1.0050x .. 1.0054x |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0623x | 1.0616x .. 1.0626x |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0650x | 1.0644x .. 1.0652x |
| b1_s16384_hq32_hkv8_d128_bf16 | baseline / cfvst_klds_vpf | 1.2453x | 1.2449x .. 1.2460x |
| b16_s4096_hq32_hkv8_d128_bf16 | baseline / cfvst | 1.1559x | 1.1559x .. 1.1566x |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst / cfvst_klds | 1.0013x | 1.0008x .. 1.0016x |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst / cfvst_vpf | 1.0015x | 1.0007x .. 1.0021x |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0582x | 1.0571x .. 1.0587x |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0580x | 1.0574x .. 1.0593x |
| b16_s4096_hq32_hkv8_d128_bf16 | baseline / cfvst_klds_vpf | 1.2248x | 1.2238x .. 1.2260x |
| b16_s4096_hq32_hkv16_d128_bf16 | baseline / cfvst | 1.1505x | 1.1493x .. 1.1510x |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst / cfvst_klds | 1.0141x | 1.0140x .. 1.0147x |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst / cfvst_vpf | 1.0034x | 1.0030x .. 1.0041x |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0504x | 1.0501x .. 1.0509x |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0617x | 1.0611x .. 1.0625x |
| b16_s4096_hq32_hkv16_d128_bf16 | baseline / cfvst_klds_vpf | 1.2256x | 1.2252x .. 1.2262x |
| b16_s8192_hq32_hkv8_d128_bf16 | baseline / cfvst | 1.1636x | 1.1631x .. 1.1637x |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst / cfvst_klds | 0.9996x | 0.9989x .. 1.0002x |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst / cfvst_vpf | 1.0030x | 1.0027x .. 1.0036x |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0657x | 1.0653x .. 1.0662x |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0621x | 1.0619x .. 1.0627x |
| b16_s8192_hq32_hkv8_d128_bf16 | baseline / cfvst_klds_vpf | 1.2395x | 1.2391x .. 1.2401x |

## Interpretation and limitations

- The comparisons above isolate CFVST, KDB, VPF and their combination.
- This measured run establishes only the stated GPU and 12 input configurations.
- Results are opt-in: the production dispatcher resolves the non-optimization
  fields, but KDB/VPF may not be reachable from tuning/dispatch by default.
- This script does **not** prove results on another CU count or compiler flavor.
- The original case-study's 15.67% claim cannot be attributed to a
  particular new shape without its original measurement record.
- Review the software stack's public-release status before posting versions.
