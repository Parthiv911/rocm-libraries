# gfx942 BF16 D128 causal prefill: five-way ablation results

**LOCAL DRAFT — review software-version disclosure before publishing.**
Performance evidence below is relative-only. Internal CSV files hold absolute
GPU timings and should not be copied into public reports.

Measured scope: 12 BF16 D128 causal GQA shapes, gfx942; device=AMD Instinct MI300X VF, CUs=304; five compile-time variants of the current `attention_dense.py`.
Source revision: `51832d8c8078267a8e5274478c3289f3096106ff`; working tree dirty: `True`. Production dispatch baseline for all shapes: `True`.

## Benchmark identity and replay

- Harness: `/root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/bench_rocke_5way_specs.py`
- Harness SHA256: `eae32fdcb1be87dbac24873a0104d01accc64873f22442096fb4ea36addb9cdf`
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
- All 360 validation checks (pre-timing and after each timed batch) passed `torch.testing.assert_close` against FP32 reference: **True**, atol=0.01, rtol=0.01, input dtype=BF16.
- Post-batch correctness is checked outside GPU-event timing; each check verifies the last of the batch's repeated launches, not every launch.
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
| 1 | 1024 | 32 | 8 | 128 | 1.1203x | 10.74% |
| 1 | 1024 | 32 | 16 | 128 | 1.1362x | 11.99% |
| 1 | 2048 | 32 | 8 | 128 | 1.1258x | 11.18% |
| 1 | 2048 | 32 | 16 | 128 | 1.1227x | 10.93% |
| 1 | 4096 | 32 | 8 | 128 | 1.1692x | 14.47% |
| 1 | 4096 | 32 | 16 | 128 | 1.1796x | 15.23% |
| 1 | 8192 | 32 | 8 | 128 | 1.1845x | 15.58% |
| 1 | 8192 | 32 | 16 | 128 | 1.1836x | 15.51% |
| 1 | 16384 | 32 | 8 | 128 | 1.1834x | 15.50% |
| 16 | 4096 | 32 | 8 | 128 | 1.1591x | 13.72% |
| 16 | 4096 | 32 | 16 | 128 | 1.1279x | 11.34% |
| 16 | 8192 | 32 | 8 | 128 | 1.1598x | 13.78% |

## Five-way ablation by shape (relative values only)

| Shape | Arm | Median speedup vs baseline | Latency reduction vs baseline | Round min..max deviation from arm median |
|---|---|---:|---:|---:|
| b1_s1024_hq32_hkv8_d128_bf16 | baseline | 1.0000x | +0.00% | -0.69% .. +0.57% |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst | 1.0370x | +3.57% | -0.70% .. +1.25% |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst_klds | 1.0353x | +3.41% | -0.55% .. +0.70% |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst_vpf | 1.0595x | +5.62% | -1.11% .. +3.31% |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst_klds_vpf | 1.1203x | +10.74% | -0.55% .. +0.50% |
| b1_s1024_hq32_hkv16_d128_bf16 | baseline | 1.0000x | +0.00% | -1.24% .. +0.43% |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst | 1.0477x | +4.55% | -0.51% .. +0.86% |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst_klds | 1.0489x | +4.66% | -1.23% .. +0.33% |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst_vpf | 1.0822x | +7.60% | -0.19% .. +0.26% |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst_klds_vpf | 1.1362x | +11.99% | -0.23% .. +1.54% |
| b1_s2048_hq32_hkv8_d128_bf16 | baseline | 1.0000x | +0.00% | -0.30% .. +1.41% |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst | 1.0529x | +5.02% | -0.09% .. +0.56% |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst_klds | 1.0535x | +5.08% | -0.56% .. +0.21% |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst_vpf | 1.0733x | +6.83% | -0.13% .. +0.83% |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst_klds_vpf | 1.1258x | +11.18% | -0.15% .. +417.89% |
| b1_s2048_hq32_hkv16_d128_bf16 | baseline | 1.0000x | +0.00% | -0.35% .. +0.44% |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst | 1.0468x | +4.47% | -0.15% .. +0.36% |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst_klds | 1.0550x | +5.21% | -0.29% .. +0.57% |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst_vpf | 1.0735x | +6.85% | -0.23% .. +0.77% |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst_klds_vpf | 1.1227x | +10.93% | -0.29% .. +0.97% |
| b1_s4096_hq32_hkv8_d128_bf16 | baseline | 1.0000x | +0.00% | -0.15% .. +0.24% |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst | 1.1094x | +9.86% | -0.18% .. +0.60% |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst_klds | 1.1285x | +11.39% | -0.19% .. +0.58% |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst_vpf | 1.1232x | +10.97% | -0.58% .. +0.73% |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst_klds_vpf | 1.1692x | +14.47% | -0.88% .. +0.37% |
| b1_s4096_hq32_hkv16_d128_bf16 | baseline | 1.0000x | +0.00% | -0.55% .. +0.49% |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst | 1.1113x | +10.02% | -0.78% .. +0.30% |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst_klds | 1.1348x | +11.88% | -0.94% .. +0.36% |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst_vpf | 1.1345x | +11.85% | -0.67% .. +0.54% |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst_klds_vpf | 1.1796x | +15.23% | -0.08% .. +0.42% |
| b1_s8192_hq32_hkv8_d128_bf16 | baseline | 1.0000x | +0.00% | -0.35% .. +0.36% |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst | 1.1117x | +10.05% | -0.46% .. +0.42% |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst_klds | 1.1375x | +12.09% | -1.23% .. +0.66% |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst_vpf | 1.1298x | +11.49% | -0.73% .. +1.10% |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst_klds_vpf | 1.1845x | +15.58% | -0.28% .. +0.17% |
| b1_s8192_hq32_hkv16_d128_bf16 | baseline | 1.0000x | +0.00% | -0.49% .. +0.33% |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst | 1.1209x | +10.78% | -0.84% .. +0.85% |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst_klds | 1.1425x | +12.47% | -0.15% .. +0.22% |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst_vpf | 1.1323x | +11.69% | -0.54% .. +1.24% |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst_klds_vpf | 1.1836x | +15.51% | -0.70% .. +0.28% |
| b1_s16384_hq32_hkv8_d128_bf16 | baseline | 1.0000x | +0.00% | -1.27% .. +0.08% |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst | 1.1256x | +11.16% | -0.53% .. +0.39% |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst_klds | 1.1424x | +12.46% | -0.06% .. +0.16% |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst_vpf | 1.1412x | +12.37% | -0.13% .. +0.15% |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst_klds_vpf | 1.1834x | +15.50% | -0.23% .. +0.19% |
| b16_s4096_hq32_hkv8_d128_bf16 | baseline | 1.0000x | +0.00% | -0.31% .. +0.20% |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst | 1.1122x | +10.09% | -0.18% .. +0.14% |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst_klds | 1.1207x | +10.77% | -0.09% .. +0.40% |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst_vpf | 1.1125x | +10.12% | -0.22% .. +0.25% |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst_klds_vpf | 1.1591x | +13.72% | -0.72% .. +0.67% |
| b16_s4096_hq32_hkv16_d128_bf16 | baseline | 1.0000x | +0.00% | -0.02% .. +0.24% |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst | 1.0995x | +9.05% | -0.68% .. +0.29% |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst_klds | 1.1091x | +9.84% | -0.52% .. +0.74% |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst_vpf | 1.0969x | +8.83% | -0.68% .. +0.01% |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst_klds_vpf | 1.1279x | +11.34% | -0.19% .. +0.75% |
| b16_s8192_hq32_hkv8_d128_bf16 | baseline | 1.0000x | +0.00% | -0.31% .. +0.68% |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst | 1.1133x | +10.18% | -0.73% .. +1.61% |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst_klds | 1.1193x | +10.66% | -0.28% .. +1.54% |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst_vpf | 1.1133x | +10.18% | -0.49% .. +0.56% |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst_klds_vpf | 1.1598x | +13.78% | -0.59% .. +1.88% |

## Same-session incremental comparisons

Per comparison below: ratio of medians from the same session, with the
range of **paired within-round** ratios (round N vs round N).

| Shape | Compared arms (A/B) | Ratio of medians | Paired-round ratio min..max |
|---|---|---:|---:|
| b1_s1024_hq32_hkv8_d128_bf16 | baseline / cfvst | 1.0370x | 1.0301x .. 1.0484x |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst / cfvst_klds | 0.9984x | 0.9913x .. 1.0165x |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst / cfvst_vpf | 1.0217x | 0.9890x .. 1.0363x |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0821x | 1.0717x .. 1.0897x |
| b1_s1024_hq32_hkv8_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0574x | 1.0404x .. 1.0930x |
| b1_s1024_hq32_hkv8_d128_bf16 | baseline / cfvst_klds_vpf | 1.1203x | 1.1187x .. 1.1221x |
| b1_s1024_hq32_hkv16_d128_bf16 | baseline / cfvst | 1.0477x | 1.0274x .. 1.0576x |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst / cfvst_klds | 1.0012x | 0.9979x .. 1.0129x |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst / cfvst_vpf | 1.0330x | 1.0251x .. 1.0424x |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0832x | 1.0537x .. 1.0892x |
| b1_s1024_hq32_hkv16_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0499x | 1.0367x .. 1.0535x |
| b1_s1024_hq32_hkv16_d128_bf16 | baseline / cfvst_klds_vpf | 1.1362x | 1.1238x .. 1.1371x |
| b1_s2048_hq32_hkv8_d128_bf16 | baseline / cfvst | 1.0529x | 1.0439x .. 1.0677x |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst / cfvst_klds | 1.0006x | 0.9997x .. 1.0073x |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst / cfvst_vpf | 1.0194x | 1.0167x .. 1.0202x |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0686x | 0.2061x .. 1.0708x |
| b1_s2048_hq32_hkv8_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0490x | 0.2042x .. 1.0497x |
| b1_s2048_hq32_hkv8_d128_bf16 | baseline / cfvst_klds_vpf | 1.1258x | 0.2167x .. 1.1413x |
| b1_s2048_hq32_hkv16_d128_bf16 | baseline / cfvst | 1.0468x | 1.0394x .. 1.0530x |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst / cfvst_klds | 1.0078x | 1.0015x .. 1.0143x |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst / cfvst_vpf | 1.0255x | 1.0177x .. 1.0307x |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0642x | 1.0509x .. 1.0718x |
| b1_s2048_hq32_hkv16_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0459x | 1.0342x .. 1.0539x |
| b1_s2048_hq32_hkv16_d128_bf16 | baseline / cfvst_klds_vpf | 1.1227x | 1.1080x .. 1.1269x |
| b1_s4096_hq32_hkv8_d128_bf16 | baseline / cfvst | 1.1094x | 1.1039x .. 1.1114x |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst / cfvst_klds | 1.0173x | 1.0152x .. 1.0232x |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst / cfvst_vpf | 1.0125x | 1.0106x .. 1.0243x |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0360x | 1.0303x .. 1.0455x |
| b1_s4096_hq32_hkv8_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0409x | 1.0349x .. 1.0520x |
| b1_s4096_hq32_hkv8_d128_bf16 | baseline / cfvst_klds_vpf | 1.1692x | 1.1649x .. 1.1781x |
| b1_s4096_hq32_hkv16_d128_bf16 | baseline / cfvst | 1.1113x | 1.1052x .. 1.1256x |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst / cfvst_klds | 1.0211x | 1.0131x .. 1.0307x |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst / cfvst_vpf | 1.0208x | 1.0097x .. 1.0308x |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0395x | 1.0263x .. 1.0441x |
| b1_s4096_hq32_hkv16_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0398x | 1.0337x .. 1.0436x |
| b1_s4096_hq32_hkv16_d128_bf16 | baseline / cfvst_klds_vpf | 1.1796x | 1.1692x .. 1.1861x |
| b1_s8192_hq32_hkv8_d128_bf16 | baseline / cfvst | 1.1117x | 1.1032x .. 1.1177x |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst / cfvst_klds | 1.0232x | 1.0165x .. 1.0346x |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst / cfvst_vpf | 1.0163x | 1.0006x .. 1.0261x |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0413x | 1.0286x .. 1.0512x |
| b1_s8192_hq32_hkv8_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0484x | 1.0408x .. 1.0581x |
| b1_s8192_hq32_hkv8_d128_bf16 | baseline / cfvst_klds_vpf | 1.1845x | 1.1783x .. 1.1912x |
| b1_s8192_hq32_hkv16_d128_bf16 | baseline / cfvst | 1.1209x | 1.1077x .. 1.1323x |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst / cfvst_klds | 1.0193x | 1.0085x .. 1.0295x |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst / cfvst_vpf | 1.0102x | 1.0033x .. 1.0157x |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0360x | 1.0331x .. 1.0417x |
| b1_s8192_hq32_hkv16_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0453x | 1.0404x .. 1.0622x |
| b1_s8192_hq32_hkv16_d128_bf16 | baseline / cfvst_klds_vpf | 1.1836x | 1.1745x .. 1.1879x |
| b1_s16384_hq32_hkv8_d128_bf16 | baseline / cfvst | 1.1256x | 1.1171x .. 1.1276x |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst / cfvst_klds | 1.0149x | 1.0089x .. 1.0188x |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst / cfvst_vpf | 1.0139x | 1.0087x .. 1.0176x |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0359x | 1.0340x .. 1.0390x |
| b1_s16384_hq32_hkv8_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0370x | 1.0353x .. 1.0393x |
| b1_s16384_hq32_hkv8_d128_bf16 | baseline / cfvst_klds_vpf | 1.1834x | 1.1710x .. 1.1846x |
| b16_s4096_hq32_hkv8_d128_bf16 | baseline / cfvst | 1.1122x | 1.1093x .. 1.1144x |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst / cfvst_klds | 1.0077x | 1.0037x .. 1.0091x |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst / cfvst_vpf | 1.0003x | 0.9979x .. 1.0026x |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0342x | 1.0313x .. 1.0408x |
| b16_s4096_hq32_hkv8_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0418x | 1.0373x .. 1.0470x |
| b16_s4096_hq32_hkv8_d128_bf16 | baseline / cfvst_klds_vpf | 1.1591x | 1.1536x .. 1.1638x |
| b16_s4096_hq32_hkv16_d128_bf16 | baseline / cfvst | 1.0995x | 1.0990x .. 1.1071x |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst / cfvst_klds | 1.0087x | 1.0013x .. 1.0129x |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst / cfvst_vpf | 0.9976x | 0.9907x .. 1.0073x |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0170x | 1.0065x .. 1.0245x |
| b16_s4096_hq32_hkv16_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0283x | 1.0207x .. 1.0303x |
| b16_s4096_hq32_hkv16_d128_bf16 | baseline / cfvst_klds_vpf | 1.1279x | 1.1219x .. 1.1320x |
| b16_s8192_hq32_hkv8_d128_bf16 | baseline / cfvst | 1.1133x | 1.1031x .. 1.1180x |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst / cfvst_klds | 1.0054x | 1.0009x .. 1.0061x |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst / cfvst_vpf | 1.0000x | 0.9946x .. 1.0120x |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst_klds / cfvst_klds_vpf | 1.0362x | 1.0328x .. 1.0394x |
| b16_s8192_hq32_hkv8_d128_bf16 | cfvst_vpf / cfvst_klds_vpf | 1.0418x | 1.0267x .. 1.0439x |
| b16_s8192_hq32_hkv8_d128_bf16 | baseline / cfvst_klds_vpf | 1.1598x | 1.1462x .. 1.1631x |

## Interpretation and limitations

- The comparisons above isolate CFVST, KDB, VPF and their combination.
- This measured run establishes only the stated GPU and 12 input configurations.
- Results are opt-in: the production dispatcher resolves the non-optimization
  fields, but KDB/VPF may not be reachable from tuning/dispatch by default.
- This script does **not** prove results on another CU count or compiler flavor.
- The original case-study's 15.67% claim cannot be attributed to a
  particular new shape without its original measurement record.
- Review the software stack's public-release status before posting versions.
