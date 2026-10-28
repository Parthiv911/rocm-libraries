# gfx942 bf16 D128 causal prefill: CFVST, K LDS double buffering, and V register prefetch

## Introduction

This optimization enables CFVST and implements K LDS Double Buffering and V register pre-fetching on D128 bf16 persistent=true causal prefill attention gfx942 kernels.

## Problem

Benchmarking, profiling and tracing highlighted bottlenecks in:

1. V read from LDS: V was stored in traditional layout in LDS. Bank conflicts occur during access.
2. Allocation of LDS per WG (33KB). Max limit per CU is 64KB. This limits WG occupancy in CU to 1 while leaving 31KB of LDS unused.
3. Stalls at sync point for loading K and V tiles before MFMA.

The concerned optimization points were V LDS layout, register usage, LDS usage, software pipelining.

## Step-0 Lever Sweep

Lever sweep was performed to check whether performance could be improved by tuning levers to affect optimization points: **LDS usage/layout, software pipelining/scheduling, and register pressure**. Available levers are listed below along with reasons for selection decision and values swept.

#### Configuration

| Parameter | Value |
|---|---|
| GPU | (`gfx942`, 304 CUs) |
| Repository / branch | [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries), `develop` |
| Source commit | [`c8980b69a3b8c41f3be97495e7ef886cff07c892`](https://github.com/ROCm/rocm-libraries/commit/c8980b69a3b8c41f3be97495e7ef886cff07c892) |
| Data type | **BF16** |
| Head dimension (`D`) | **128** |
| Sequence length (`Sq = Skv = S`) | **1024, 2048, 4096, 8192, 16384** |
| Batch size (`B`) | **1, 16** |
| Query heads (`Hq`) | **32** |
| KV heads (`Hkv`) | **8, 16** |
| Attention | Dense, causal prefill |

Twelve `(B, S, Hkv)` workloads were swept independently. The completed sweep report records this source commit with LLVM flavor `llvm23`.

#### Lever inventory and exclusions

| Lever | Swept? | Group | Reason if excluded |
|---|---|---|---|
| `use_cfvst` | No | — | Not supported when enabled for BF16 D128 at this commit; keep the non-CFVST baseline. |
| `lds_row_pad` | **Yes** | A | — |
| `lds_k_group_pad` | No | — | Only changes the D64 K-LDS row-group layout; D128 uses `lds_row_pad`. |
| `persistent` | No | — | Selects persistent vs nonpersistent grid scheduling, not the inner K/V LDS/register pipeline; preserve production dispatch. |
| `block_m` | **Yes** | A | — |
| `block_n` | **Yes** | A | — |
| `num_persistent` | No | — | Changes persistent workgroup count and device-level utilization, not the inner software pipeline; preserve production dispatch. |
| `persist_decode` | No | — | Changes grid-level mapping of work to query blocks/heads/batches; preserve production dispatch. |
| `interleave` | No | — | Changes query-block work ordering across CTAs, not instruction interleaving within the K/V pipeline; preserve production dispatch. |
| `use_v_swizzle` | No | — | Only affects the CFVST transposed-V LDS path, which is disabled for BF16 D128. |
| `v_row_pad` | No | — | Only affects the CFVST transposed-V LDS layout; has no effective tuning impact on the selected baseline path. |
| `use_exp2_fast` | **Yes** | C | — |
| `iglp` | **Yes** | B | — |
| `iglp_mode` | **Yes** | B | — |
| `pv_sched_fence_mask` | **Yes** | B | — |
| `pv_priority` | **Yes** | B | — |
| `pv_loop_order` | **Yes** | A, B | — |
| `o_store_width` | **Yes** | D | — |
| `causal_diag_split` | **Yes** | A, C | — |
| `waves_per_eu` | **Yes** | A, B, C | — |
| `lds_num_buffers` | No | — | Only `1` is implemented/valid for the kernel at this revision; no useful sweep axis. |
| `lazy_rescale` | No | — | Inactive in the gfx942 kernel body; only changes the kernel-name token. |
| `emit_lse` | No | — | Requests an additional LSE output, rather than tuning the same output contract. |

**Count: 12 swept.** The grid knobs were **not forced off**; each workload retained its dispatch-resolved settings. CFVST, K LDS double buffering, and V register prefetch were not enabled in the baseline sweep.

#### Grouping the 12 swept knobs

| Group | Purpose | Knobs |
|---|---|---|
| **A — Geometry / LDS** | Tile size, LDS layout, occupancy-related interactions | `block_m`, `block_n`, `lds_row_pad`, `waves_per_eu`, `pv_loop_order`, `causal_diag_split` |
| **B — Instruction scheduling** | IGLP, barriers, PV ordering and priority | `iglp`, `iglp_mode`, `pv_sched_fence_mask`, `pv_priority`, `pv_loop_order`, `waves_per_eu` |
| **C — Softmax / causal computation** | Exponential policy, diagonal handling and wave interactions | `use_exp2_fast`, `causal_diag_split`, `waves_per_eu` |
| **D — Output stores** | Store vector width | `o_store_width` |

There are **12 unique knobs**, not 16: `waves_per_eu`, `pv_loop_order`, and `causal_diag_split` appear in multiple groups so their interactions can be revisited.

#### Staged sweep methodology

The sweep followed **A1 → A2 → C → B1 → B2 → D → R1 → R2**, independently for each workload.

| Stage | Procedure |
|---|---|
| **A1 — Geometry screening** | Sweep `block_m={64,128,256}`, `block_n={32,64}`, `lds_row_pad={0,4,8,16}`, `waves_per_eu={1,2,4}` (72 combinations before legality/deduplication). Shortlist promising complete configurations. |
| **A2 — Geometry interactions** | On A1 finalists, retune `waves_per_eu={1,2,3,4}`, `pv_loop_order={d_major,k_major}`, `causal_diag_split={False,True}`. |
| **C — Computation** | On A2 finalists, tune `use_exp2_fast={Auto,False,True}`, `causal_diag_split`, and `waves_per_eu`. |
| **B1 — Scheduling** | On C finalists, test legal `iglp`/`iglp_mode`/`pv_sched_fence_mask` combinations together with `pv_loop_order` and `pv_priority={0,1,2,3}`. IGLP and explicit scheduling fences are mutually exclusive. |
| **B2 — Scheduling interactions** | On B1 finalists, retune `waves_per_eu` and `pv_loop_order`. |
| **D — Output** | On B2 finalists, test `o_store_width={1,2,4}`. |
| **R1, R2 — Cross-group refinement** | Two bounded rounds of single-knob neighborhood changes around the strongest **full configurations**, allowing adjustments to earlier group choices. |

At every stage, the sweep rejects unsupported/duplicate effective specs, numerically checks legal candidates against FP32 causal SDPA (`atol=rtol=1e-2`), screens GPU latency, then remeasures promising candidates against the production baseline. **Several top full configurations** are carried forward, rather than choosing one winner per independent knob. This is a **bounded staged search**, not an exhaustive Cartesian-product sweep.

#### Baseline vs winning kernels

These are the best **final A/B-tested** configurations, compared with each workload's production-dispatch baseline. Each finalist passed a BF16-output-versus-FP32 causal SDPA numerical check (`atol=rtol=1e-2`) before timing. The timing statistic is the median of five alternating paired baseline/candidate rounds, with 50 timed launches per round.

| B | S | Hkv | Speedup | Latency reduction | Paired min–max | Baseline kernel | Winning kernel |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 1024 | 8 | **1.66346×** | **39.88%** | 1.64369–1.66779× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_dsplit` |
| 1 | 1024 | 16 | **1.65172×** | **39.46%** | 1.63562–1.65811× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_dsplit` |
| 1 | 2048 | 8 | **1.20801×** | **17.22%** | 1.20781–1.21174× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_pvkmaj_dsplit` |
| 1 | 2048 | 16 | **1.20041×** | **16.70%** | 1.19735–1.20419× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_pvkmaj_dsplit` |
| 1 | 4096 | 8 | **1.14929×** | **12.99%** | 1.13996–1.15458× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit` |
| 1 | 4096 | 16 | **1.14466×** | **12.64%** | 1.14171–1.14902× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2_krowpad4_iglp1_iglpm1_pvkmaj_prio2_dsplit` |
| 1 | 8192 | 8 | **1.14749×** | **12.85%** | 1.14341–1.15020× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2_krowpad4_iglp1_iglpm1_pvkmaj_prio2_dsplit` |
| 1 | 8192 | 16 | **1.14439×** | **12.62%** | 1.14271–1.14813× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio3_dsplit` |
| 1 | 16384 | 8 | **1.13920×** | **12.22%** | 1.13823–1.14506× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit` |
| 16 | 4096 | 8 | **1.09626×** | **8.78%** | 1.09536–1.09692× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe1_krowpad4_iglp1_iglpm1_pvkmaj_prio3_dsplit` |
| 16 | 4096 | 16 | **1.11022×** | **9.93%** | 1.10837–1.11122× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe1_krowpad4_iglp1_iglpm1_pvkmaj_prio2_dsplit` |
| 16 | 8192 | 8 | **1.10214×** | **9.27%** | 1.09696–1.10866× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit` |

The lever sweep improved performance by around 40% for S=1024 and 17% for S=2048. For B=1 S=4096+ its around 12%. While batch=16 around 9%.

We explore whether introducing optimizations:

1. That targets bank conflicts in V LDS loads (CFVST) (in contrast to K LDS loads) could also improve performance.
2. That directly targets the 3rd bottleneck would give greater performance gains. (such as K Double Buffering + V Register Prefetching)

## Optimization 1: Enable CFVST

V was stored in traditional layout in LDS. This caused bank conflicts during access. The hypothesis was that transposed layout would reduce bank conflicts. This feature was already present (CFVST) but disabled for bf16 due to register spillage concerns. Performance improved substantially with it enabled, so we adopted this optimization. However, we do not claim that it reduced bank conflicts as it was not measured.

### Register Usage and Static Occupancy (§R.4)

Compiled gfx942 BF16 D128 causal prefill (BM256/BN64, persistent-304, 512 threads/WG) with LLVM20 and LLVM23. The saved HSACO metadata was analyzed with `probe_regs.py` and `probe_occupancy.py`.

| Configuration | LLVM20 VGPR | LLVM23 VGPR | VGPR/SGPR spills | Scratch |
|---|---:|---:|---:|---:|
| Baseline | 193 | 194 | 0 / 0 | 0 B/thread |
| CFVST | 228 | 225 | 0 / 0 | 0 B/thread |
| CFVST + KDB | 210 | 230 | 0 / 0 | 0 B/thread |
| CFVST + VPF | 218 | 230 | 0 / 0 | 0 B/thread |
| **CFVST + KDB + VPF** | **216** | **230** | **0 / 0** | **0 B/thread** |
| Full + pad4 | 218 | 230 | 0 / 0 | 0 B/thread |
| Full + pad4 + diagonal split | 218 | 232 | 0 / 0 | 0 B/thread |
| Full + LSE | 218 | 232 | 0 / 0 | 0 B/thread |

All 16 compiled variants were spill-free. `probe_occupancy.py` estimates **8 waves/CU, 1 WG/CU**, with LDS reported as the limiting resource. Base line kernel here is `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2`. The source commit is same as below benchmark.

## Optimization 2 and 3: K LDS Double Buffering + V register pre-fetching

Analysis of stalls using threadtrace found a large number of waiting at sync points. One such point was waiting for K and V tiles to be loaded before MFMA. This wait restricted MFMA and memory loading parallelism. The hypothesis was that K LDS Double Buffering + V register pre-fetching would introduce parallelism and latency hiding. Optimizing 1 would leave other the bottleneck. The optimization improved performance substantially hence it was adopted.

#### Results

## Performance vs. Step-0 Lever-Sweep Winners

All results are for **gfx942, BF16 D128 causal attention, persistent-304**, with `Hq=32`. The **baseline** for every comparison is the shape-specific **Step-0 lever-sweep winner (`A`)**, not the production-dispatch baseline.

Measurements use LLVM23, 10 warmup launches, 5 randomized rounds, and 50 timed launches per round. All compared kernels passed BF16-versus-FP32 correctness checks (`atol=rtol=1e-2`).

### Ablation: Each Configuration vs. Step-0 Winner

| dtype | B | S | Hq | Hkv | D | CFVST | CFVST + KDB | CFVST + VPF | CFVST + KDB + VPF | **CFVST + KDB + VPF + LDS pad4 + causal split** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 1 | 4096 | 32 | 8 | 128 | 2.46% | 4.37% | 7.54% | 8.62% | **12.28%** |
| bf16 | 1 | 4096 | 32 | 16 | 128 | 2.21% | 4.39% | 7.65% | 8.78% | **12.39%** |
| bf16 | 1 | 8192 | 32 | 8 | 128 | 2.71% | 4.94% | 8.27% | 9.54% | **14.12%** |
| bf16 | 1 | 8192 | 32 | 16 | 128 | 1.45% | 3.56% | 6.90% | 8.70% | **13.02%** |
| bf16 | 1 | 16384 | 32 | 8 | 128 | 3.53% | 5.44% | 9.58% | 10.58% | **15.72%** |
| bf16 | 16 | 4096 | 32 | 8 | 128 | 0.80% | 2.00% | 5.54% | 6.90% | **10.40%** |
| bf16 | 16 | 4096 | 32 | 16 | 128 | 1.07% | 3.38% | 6.32% | 7.80% | **10.13%** |
| bf16 | 16 | 8192 | 32 | 8 | 128 | 0.34% | 2.20% | 5.90% | 7.54% | **11.52%** |

### Final Configuration vs. Step-0 Winner (Paired Range)

**Final configuration:** CFVST + K LDS double buffering (KDB) + V register prefetch (VPF) + `lds_row_pad=4` + `causal_diag_split=True`.

| dtype | B | S | Hq | Hkv | D | **Latency reduction** | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 1 | 4096 | 32 | 8 | 128 | **12.28%** | 11.68% | 13.53% |
| bf16 | 1 | 4096 | 32 | 16 | 128 | **12.39%** | 11.67% | 12.68% |
| bf16 | 1 | 8192 | 32 | 8 | 128 | **14.12%** | 13.63% | 14.32% |
| bf16 | 1 | 8192 | 32 | 16 | 128 | **13.02%** | 12.78% | 13.21% |
| bf16 | 1 | 16384 | 32 | 8 | 128 | **15.72%** | 15.53% | 16.00% |
| bf16 | 16 | 4096 | 32 | 8 | 128 | **10.40%** | 10.29% | 10.62% |
| bf16 | 16 | 4096 | 32 | 16 | 128 | **10.13%** | 9.80% | 10.20% |
| bf16 | 16 | 8192 | 32 | 8 | 128 | **11.52%** | 11.38% | 11.90% |

**Min/max** are the extrema of same-round paired speedups converted to latency-reduction percentages. Central values use the ratio of the five-round median latencies.

**Scope:** These are direct, same-checkout comparisons against the Step-0 winners. Candidate kernels are derived from production dispatch with the named optimizations enabled; they do **not** inherit the Step-0 winner's other tuning knobs.

## Results and Ablation Methodology

The benchmark compares each shape's **Step-0 existing-lever sweep winner (`A`)** against seven independently compiled configurations derived from the **current production dispatch**. It uses the eight supported persistent-304 BF16 D128 causal attention shapes (not the earlier twelve-shape cohort).

For each of the eight shapes:

1. Start one **torch-free** rocKE/NumPy Python process with one ROCm stack, the Python compilation backend, and LLVM23. Record the GPU, actual HIP runtime/COMGR libraries, Git HEAD, working-tree status, and source hashes.
2. Create one dedicated HIP stream using `rocke.runtime.hip_module` and use that stream for every kernel launch and timing event.
3. Build the shape's `AttentionRequest` and resolve the production specification using `attention_tuning_spec(req, "gfx942_dense").kernel_spec`. Require persistent dispatch (`persistent=True`, `num_persistent=304`); reject unsupported shapes.
4. Construct the separate **`A` comparator** by applying the recorded shape-specific Step-0 winning *existing* levers. Keep `use_cfvst=False`, `lds_num_buffers=1`, and `v_reg_prefetch=False`. The previous winner is rebuilt using the **current checkout**, not timed using historical measurements.
5. Construct seven **B-stage** specifications from the *current production-dispatch specification*, **not** from `A`. Preserve all non-ablation production knobs except the explicitly stated LDS-padding and causal-split overrides:

   | Arm | Variant | `use_cfvst` | `lds_num_buffers` (K) | `v_reg_prefetch` | `lds_row_pad` | `causal_diag_split` |
   |---:|---|---:|---:|---:|---|---|
   | A | `A` (Step-0 winner) | `False` | 1 | `False` | Shape-specific winner | Shape-specific winner |
   | B1 | `baseline` | `False` | 1 | `False` | Production dispatch | Production dispatch |
   | B2 | `cfvst` | `True` | 1 | `False` | Production dispatch | Production dispatch |
   | B3 | `cfvst_kdb` | `True` | 2 | `False` | Production dispatch | Production dispatch |
   | B4 | `cfvst_vpf` | `True` | 1 | `True` | Production dispatch | Production dispatch |
   | B5 | `pipeline` | `True` | 2 | `True` | Production dispatch | Production dispatch |
   | B6 | `pipeline_pad` | `True` | 2 | `True` | 4 | Production dispatch |
   | B7 | `pipeline_pad_dsplit` | `True` | 2 | `True` | 4 | `True` |

6. Assert that the A configuration disables the new pipeline and that production dispatch starts with the three new knobs disabled. Verify every successive B-stage change modifies exactly its advertised field(s), including resolved knob values.
7. Check `supports_attention_dense` for all eight specifications. Require eight distinct compile-time kernel names; retain the original and rebuilt Step-0 winner names for auditing.
8. Run the Python kernel emitter and compile **eight independent HSACOs** per shape (one for `A` and seven for B1–B7); record their kernel names, code-object hashes, and specifications. No runtime GPU-kernel variant-selection branch is used.
9. Generate one shared BF16-rounded Q/K/V input set. Allocate a separate output buffer and construct a direct `KernelLauncher` plus `LaunchConfig` for each compiled kernel.
10. Compute one independent **chunked NumPy FP32 causal GQA SDPA** reference from the shared BF16-rounded inputs. No PyTorch import or `torch.testing.assert_close` is used.
11. **Before timing**, fill each output with `0xFF` bytes (NaN poison), invoke its kernel, synchronize, and compare **every output element** against the FP32 reference using `numpy.testing.assert_allclose(atol=1e-2, rtol=1e-2, equal_nan=False)`. Record maximum absolute and relative errors; abort on failure.
12. Perform **10 round-robin warmup launches per variant**, then synchronize the HIP stream.
13. Execute **five measurement rounds**, randomly shuffling the order of all eight variants within each round.
14. For each variant in each round, record a HIP start event, make **50 direct `KernelLauncher` calls**, record a HIP end event, synchronize that event, and divide elapsed GPU-event time by 50 to obtain per-launch latency.
15. After each timed batch, validate its final output against the same FP32 reference with the same tolerances, **outside the GPU-event timing interval**. Record errors and abort on failure. Validation checks the last output of each 50-launch batch, not all 50 intermediate outputs.
16. Save the shape, variant, measurement round, randomized execution order, configuration knobs, latency, and correctness result. Preserve the resolved specs, kernel names, HSACO hashes, and environment provenance.

**Correctness coverage:** Each shape has eight initial checks and `8 × 5 = 40` post-batch checks, or **48 checks per shape**. Across eight shapes there are **384 checks** (64 initial and 320 post-batch).

17. Collect five per-round latency samples per variant and compute each variant's independent median:

    `T_variant = median(t_variant,1, ..., t_variant,5)`

18. Use the shape-specific **Step-0 winner (`A`) as the reference denominator** for the main tables. For each candidate B2, B3, B4, B5, or B7, compute its speedup as:

    `Speedup_A_to_candidate = T_A / T_candidate`

19. Compute the main percentage latency reduction **against `A`**, not against production dispatch:

    `Reduction_A_to_candidate (%) = 100 × (1 - T_candidate / T_A)`

20. For the five-column ablation table, report B2 (`CFVST`), B3 (`CFVST+KDB`), B4 (`CFVST+VPF`), B5 (`CFVST+KDB+VPF`), and B7 (`CFVST+KDB+VPF+LDS pad4+causal split`), **each compared directly with `A`**.
21. For the final full-optimization table, compare only **B7 versus `A`**. Calculate the percentage reduction for each matched round:

    `R_r = 100 × (1 - t_B7,r / t_A,r)`

22. Report the observed same-round range:

    `Min = min(R_1, R_2, R_3, R_4, R_5)`  
    `Max = max(R_1, R_2, R_3, R_4, R_5)`

23. Report the central B7-vs-A reduction from the **ratio of the two medians**, not the median of the five round-level reductions.
24. Independently calculate *within-B-stage* comparisons to isolate pipeline effects: `baseline → cfvst`, `cfvst → cfvst_kdb`, `cfvst → cfvst_vpf`, `cfvst_kdb → pipeline`, `cfvst_vpf → pipeline`, `pipeline → pipeline_pad`, and `pipeline_pad → pipeline_pad_dsplit`.
25. Export `ablation_runs.csv`, `ablation_summary.csv`, `ablation_ratios.csv`, `ablation_comparison.csv`, `ablation_comparison.md`, `A_vs_B7.csv`, `validation.csv`, `resolved_specs.json`, and `provenance.json` for reproduction and review.

**Interpretation:** `A` and B1–B7 are measured within the same session and compiled from the same current source, so direct **Step-0-winner-versus-optimized-production** comparisons are valid. However, the B stages **do not inherit `A`'s geometry or scheduling knobs**: the results do **not** quantify the incremental benefit of adding CFVST/KDB/VPF on top of the Step-0 winner itself. The eight persistent shapes exclude the four nonpersistent B=1, S=1024/2048 cases.

#### Benchmark Environment and Provenance

| Component | Configuration |
| :--- | :--- |
| Hardware target | gfx942, 304 CUs |
| Benchmarked source HEAD | `74fec8a4c863d0a6362f647e0fc203feecfb1451` |
| Kernel source SHA-256 | `fcade13e9212e3e6c7c50018996c7c19bfe92eaa973808ed97800c75aa057d80` |
| Benchmark harness SHA-256 | `343a65ea3e06eaaa37c4841920047449a55d67f10387b28bf359fa8ebe1c87ea` |
| ROCm stack | 10.0.0 |
| HIP | `7.15.26333` |
| Python | `3.12.3` |
| Triton | `3.8.0+git4cff872c.rocm10.0.0` |
| ROCKE compiler backend | Python |
| LLVM flavor | `llvm23` |

#### Shape-by-shape default kernel identity built by `attention_tuning_spec(req, "gfx942_dense").kernel_spec`

| dtype | B | S | Hq | Hkv | D | Exact baseline kernel name | persistent | exp2_fast (resolved) | decode (resolved) | grid | block |
|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| bf16 | 1 | 1024 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `false` | `false` | `qb_major` | `[4, 32, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 1024 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `false` | `false` | `qb_major` | `[4, 32, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 2048 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `false` | `false` | `qb_major` | `[8, 32, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 2048 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `false` | `false` | `qb_major` | `[8, 32, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 4096 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `true` | `true` | `qb_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 4096 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `true` | `true` | `qb_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 8192 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `true` | `true` | `qb_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 8192 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `true` | `true` | `qb_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 16384 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `true` | `true` | `qb_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 16 | 4096 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2` | `true` | `true` | `hkv_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 16 | 4096 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `true` | `true` | `qb_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 16 | 8192 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2` | `true` | `true` | `hkv_major` | `[304, 1, 1]` | `[512, 1, 1]` |

**Benchmark harness:** `library/benchmarks/gfx942/attention/prefill/bench_rocke_7way_specs.py`

**Kernel source:** `library/kernels/gfx942/attention_dense.py`

#### Reproducibility

Clone repo to gpu-bench/.

**`setup.sh`** — install the ROCm Python environment and rocKE:

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$(git rev-parse --show-toplevel)"
ROOT="${GPU_BENCH_ROOT:-/root/gpu-bench}"
ROCKE="$REPO/dnn-providers/hip-kernel-provider/rocke"
VENV="$ROOT/venvs"
ENV="$ROOT/env"

AMD_INDEX=https://stable.repo.amd.com/rocm/whl-next/

mkdir -p "$VENV" "$ENV"

# ----------------------------------------------------------------------
# Host dependencies
# ----------------------------------------------------------------------

apt-get update
apt-get install -y \
  git python3.12 python3.12-venv python3.12-dev python3-pip \
  cmake ninja-build build-essential pkg-config libnuma-dev

# ----------------------------------------------------------------------
# ROCKE virtual environment
# ----------------------------------------------------------------------

ROCKE_VENV="$VENV/rocke"

if [ ! -x "$ROCKE_VENV/bin/python" ] || \
   ! "$ROCKE_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  rm -rf "$ROCKE_VENV"
  python3.12 -m venv "$ROCKE_VENV"
fi

"$ROCKE_VENV/bin/python" -m pip install -U \
  pip setuptools wheel packaging ninja psutil

"$ROCKE_VENV/bin/python" -m pip install \
  --index-url "$AMD_INDEX" \
  'rocm[devel,device-gfx942,libraries]==10.0.0' \
  'torch[device-gfx942]==2.13.0+rocm10.0.0'

"$ROCKE_VENV/bin/rocm-sdk" init >/dev/null

# ----------------------------------------------------------------------
# Install ROCKE
# ----------------------------------------------------------------------

"$ROCKE_VENV/bin/python" -m pip install -e "$ROCKE/platform"
"$ROCKE_VENV/bin/python" -m pip install -e "$ROCKE/library" --no-deps

# ----------------------------------------------------------------------
# ROCKE environment activation script
# ----------------------------------------------------------------------

cat > "$ENV/rocke_env.sh" <<EOF
source "$ROCKE_VENV/bin/activate"

rocm-sdk init >/dev/null

export ROCM_HOME="\$(rocm-sdk path --root)"
export ROCM_PATH="\$ROCM_HOME"
export HIP_PATH="\$ROCM_HOME"
export PATH="\$(rocm-sdk path --bin):\$PATH"

export HIP_PLATFORM=amd
export ROCKE_BACKEND=python
export ROCKE_LLVM_FLAVOR=llvm23

LIB="\$(python -c 'import importlib.util, pathlib; s=importlib.util.find_spec("_rocm_sdk_core"); print(pathlib.Path(list(s.submodule_search_locations)[0]) / "lib")')"

export ROCKE_COMGR_LIB="\$LIB/libamd_comgr.so.3"
export ROCKE_HIP_LIB="\$LIB/libamdhip64.so.7"
export LD_LIBRARY_PATH="\$LIB:\${LD_LIBRARY_PATH:-}"

unset LIB
EOF

bash -n "$ENV/rocke_env.sh"

echo
echo "ROCKE setup complete."
echo "ROCKE env: source $ENV/rocke_env.sh"
```

Run the setup script from its directory:

```bash
chmod +x setup.sh
./setup.sh
```

Run the five-way benchmark:

```bash
cd /root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke

export ROCKE_LLVM_FLAVOR=llvm23
export PYTHONPATH="$PWD/platform/python:$PWD/library:$PWD"

/root/gpu-bench/venvs/rocke/bin/python \
  library/benchmarks/gfx942/attention/prefill/bench_rocke_7way_specs.py \
  --warmup 10 \
  --rounds 5 \
  --repeat 50 \
  --atol 1e-2 \
  --rtol 1e-2 \
  --out-dir /root/gpu-bench/rocke_7way_specs_results

```

Results are written to `/root/gpu-bench/rocke_7way_specs_results`
when using the `--out-dir` argument shown above. When omitted,
the benchmark uses the default directory configured in
`bench_rocke_7way_specs.py`.

### Global-Memory Coalescing and ISA Verification (§R.5)

Verified on **gfx942**, BF16 D128 causal attention, BM256/BN64, persistent-304, using LLVM20 and LLVM23.

| Global access | Modeled positions | Coalesced | Cache lines | Footprint | ISA |
|---|---:|---|---:|---:|---|
| K async → LDS | 64 | Yes | 2 | 1.0× | 32-bit |
| V CFVST x0 | 32 | Yes | 2 | 1.0× | 32-bit |
| V CFVST x1 | 32 | Yes | 2 | 1.0× | 32-bit |

`vector_transactions`, `addr_fn_from_strides`, `analyze_coalescing`, and `CoalescingReport.summary()` confirmed fully coalesced accesses using `line_bytes_for("gfx942") = 128`. **All `assert_asm_backed` checks passed** across six compiled HSACOs.

ISA inspection confirmed separate V x0/x1 loads and their **2048-byte token-stride difference** under both LLVM20 and LLVM23.

**Result:** No modeled global-memory coalescing regression in the verified configurations. Evidence covers the tested K/V access paths, not every tuning configuration.


## Runbook and Architecture Compliance

The CFVST, K LDS double-buffering (KDB), and V register-prefetching (VPF) optimizations follow the CK DSL Optimization Runbook (`platform/dsl_docs/optimization/optimization_runbook.md`) and the gfx942-specific architecture constraints (`platform/dsl_docs/optimization/arch/gfx942.md`, §21).

### Per-Lever Mapping

| Lever | Implementation | Runbook sections | §13.3 Attention checklist | gfx942 architecture reference |
|---|---|---|---|---|
| `use_cfvst=True` | `perm_b32` store-path transpose into transposed V LDS, with optional XOR swizzling | §6.3 LDS, §6.4 bank conflicts, §6.5 registers, §7.2 operand layouts | Tile Q / K / V; shared memory capacity; register pressure from accumulators | §21.2 LDS geometry and banks; §21.3 `v_perm_b32` cross-lane primitive; §21.4 VGPR constraints; §21.8 transpose-read limitations |
| `lds_num_buffers=2` | Double-buffer K in LDS to overlap next-tile K loading with current-tile computation; V LDS remains single-buffered | §6.3 LDS, §8.1 software pipelining, §8.2 async copy, §8.3 waits and barriers | Tile Q / K / V; shared memory capacity; register pressure from accumulators | §21.2 64 KiB LDS capacity; §21.4 occupancy constraints; §21.6 async LDS load width; §21.8 ISA constraints |
| `v_reg_prefetch=True` | Prefetch next-tile V into registers to overlap memory loading with current-tile computation | §6.5 registers, §8.1 software pipelining, §8.2 async copy, §8.3 waits and barriers | Tile Q / K / V; register pressure from accumulators | §21.4 VGPR pressure and occupancy; §21.6 global buffer-load widths; §21.8 ISA constraints |

### Architecture and Configuration Gating

The optimizations are implemented in `library/kernels/gfx942/attention_dense.py`, using `Gfx942AttentionDenseSpec`.

Architecture eligibility is enforced by the existing `arch != "gfx942"` checks in `supports_attention_dense(spec, arch=...)` and `build_attention_dense(spec, arch=...)`. The support predicate returns `(False, reason)` for other architectures, while the builder raises `NotImplementedError`. Dispatch also restricts the gfx942 dense candidate to gfx942 through its architecture-specific registration and spec factory.

The optimization-specific eligibility conditions are enforced by `supports_attention_dense()`:

- **CFVST (`use_cfvst=True`):** For BF16, requires D128 causal attention without sliding window or attention sinks. Causal diagonal splitting is supported. Existing FP16 CFVST behavior is preserved.
- **KDB (`lds_num_buffers=2`):** Requires BF16 D128 causal attention, CFVST enabled, `block_n=64`, no sliding window, and no attention sinks.
- **VPF (`v_reg_prefetch=True`):** Has the same eligibility conditions as KDB but is independently controlled. Enabling VPF does not require enabling KDB.

The combined K/V LDS allocation is checked against the gfx942 capacity using `attention_lds_capacity_bytes(arch)` and `_lds_bytes(spec)`. Unsupported configurations are rejected before kernel emission.

For BF16, the default configuration retains `use_cfvst=False`, `lds_num_buffers=1`, and `v_reg_prefetch=False`. All three optimizations remain opt-in and do not change the default kernel behavior.
