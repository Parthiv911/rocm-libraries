

## Baseline vs winning configurations

<!-- BEGIN MEASURED CONFIGURATIONS -->

| B | S | Hkv | Speedup | Paired min–max | Baseline kernel | Winning kernel |
|---:|---:|---:|---:|---|---|---|
| 1 | 1024 | 8 | **1.66346×** | 1.64369–1.66779× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_dsplit` |
| 1 | 1024 | 16 | **1.65172×** | 1.63562–1.65811× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_dsplit` |
| 1 | 2048 | 8 | **1.20801×** | 1.20781–1.21174× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_pvkmaj_dsplit` |
| 1 | 2048 | 16 | **1.20041×** | 1.19735–1.20419× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_pvkmaj_dsplit` |
| 1 | 4096 | 8 | **1.14929×** | 1.13996–1.15458× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit` |
| 1 | 4096 | 16 | **1.14466×** | 1.14171–1.14902× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2_krowpad4_iglp1_iglpm1_pvkmaj_prio2_dsplit` |
| 1 | 8192 | 8 | **1.14749×** | 1.14341–1.15020× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2_krowpad4_iglp1_iglpm1_pvkmaj_prio2_dsplit` |
| 1 | 8192 | 16 | **1.14439×** | 1.14271–1.14813× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio3_dsplit` |
| 1 | 16384 | 8 | **1.13920×** | 1.13823–1.14506× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit` |
| 16 | 4096 | 8 | **1.09626×** | 1.09536–1.09692× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe1_krowpad4_iglp1_iglpm1_pvkmaj_prio3_dsplit` |
| 16 | 4096 | 16 | **1.11022×** | 1.10837–1.11122× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe1_krowpad4_iglp1_iglpm1_pvkmaj_prio2_dsplit` |
| 16 | 8192 | 8 | **1.10214×** | 1.09696–1.10866× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit` |

### B=1, S=1024, Hq=32, Hkv=8, D=128

**Baseline kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2`

**Baseline knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": false,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 8,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": false,
  "pv_loop_order": "d_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 2
}
```

**Winning kernel:** `rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_dsplit`

**Winning knobs:**

```json
{
  "block_m": 128,
  "block_n": 32,
  "causal_diag_split": true,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 4,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": false,
  "pv_loop_order": "d_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": true,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 4
}
```

### B=1, S=1024, Hq=32, Hkv=16, D=128

**Baseline kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2`

**Baseline knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": false,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 8,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": false,
  "pv_loop_order": "d_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 2
}
```

**Winning kernel:** `rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_dsplit`

**Winning knobs:**

```json
{
  "block_m": 128,
  "block_n": 32,
  "causal_diag_split": true,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 4,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": false,
  "pv_loop_order": "d_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": true,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 4
}
```

### B=1, S=2048, Hq=32, Hkv=8, D=128

**Baseline kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2`

**Baseline knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": false,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 8,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": false,
  "pv_loop_order": "d_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 2
}
```

**Winning kernel:** `rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_pvkmaj_dsplit`

**Winning knobs:**

```json
{
  "block_m": 128,
  "block_n": 32,
  "causal_diag_split": true,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 4,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": false,
  "pv_loop_order": "k_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": true,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 4
}
```

### B=1, S=2048, Hq=32, Hkv=16, D=128

**Baseline kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2`

**Baseline knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": false,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 8,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": false,
  "pv_loop_order": "d_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 2
}
```

**Winning kernel:** `rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_pvkmaj_dsplit`

**Winning knobs:**

```json
{
  "block_m": 128,
  "block_n": 32,
  "causal_diag_split": true,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 4,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": false,
  "pv_loop_order": "k_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": true,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 4
}
```

### B=1, S=4096, Hq=32, Hkv=8, D=128

**Baseline kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2`

**Baseline knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": false,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 8,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "d_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 2
}
```

**Winning kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit`

**Winning knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": true,
  "emit_lse": false,
  "iglp": true,
  "iglp_mode": 1,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 4,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "k_major",
  "pv_priority": 1,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 3
}
```

### B=1, S=4096, Hq=32, Hkv=16, D=128

**Baseline kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2`

**Baseline knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": false,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 8,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "d_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 2
}
```

**Winning kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2_krowpad4_iglp1_iglpm1_pvkmaj_prio2_dsplit`

**Winning knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": true,
  "emit_lse": false,
  "iglp": true,
  "iglp_mode": 1,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 4,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "k_major",
  "pv_priority": 2,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 2
}
```

### B=1, S=8192, Hq=32, Hkv=8, D=128

**Baseline kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2`

**Baseline knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": false,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 8,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "d_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 2
}
```

**Winning kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2_krowpad4_iglp1_iglpm1_pvkmaj_prio2_dsplit`

**Winning knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": true,
  "emit_lse": false,
  "iglp": true,
  "iglp_mode": 1,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 4,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "k_major",
  "pv_priority": 2,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 2
}
```

### B=1, S=8192, Hq=32, Hkv=16, D=128

**Baseline kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2`

**Baseline knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": false,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 8,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "d_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 2
}
```

**Winning kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio3_dsplit`

**Winning knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": true,
  "emit_lse": false,
  "iglp": true,
  "iglp_mode": 1,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 4,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "k_major",
  "pv_priority": 3,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 3
}
```

### B=1, S=16384, Hq=32, Hkv=8, D=128

**Baseline kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2`

**Baseline knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": false,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 8,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "d_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 2
}
```

**Winning kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit`

**Winning knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": true,
  "emit_lse": false,
  "iglp": true,
  "iglp_mode": 1,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 4,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "k_major",
  "pv_priority": 1,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 2
}
```

### B=16, S=4096, Hq=32, Hkv=8, D=128

**Baseline kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2`

**Baseline knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": false,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 8,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "d_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 2
}
```

**Winning kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe1_krowpad4_iglp1_iglpm1_pvkmaj_prio3_dsplit`

**Winning knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": true,
  "emit_lse": false,
  "iglp": true,
  "iglp_mode": 1,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 4,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "k_major",
  "pv_priority": 3,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 1
}
```

### B=16, S=4096, Hq=32, Hkv=16, D=128

**Baseline kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2`

**Baseline knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": false,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 8,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "d_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 2
}
```

**Winning kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe1_krowpad4_iglp1_iglpm1_pvkmaj_prio2_dsplit`

**Winning knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": true,
  "emit_lse": false,
  "iglp": true,
  "iglp_mode": 1,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 4,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "k_major",
  "pv_priority": 2,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 1
}
```

### B=16, S=8192, Hq=32, Hkv=8, D=128

**Baseline kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2`

**Baseline knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": false,
  "emit_lse": false,
  "iglp": false,
  "iglp_mode": 0,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 8,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "d_major",
  "pv_priority": 0,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 2
}
```

**Winning kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit`

**Winning knobs:**

```json
{
  "block_m": 256,
  "block_n": 64,
  "causal_diag_split": true,
  "emit_lse": false,
  "iglp": true,
  "iglp_mode": 1,
  "interleave": false,
  "lazy_rescale": true,
  "lds_k_group_pad": 8,
  "lds_num_buffers": 1,
  "lds_row_pad": 4,
  "num_persistent": 304,
  "o_store_width": 4,
  "persist_decode": "auto",
  "persistent": true,
  "pv_loop_order": "k_major",
  "pv_priority": 1,
  "pv_sched_fence_mask": null,
  "use_cfvst": null,
  "use_exp2_fast": null,
  "use_v_swizzle": null,
  "v_row_pad": null,
  "waves_per_eu": 3
}
```

<!-- END MEASURED CONFIGURATIONS -->
