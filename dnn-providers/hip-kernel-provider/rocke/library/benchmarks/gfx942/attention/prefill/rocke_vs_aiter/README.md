

## Baseline vs winning configurations

<!-- BEGIN MEASURED CONFIGURATIONS -->

| B | S | Speedup | Paired min–max | Baseline kernel | Winning kernel |
|---:|---:|---:|---|---|---|
| 1 | 1024 | **1.64699×** | 1.64198–1.64931× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_dsplit` |
| 1 | 2048 | **1.20472×** | 1.20405–1.20677× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_gfx942_wpe2` | `rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_pvkmaj_dsplit` |
| 1 | 4096 | **1.15644×** | 1.15277–1.15950× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit` |
| 16 | 1024 | **1.09557×** | 1.09059–1.09680× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe4_krowpad4_iglp1_iglpm1_pvkmaj_prio2_dsplit` |
| 16 | 2048 | **1.11065×** | 1.10802–1.11295× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe4_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit` |
| 16 | 4096 | **1.09801×** | 1.09423–1.09848× | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2` | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe1_krowpad4_iglp1_iglpm1_pvkmaj_prio3_dsplit` |

### B=1, S=1024

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

### B=1, S=2048

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

### B=1, S=4096

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

### B=16, S=1024

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

**Winning kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe4_krowpad4_iglp1_iglpm1_pvkmaj_prio2_dsplit`

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
  "waves_per_eu": 4
}
```

### B=16, S=2048

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

**Winning kernel:** `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe4_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit`

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
  "waves_per_eu": 4
}
```

### B=16, S=4096

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

<!-- END MEASURED CONFIGURATIONS -->
