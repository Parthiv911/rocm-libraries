# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dense flash-attention prefill kernel for gfx942 (CDNA3).

Port of the gfx950 dense prefill kernel (``kernels/gfx950/attention_dense.py``,
from PR #9480) to CDNA3.

This is a SEPARATE kernel from the gfx950 sibling by design: the gfx950 body bakes
in CDNA4-only primitives that do not exist on gfx942, so the two algorithms
genuinely diverge (per ``dsl_docs/architecture/multi_arch_data_layout.md``: split
into per-gfx modules the moment the K-loop shape / memory path / fused phases
differ). Keeping them in separate files also keeps the gfx950 golden IR
(``tests/golden/attention_dense_ir_sha256.json``) byte-identical by construction.

Why this is a port, not a copy (the CDNA3 deltas)
-------------------------------------------------
  * **MFMA atom.** gfx950 uses the wide-K ``mfma_f32_32x32x16`` atom (K=16). CDNA3
    has NO 32x32x16 fp16/bf16 instruction — only ``mfma_f32_32x32x8`` (K=8, the
    ``.1k`` bf16 variant). The 32x32x16 warp tile lowers to 2x ``32x32x8`` on
    gfx942 (K-loop doubling; A/B repack ``<8xelem>`` -> ``<4xelem>``/lane). The
    C-output lane layout is IDENTICAL between the two atoms, so softmax reductions
    and the epilogue are unchanged. See ``helpers.attention.mfma_32x32x8_for_dtype``.
  * **Conflict-free V.** gfx950's CK-1 transposed PV uses ``ds_read_b64_tr_b16``
    (transpose read) for the half-local V load. gfx942 has NO ``ds_read_tr16`` and
    NO ``permlane32_swap``. The CDNA3 vehicle is a ``perm_b32`` STORE-path transpose
    (CK ``transpose_vectors`` masks) into a transposed ``V_lds`` — the same vehicle in
    ``kernels/gfx942/attention_tiled_2d.py`` (``use_conflict_free_v_store``). Its
    parked "1/8 sign-flip" symptom was reproduced-or-disproven before anything was
    lifted: DISPROVEN cross-part (both the 228- and 304-CU parts, bit-identical, no
    sign mismatches), so the vehicle is proven correct rather than merely untested.
    It is lifted here for D128 (fp16 and bf16) as :func:`_use_cfvst` — the barrier caveat is
    that V publication becomes an in-loop ``ds_write``, so the tile-START rendezvous
    needs an lgkmcnt drain (``sync_lds_only``), not a bare barrier. The disproof
    procedure and its evidence are recorded in the gfx942 dense-prefill
    optimization plan, which is kept outside the repo.
  * **LDS / occupancy.** gfx942 has 64 KB LDS/CU (vs gfx950's 160 KB); occupancy /
    ``num_persistent`` / block sizing must be re-derived for the 228- and 304-CU
    gfx942 parts rather than inherited from the gfx950 spec.

Problem category (drives the optimization order)
------------------------------------------------
**Measured bound (rocprofv3 PMC over the Step-0 config sweep): occupancy-bound ->
MFMA-starved.** MFMA issue rate sits far under the runbook's compute-bound
threshold while waves/CU sit at a small fraction of the per-CU maximum; L2 hit rate
and the memory-stall counter rule out HBM/L2 as the primary limit, and no in-scope
config spills. Per the runbook §3.1a decision tree -- a short-circuiting chain whose
FIRST branch is ``occupancy < 0.5`` -- the classification resolves at occupancy and
never reaches the ``LDS_BOUND`` branch.

**This SUPERSEDES the earlier "LDS-bank-conflict-bound" framing** that was inherited
from the gfx942 tiled-2D kernel and used to order P0-P5. That framing was a good
guide and it paid off -- P1 conflict-free V and the P3 D64 K bank-pad each removed a
large bank-conflict cost -- but it does not describe the kernel that remains. Two
findings retire it:

  * Post-P1/P3 the residual bank-conflict RATE no longer tracks throughput: within a
    fixed shape/dtype, configs with several times the conflict rate are level with or
    faster than their low-rate siblings. The counter is conflicts per LDS-active
    cycle, so a lever that removes LDS instructions (cfvst: 128 -> ~32 LDS instrs per
    tile at D128) shrinks the denominator and can RAISE the rate while lowering the
    cost. Rate is not cost; only an ablation prices LDS.
  * Occupancy trips the decision tree first, so LDS conflicts cannot be the governing
    limiter regardless of their rate.

What still follows from this, unchanged: NOT compute-bound and NOT HBM-bound, so
compute-side scheduling levers that win on the compute-bound gfx950 dense kernel
(``s_setprio``, diagonal two-phase peel) stay PROVEN-NEGATIVE on gfx942 and are NOT
ported, and bandwidth-saving levers stay demoted. What changes: the dominant
remaining lever is **occupancy** (D128 -> 2 WG/CU), not a further LDS-conflict fix.
That is a combined-constraint problem -- it needs LDS <= 32 KB AND the total-register
floor cut TOGETHER (D128 carries a large AGPR residency on top of its architected
VGPRs). Dropping the LDS pads alone to buy the LDS half is PROVEN-NEGATIVE and
catastrophically so: occupancy does not move (the register floor co-limits) and the
K/V bank conflicts come straight back.

Measured conflict rates, pipe utilisation and per-lever deltas are recorded outside
the repo -- see the optimization plan and the protected results page.

Implementation status (the optimization plan holds the full ordered work list)
-----------------------------------------------------------------------------
  * P0  enablement + 32x32x8 atom + K-loop doubling ............ DONE (this file)
  * P1  conflict-free V (perm_b32 store-path transpose) ........ DONE (D128)
  * P2  exp2_fast + fused rescale .............................. DONE (exp2_fast for
        ALL configs incl. bf16 D128; fused rescale bit-identical, and is what
        enabled bf16 D64 exp2_fast)
  * P3  occupancy: waves-per-eu tune (bf16 D64 -> wpe4 = 2 WG/CU),
        D64 K bank-pad, wide4 (WG=256), K single-buffer ........... IN PROGRESS
        (waves-per-eu DONE via _tuned_waves_per_eu; **D64 K bank-pad DONE and ADOPTED**
        -- cross-part-confirmed large win on both D64 dtypes, wired per-config via
        the SHARED ``spec.lds_k_group_pad`` (the same field and formula gfx950
        uses -- one lever, not a per-arch duplicate), golden re-blessed against the
        shipped path; wide4 / K-single-buf still TODO. D128 -> 2 WG/CU is the open
        item and needs the LDS and register floors cut together -- see "Problem
        category" above. ``block_m`` is a further occupancy axis: contrary to the
        gfx950 sibling's "kernel FAULTS at other values" comment, every ``BLOCK_M``
        use in THIS body is parametric (the invariant is ``BLOCK_M == 32 * WAVES``),
        but shrinking it adds CTAs without adding a CTA/CU -- LDS is BLOCK_M-
        invariant -- and costs VGPRs on the K-side DMA addressing, so it is a modest
        win on two of four shapes and a loss elsewhere; now exposed as
        ``Gfx942AttentionDenseSpec.block_m`` so re-testing it is a spec field rather
        than a source edit, and still DEFAULT-OFF at the shared ``_BLOCK_M``.)
  * P4  persistent grid-stride + qb/hkv_major decode .......... DONE (both decodes;
        shared _run_work_item body; auto-on for large Sq via dispatch, validated
        cross-part for every config by the ON/OFF ablation -- KEEP everywhere)
  * P5  diagonal masking (re-test only), partial-vmcnt prefetch  N/A (see below)

P5 is deliberately no-op for this kernel (a decision, not code):
  * Diagonal two-phase peel: the plan gates it on "re-test ONLY if the bound shifts
    to compute after P1-P4." It has not -- the rocprof sweep confirms D128 is still
    occupancy/latency-bound at 1 WG/CU and nowhere near compute-bound -- and the peel
    was proven negative on gfx942 tiled_2d via register pressure, which D128 (at its
    register floor) is the most exposed to. Keep the single-phase mask.
  * partial-vmcnt software prefetch: N/A on the default build. It is a
    double-buffered pipelining lever and the default build is single-buffered
    (NBUF=1), so there is no prefetch to partially overlap. A full K+V NBUF=2 does not
    fit 64 KB LDS at D128 ``block_n=64``; at ``block_n=32`` it fits, but ``block_n=32``
    was GPU-timed in the Step-0 funnel and is proven-negative for fp16 D128 and only
    part-dependently positive for bf16 D128 (halving the KV tile doubles the tile/grid
    count and the extra loop/barrier overhead outweighs the LDS relief while occupancy
    stays pinned). The BF16 D128 CFVST path on the persistent grid can opt in to
    double only K at ``block_n=64`` and/or prefetch next V into registers. Both are
    separate opt-in tuning knobs, off by default; the combined path fits and waits
    with a full ``vmcnt(0)``, not a partial one. Bounding the
    live set with partial ``vmcnt=N`` waits entangles with the K async DMA, which
    shares the same counter -- see the reverted cfvst-chunking experiment recorded in
    the ``V_ITEMS`` NOTE below.

P1 conflict-free V (:func:`_use_cfvst`): V is stored TRANSPOSED via a perm_b32
store-path transpose so the PV A-operand read is a contiguous ds_read_b64 instead of
P0's 4 element-wise ds_read_u16 (128 -> ~32 LDS instrs/tile at D128). Gated to
**D128**: D64 is VGPR-bound so the register round-trip regresses it. bf16 D128 fits
the waves-per-eu=2 cap once its P cast is the NaN-free pair pack. The lever
is numerically identity-preserving (same MFMA operands; only the LDS layout and read
width change) -- the cohort's ``max_abs`` is unchanged from the naive path. The
conflict-free vehicle was proven correct cross-part before being lifted (see the
optimization plan for the disproof procedure).

P0 is CORRECTNESS-FIRST: the naive-V path (D64) is non-pipelined (a
single LDS buffer) and reads V element-wise; the remaining perf levers (P2-P4) layer
on top. It is validated against an fp32 SDPA reference across the in-scope cohort.

Opt-in pipeline levers on the bf16 D128 causal CFVST body (CFVST is the D128 default)
on the persistent grid: ``lds_num_buffers=2``
independently uses two K LDS slots and ``v_reg_prefetch=True`` loads next-tile V
registers before current PV. V LDS stays single-buffered for all combinations. All
shipped defaults use ``lds_num_buffers=1`` and ``v_reg_prefetch=False``.

:func:`supports_attention_dense` is the SINGLE gate: it rejects every spec
:func:`build_attention_dense` cannot emit -- including the modes deferred to later
phases -- so ``supports_attention_dense(spec)[0] is True`` implies the build
succeeds. Dispatch keys on it, which is what stops an out-of-scope request from
selecting this arm and then failing at build time.

The compile-time spec (:class:`AttentionDenseSpec`) is REUSED from the gfx950
module: batch / seqlen / heads / head_size / causal / dtype / knobs are arch-neutral
(compile-time shape + tuning). gfx942-specific DEFAULTS for those shared fields (e.g.
num_persistent for the gfx942 CU count) are applied in the dispatch layer, not by
forking the dataclass; the gfx942-ONLY fields are added by SUBCLASSING it (see
:class:`Gfx942AttentionDenseSpec` below).

Knobs that are gfx942-ONLY -- because gfx950 either cannot emit them or emits them
from a different derivation -- are FLAT FIELDS on :class:`Gfx942AttentionDenseSpec`,
a frozen subclass of the shared spec. One flat spec per arch, one builder name, one
signature: every entry point here takes a single spec object, exactly like the gfx950
sibling, so a kernel-descriptor format does not need an arch-specific shape. The
specs stay different in CONTENT between arches; they are no longer different in SHAPE.

Subclassing rather than extending the shared dataclass is what keeps the gfx950
golden still: ``AttentionDenseSpec.kernel_name()`` is a hand-curated parts list, not
field auto-serialization, and gfx950 keeps using the base class untouched. The gfx942
extras are tagged by the :meth:`Gfx942AttentionDenseSpec.kernel_name` OVERRIDE, which
only ever appends -- and appends nothing at the shipped configuration, so the gfx942
default name is byte-identical to the pre-split one too.

Entry points accept a plain :class:`AttentionDenseSpec` as well: it is promoted to
the subclass at its shipped defaults (:func:`_as_gfx942_spec`), so dispatch -- which
never sets a gfx942-private knob -- and every shape-only caller are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as _dataclass_fields

from rocke.core.ir import (
    IRBuilder,
    KernelDef,
    PtrType,
    VectorType,
    BF16,
    F16,
    F32,
    I32,
    I64,
)
from rocke.helpers.attention import mfma_32x32x8_for_dtype

# Shared problem/geometry fields live in an architecture-neutral module.
from kernels.common.attention_dense_decode import (
    NONPERSIST_DECODES,
    PERSIST_DECODES,
    DecodeShape,
    PersistBtHkvMinor,
    PersistHkvMajor,
    PersistHqMinorSwz,
    PersistQbMajor,
    fastdiv_kernargs,
    make_fastdiv,
)
from kernels.common.attention_dense_spec import (
    AttentionDenseSpec,
    DENSE_TILE_GEOMETRIES,
    attention_dense_cache_key,
    check_dense_spec_preflight,
    partial_tile_reason,
)

# C-output lane maps: IDENTICAL between the 32x32x8 (gfx942) and 32x32x16 (gfx950)
# atoms (mfma_atom_catalog.md), so the softmax reductions + epilogue port verbatim.
from kernels.gfx942.attention_tiled_2d import _mfma_32x32_c_row, _mfma_32x32_c_col
from kernels.common.attention_dense_kv_split import (
    SEGMENT_PARAMS,
    allocate_segment_workspace,
    check_kv_segments,
    declare_segment_params,
    emit_segment_store,
    launch_segment_combine,
    segment_tile_range,
)
from kernels.common._lse_store import (
    dense_row_has_key,
    lse_store,
    validate_dense_lse,
    zero_keyless_row,
)

LOG2E = 1.4426950408889634
_DTYPE_IR = {"bf16": BF16, "fp16": F16}

# Pipeline constants (gfx942 BF16 D128 K-buffer count and V prefetch are opt-in).
# The shipped geometry starts at 256 query rows per CTA = 8 wave64s. ``block_m``
# is inherited from AttentionDenseSpec, so the grid and body read the same
# immutable value and cannot disagree silently (rows written twice/never).
_DEFAULT_BLOCK_M = int(DENSE_TILE_GEOMETRIES["default"]["block_m"])
# Compatibility alias used by existing tests and out-of-tree geometry probes.
_BLOCK_M = _DEFAULT_BLOCK_M
#
# What that single constant still has to satisfy for THIS body: the wave count is
# WAVES = _BLOCK_M // 32, a FLOOR. At a non-multiple of 32 the emitted wave count
# under-covers the query tile and the top _BLOCK_M % 32 rows of every block get no
# wave at all -- they are never computed and never written, a silent wrong answer with
# no fault. Bind the invariant at import instead of by comment. Not an `assert`:
# `python -O` strips asserts and this guards a silent wrong answer.
if _BLOCK_M % 32 != 0:
    raise ValueError(
        f"_BLOCK_M ({_BLOCK_M}) must be a multiple of 32: the gfx942 dense body "
        f"derives its wave count as WAVES = _BLOCK_M // 32, which floors, so a "
        f"non-multiple leaves the top {_BLOCK_M % 32} query rows of every block "
        f"uncovered by any wave -- never computed and never written"
    )


# Shipped defaults for the gfx942-private fields below. Named constants rather than
# repeated literals because each one is used as both a field default/policy result
# and the baseline a conditional name tag compares against.
_DEFAULT_LDS_ROW_PAD = 8
_DEFAULT_IGLP = False
_DEFAULT_IGLP_MODE = 0
_DEFAULT_PV_LOOP_ORDER = "d_major"
_DEFAULT_O_STORE_WIDTH = 4
_PV_LOOP_ORDERS = ("d_major", "k_major")
_O_STORE_WIDTHS = (1, 2, 4)
# sched_barrier mask: 11 instruction-class bits (__builtin_amdgcn_sched_barrier).
_SCHED_BARRIER_MASK_MAX = 0x7FF


@dataclass(frozen=True)
class Gfx942AttentionDenseSpec(AttentionDenseSpec):
    """The gfx942 dense prefill spec: the shared spec plus gfx942-private knobs.

    A SUBCLASS of :class:`AttentionDenseSpec`, not a fork and not a sidecar struct.
    The base class carries the arch-neutral compile-time problem (shape, plus the
    knobs gfx950 also owns); the fields added here are levers only this body can
    emit, either because gfx950 cannot emit them at all or because it derives them
    differently. Adding them flat -- as siblings of the shared fields rather than
    nested behind a second argument -- is what keeps ONE builder signature,
    ``build_attention_dense(spec, *, arch=...)``, on every arch: the arch varies the
    spec TYPE, never the call shape. That in turn is what lets one kernel-descriptor
    file format describe a rocke instance on any part.

    Extending the spec does not disturb gfx950: ``kernel_name`` is a hand-curated
    parts list, not an auto-serialization of the field set, so the gfx950 goldens
    (``platform/tests/golden/rocke_representative_ir_sha256/``) are untouched by
    fields that only this subclass has. Dispatch still leaves every field below at
    its default, so these knobs stay invisible in production and fully addressable
    from a harness.

    Tri-state fields: ``None`` means "use the measured policy"
    ---------------------------------------------------------
    ``use_cfvst`` / ``use_exp2_fast`` / ``v_row_pad`` / ``use_v_swizzle`` default to
    ``None``, which resolves through the shipping policy (:func:`_use_cfvst`,
    :func:`_use_exp2_fast`, :func:`_v_row_pad`, :func:`_use_v_swizzle`). That
    tri-state is load-bearing: a harness that omits a VALUED field freezes the config
    at whatever the default happened to be the day it was written and then silently
    reports a stale verdict (that is exactly how a real regression got mis-reported
    in this tree). A harness that omits a ``None`` field auto-tracks whatever ships.
    Omission becomes safe.

    The policy functions stay the SINGLE resolution point -- their docstrings are the
    evidence ledger, and their verdicts are deliberately NOT copied into the defaults
    here, so there is nothing to drift.

    Naming / launcher-cache safety
    ------------------------------
    Every field that can change codegen is tagged into :meth:`kernel_name`, but ONLY
    when it differs from the default (or, for the tri-state fields, from what the
    policy would have produced) -- so at shipped defaults the names, and the goldens,
    are byte-identical to the pre-subclass ones. Conditional tagging is the existing
    idiom (the base ``kernel_name`` emits ``kpad{N}`` only on the packed path and
    ``persist{N}`` only when persistent), and it is load-bearing rather than
    cosmetic: ``_DENSE_LAUNCHER_CACHE`` is keyed on the kernel name and its
    ``assert art.kernel_name == key`` PASSES on a name collision, so an untagged
    IR-affecting knob silently serves a stale binary. This kernel has shipped that
    exact bug twice (``batch``, then ``waves_per_eu``).

    Overriding :meth:`kernel_name` rather than post-processing it in a free function
    is what makes identity correct BY CONSTRUCTION: any code path holding a spec --
    the builder, the launcher cache, dispatch, a harness -- gets the gfx942 symbol
    from the object itself, and cannot forget to apply the arch suffix.

    :func:`supports_attention_dense` validates the added fields, not just the shared
    ones: the module contract is that ``supports(spec)[0] is True`` implies
    ``build_attention_dense(spec)`` succeeds.
    """

    # block_m is inherited from AttentionDenseSpec. Both gfx942 and gfx950 now
    # consume that field directly; gfx950 admits its named 128/256 presets while
    # this body retains the broader validated multiple-of-32 sweep surface.

    # lds_row_pad: K_lds per-ROW bank-conflict pad, in elements. Applied only when one
    #   K row is packed per async-DMA instruction (D128 here); see
    #   :func:`_lds_row_stride` for why D64 cannot carry a per-row pad and uses the
    #   per-GROUP ``spec.lds_k_group_pad`` instead. The two are DIFFERENT,
    #   mutually-exclusive levers that merely happen to share the value 8.
    #
    #   SETTLED -- but scoped strictly to REMOVAL. Dropping this pad (and
    #   ``v_row_pad``) to cut D128 LDS under the 32 KB a 2nd WG/CU needs is
    #   PROVEN-NEGATIVE and catastrophically so: D128 stays at 1 WG/CU even unpadded
    #   because the register floor co-limits, so the pads come out for nothing and the
    #   K/QK (and cfvst V) bank conflicts come straight back. Do not re-attempt pad
    #   REMOVAL.
    #
    #   The VALUE is a different question and it is OPEN. 8 is INHERITED from the
    #   gfx950 sibling and has never been re-derived for gfx942. Two concrete reasons
    #   to distrust it:
    #     * Bank geometry does not transfer. gfx950 has 64 LDS banks, gfx942 has 32
    #       (``platform/dsl_docs/optimization/arch/gfx950.md`` §21.2 "LDS specifics"
    #       table; ``platform/dsl_docs/optimization/arch/gfx942.md`` §21.2), and the
    #       shared pad derivation is written against the 64-bank model verbatim --
    #       "LDS delivers 64 banks x 4 B = 256 B/cycle" (``AttentionDenseSpec``'s
    #       ``lds_k_group_pad`` comment in ``kernels/gfx950/attention_dense.py``). The
    #       same gfx950 table also lists the preferred swizzle as *padding* for
    #       gfx950 and *XOR* for gfx942, so even the FORM of the fix is not obviously
    #       transferable, let alone the magnitude.
    #     * 8 did not survive a sweep on the arch it came from. gfx950's own sweep of
    #       the analogous V pad measured conflicts {pad 0: 30, pad 8: 29, pad 16: 11,
    #       pad 32: 0} -- i.e. +8 was essentially indistinguishable from no pad at all
    #       (``Gfx950AttentionDenseSpec.lds_v_row_pad`` in the gfx950 sibling).
    #   Re-deriving both pad VALUES on gfx942 is the pad-value sweep tracked in the
    #   optimization plan; THIS field is the knob that sweep turns.
    lds_row_pad: int = _DEFAULT_LDS_ROW_PAD

    # v_row_pad: V^T row (token axis) width pad, in elements, for the P1 conflict-free-V
    #   store. V_lds is transposed to [D, block_n+pad] so the PV A-operand read is a
    #   contiguous ds_read_b64. On the cfvst path (D128) the pad is DERIVED from
    #   block_n by :func:`_v_row_pad` so the row is the smallest pow2 the XOR bank-
    #   conflict swizzle needs (:func:`_v_swizzle_width`): pad 64->0, 32->32, 128->0; 8
    #   (the original unswizzled layout) elsewhere. The pad VALUE is no longer OPEN --
    #   the derivation supersedes the inherited-8 pad-value sweep, since the swizzle
    #   constrains the row to a pow2 (an arbitrary swept pad would just disable it).
    #   REMOVAL is still settled-negative (see ``lds_row_pad``). Whether the swizzle is
    #   EMITTED is the separate, independently-sweepable ``use_v_swizzle`` knob, so a
    #   pad sweep no longer silently toggles it. Resolved via
    #   :meth:`resolved_v_row_pad`, read by both the ``V_lds`` alloc and the
    #   :func:`_lds_bytes` budget so the two cannot drift.
    v_row_pad: int | None = None

    # use_cfvst: force the P1 conflict-free perm_b32 store-path V transpose on/off.
    #   None (default) -> :func:`_use_cfvst`, which owns the measured verdict.
    #   Forcing it ON where the policy says OFF is REJECTED by
    #   supports_attention_dense (D64 regresses under it -- VGPR-bound), so this
    #   field can only turn cfvst off, or
    #   pin it on where it already is.
    use_cfvst: bool | None = None

    # use_v_swizzle: force the V^T XOR bank-conflict swizzle on/off, INDEPENDENTLY of
    #   the pad value -- so the pad-value sweep can vary ``v_row_pad`` without silently
    #   toggling the swizzle (an explicit non-derived pad used to turn it off by making
    #   the row non-pow2). None (default) -> :func:`_use_v_swizzle` (the cfvst path).
    #   Only the transposed-V (D128) store has a V_lds to swizzle; forcing it ON
    #   off that path, or with a pad that leaves V_LDROW non-pow2 / < 64, is REJECTED
    #   by supports_attention_dense rather than silently mis-addressing LDS.
    use_v_swizzle: bool | None = None

    # use_exp2_fast: force the P2 single-instruction exp2 on/off.
    #   None (default) -> :func:`_use_exp2_fast`, which owns the measured verdict
    #   (on for all configs except bf16 head_dim=128 on the default grid, which
    #   reverts to plain exp2 -- see _use_exp2_fast for the grid gate).
    #   Numerically safe in both directions here -- both softmax arguments are
    #   always <= 0 -- so correctness never depends on it; the gate is pure perf.
    use_exp2_fast: bool | None = None

    # NOTE -- waves_per_eu is deliberately NOT redeclared here. It is an INHERITED
    #   field (``AttentionDenseSpec.waves_per_eu``), and the flat spec is what makes
    #   that possible: the old sidecar struct carried a second, tri-state
    #   ``waves_per_eu`` whose only resolution rule was "None -> spec.waves_per_eu",
    #   i.e. a same-named shadow of the field one attribute lookup away. Flattening
    #   collapses the two into the one the dispatch spec factory already fills from
    #   :func:`_tuned_waves_per_eu`, so there is exactly one place to set it and the
    #   ``wpe{N}`` name token and the emitted ``amdgpu-waves-per-eu`` attribute cannot
    #   disagree about which of two fields won.

    # iglp: emit runbook lever 7 (optimization_runbook.md §8.4) -- ``iglp_opt``, the
    #   canned backend MFMA / ds_read / ds_write interleave (llvm.amdgcn.iglp.opt),
    #   distinct from the proven-negative s_setprio / sched_barrier hints (plan §2
    #   DO-NOT-PORT). Its precondition is in-loop ds_write traffic to interleave
    #   against the MFMAs -- which only the P1 cfvst path has (V is stored to V_lds
    #   inside the loop; the naive path is direct-load, 0 ds_write, so iglp is a
    #   priori neutral). Placed once at the top of the main-loop body per the runbook.
    #
    #   RESOLVED -- do not re-attempt. This is no longer a pending probe: the GPU
    #   timing it was gated on has been run, cross-part, on the cfvst config it was
    #   supposed to help.
    #     * Static (runbook §8.4): the intrinsic is a compile-time scheduler DIRECTIVE
    #       and leaves NO runtime instruction -- the ISA opcode histogram is unchanged
    #       and ds_read / ds_write / mfma counts are identical, with no spill or LDS
    #       delta. So the effect could never be confirmed by artifact and had to be
    #       timed.
    #     * Timed: PROVEN-NEUTRAL on both parts. Its only visible effect is shifting
    #       the architected-VGPR / AGPR split (88/128 -> 128/128 at fp16 D128) at
    #       equal total registers, which does not move the clock.
    #   The neutrality is itself the useful result, per runbook §8.6 (barrier-bound vs
    #   schedule-bound): a canned MFMA/DS interleave buying nothing means the main
    #   loop is NOT schedule-bound -- it is barrier-rendezvous-bound, which no
    #   scheduling hint can fix. That matches the measured occupancy bound in the
    #   module docstring, and it is why the whole scheduling-hint family stays
    #   unported. Stays default OFF; it is kept as a knob only because it toggles IR.
    iglp: bool = _DEFAULT_IGLP

    # --- Performance-only codegen knobs, keyword-only so existing positional callers
    #   do not shift. Every legal value computes the same attention output; the
    #   defaults reproduce the shipped kernel byte-for-byte. The shared
    #   ``lazy_rescale`` field has no effect here: this body always rescales, and
    #   only the shared kernel_name() reads it (the ``lazyrs`` token).
    #
    # lds_num_buffers: K LDS buffer count, independent of V register prefetch.
    #   1 keeps a single K slot; 2 allows K[j+1] DMA to overlap PV[j] in a
    #   different slot. V LDS is always single-buffered. The opt-in pipeline
    #   requires persistent=True and BF16 D128 causal CFVST at block_n=64,
    #   with or without causal_diag_split.
    lds_num_buffers: int = field(default=1, kw_only=True)
    # v_reg_prefetch: issue the next V tile's register loads before PV[j]
    #   rather than after PV[j]. Independent of the K LDS buffer count; same
    #   scope as lds_num_buffers=2.
    v_reg_prefetch: bool = field(default=False, kw_only=True)
    # iglp_mode: the iglp_opt argument emitted when ``iglp`` is on (0 or 1).
    iglp_mode: int = field(default=_DEFAULT_IGLP_MODE, kw_only=True)
    # pv_loop_order: PV MFMA traversal. "d_major" keeps the output tile outer,
    #   "k_major" the key step outer; each tile accumulates in the same order.
    pv_loop_order: str = field(default=_DEFAULT_PV_LOOP_ORDER, kw_only=True)
    # o_store_width: output elements per global store (1, 2 or 4). bf16 only below
    #   4: narrower fp16 stores change which f32->f16 conversion the backend selects
    #   for some elements, moving them by one ulp, so fp16 is fixed at 4.
    o_store_width: int = field(default=_DEFAULT_O_STORE_WIDTH, kw_only=True)
    # pv_priority: s_setprio level around the PV MFMAs; 0 emits nothing. Off by
    #   default: PV-only s_setprio is recorded negative on this occupancy-bound
    #   kernel (module docstring); the knob exists so a sweep can re-measure it.
    pv_priority: int = field(default=0, kw_only=True)
    # pv_sched_fence_mask: sched_barrier(mask) between the softmax and the PV
    #   MFMAs; None emits no fence. Off by default for the same reason.
    pv_sched_fence_mask: int | None = field(default=None, kw_only=True)
    # causal_diag_split: causal without a sliding window only -- an unmasked KV loop
    #   over below-diagonal tiles plus a masked diagonal tail, instead of masking
    #   every tile. The CFVST KDB/VPF pipeline also preserves K/V publication across
    #   the two loop phases. Off by default; kept to re-measure on dense.
    causal_diag_split: bool = field(default=False, kw_only=True)
    # window_wave_skip: causal sliding window only -- a wave whose 32 query rows see
    #   no key of a tile (all past the diagonal or all behind the window) skips that
    #   tile's score, softmax and PV step; staging and barriers stay in every wave.
    #   Off by default; the dispatch narrow-window geometry turns it on.
    window_wave_skip: bool = field(default=False, kw_only=True)
    # emit_lse: also write the natural-log softmax normalizer (LSE) as FP32
    #   [B, Hq, Sq, 1] through a trailing ``lse_ptr`` kernel argument. Off by default;
    #   off emits no new IR, so the shipped kernel and its goldens are unchanged.
    emit_lse: bool = field(default=False, kw_only=True)
    # k_lds_swizzle (prototype): unpadded K_lds rows with an XOR bank swizzle,
    #   col' = col ^ ((row & 15) << 2), in place of the lds_row_pad pad. The K DMA
    #   fetches the permuted global column so each lane still writes LDS dword `lane`.
    k_lds_swizzle: bool = field(default=False, kw_only=True)
    # k_read_b128 (prototype): paired-step D map (lane half h holds dims 16t+8h..+7
    #   for steps 2t, 2t+1) so K is read with ds_read_b128 and Q with dwordx4 loads.
    #   Needs a 16 B-aligned K row (lds_row_pad % 8 == 0); pad 8 is conflict-free.
    k_read_b128: bool = field(default=False, kw_only=True)
    # tile_interleave (prototype): unmasked pipeline tiles interleave VALU with MFMA
    #   inside the tile -- QK nsub1 MFMAs cover nsub0's scale + partial max, and PV
    #   runs per P pack with the next pack's exps and the O rescale between its
    #   MFMAs. Every accumulator keeps its op order, so the output is bit-identical.
    tile_interleave: bool = field(default=False, kw_only=True)
    # tilv_v_write_qk (prototype, needs tile_interleave): carry V(j)'s register
    #   payload into tile j and write it to LDS between QK nsub0's MFMAs (then
    #   lgkmcnt(0) + barrier before nsub1), instead of a serial V-write section
    #   between two barriers after PV. V(j+1) loads issue in the PV MFMA gaps.
    tilv_v_write_qk: bool = field(default=False, kw_only=True)
    # tilv_rescale_skip (prototype, needs tile_interleave): when alpha == 1 on every
    #   lane (no row max changed), run a PV copy without the O rescale. O * 1 == O,
    #   so the result is bit-identical.
    tilv_rescale_skip: bool = field(default=False, kw_only=True)
    # qk_lookahead (prototype, needs tilv_v_write_qk): pipeline step j carries tile
    #   j's raw scores and runs QK(j+1) (K slot (j+1)%2) under tile j's V write and
    #   softmax head; K(j+2) DMA goes to slot j%2. Same op order: bit-identical.
    qk_lookahead: bool = field(default=False, kw_only=True)
    # qkla_pack_ahead (prototype, needs qk_lookahead): P packs 1..N exps + packs
    #   move from the PV gaps into the head under QK(j+1), paced evenly over its
    #   MFMAs. The row-sum keeps its pack order: bit-identical.
    qkla_pack_ahead: int = field(default=0, kw_only=True)
    # qkla_rescale_plain (prototype, needs qk_lookahead): route each O rescale
    #   product through an optimization barrier so it stays v_mul_f32 (a packed
    #   v_pk_mul_f32 cannot issue while either wave's MFMA holds the pipe).
    qkla_rescale_plain: bool = field(default=False, kw_only=True)
    # qkla_read_ahead (prototype, needs qk_lookahead): QK(j+1) K ds_reads in flight
    #   ahead of their MFMA; 0 keeps the default of 3.
    qkla_read_ahead: int = field(default=0, kw_only=True)
    # qkla_qk_tail (prototype, needs qk_lookahead): the last N QK(j+1) MFMAs move
    #   below the mid barrier, one after each of PV(j)'s first N MFMAs, and the
    #   K(j+2) DMA starts after them (a K read after a K DMA row waits vmcnt(0):
    #   the scopes cannot tell the two slots apart). K(j+1) is already in its LDS
    #   slot; each accumulator keeps its step order: bit-identical.
    qkla_qk_tail: int = field(default=0, kw_only=True)
    # qkla_rhythm_b / qkla_rhythm_pv (prototype, need qk_lookahead): ask the
    #   scheduler for a fixed rhythm of 1 MFMA then N VALU through the region above
    #   the mid barrier (b) and the PV region below it (pv), via
    #   sched_group_barrier. Scheduling hints only: bit-identical.
    qkla_rhythm_b: int = field(default=0, kw_only=True)
    # qkla_pv_spread (prototype, needs qk_lookahead): even PV chunks. The exps of
    #   packs PA+1..7 spread evenly over the PV MFMAs (each pack built right after
    #   its last exp, before its PV), the QK tail on every other PV MFMA, the K DMA
    #   after the tail. Same op order per accumulator and row sum: bit-identical.
    qkla_pv_spread: bool = field(default=False, kw_only=True)
    # qkla_pv_fence (prototype, needs qkla_pv_spread): sched_barrier(0) after every
    #   PV pack (1) or after every PV MFMA's chunk (2), so the spread survives
    #   scheduling. Hints only: bit-identical.
    qkla_pv_fence: int = field(default=0, kw_only=True)
    # qkla_chunked (prototype, needs qk_lookahead): every MFMA of the tile gets a
    #   non-MFMA chunk of about the same estimated cycles. Above the mid barrier the
    #   head (V stores, max, alpha, plain O rescale, packs 0..PA) fills the QK(j+1)
    #   MFMAs; below it the pack exps, V reads, K DMA, V loads and S(j+1)'s scale
    #   fill the PV + QK-tail MFMAs, each chunk fenced. S is carried scaled. Same op
    #   order per value: bit-identical.
    qkla_chunked: bool = field(default=False, kw_only=True)
    # qkla_loads_b (prototype, needs qkla_chunked): issue V(j+1)'s loads in the
    #   chunks above the mid barrier instead of PV.
    qkla_loads_b: bool = field(default=False, kw_only=True)
    # qkla_chunk_fence (prototype): chunk boundary kind for qkla_chunked,
    #   0 = sched_barrier(0), 1 = s_setprio 0 (a hard scheduling boundary).
    qkla_chunk_fence: int = field(default=0, kw_only=True)
    # qkla_chunk_rev (prototype): 0 = first qkla_chunked placement; 1 = K(j+2) DMA
    #   pinned to PV's first chunks, at least one unit per chunk, max steps in pairs,
    #   and the head never exhausted before B's last MFMA. 2 = PV from a fixed
    #   per-chunk plan, V reads two packs ahead, no packed subtracts. 3 = rev 2 with V
    #   reads one pack ahead and the last QKLA_CARRY PV MFMAs of tile j run at the
    #   start of tile j+1 (their V and P carried; zero for the first tile, flushed
    #   at loop exit). 4 = rev 2 with V reads spread one ds_read per chunk (two
    #   packs ahead) and each row-sum add delayed past the next exp (same order).
    #   5 = rev 1's adaptive PV (V reads one pack ahead) plus: no packed subtracts,
    #   each pair's rounding after the next exp, packs finished a chunk early,
    #   S(j+1)'s scale split over the last chunks, delayed row-sum add.
    qkla_chunk_rev: int = field(default=0, kw_only=True)
    # q_lds_stage (prototype): load Q row-contiguous (16 B per lane, 64 B row
    #   segments) and transpose it into the MFMA layout through the V buffer, which
    #   is free until the prologue barrier. Replaces 16 row-scattered 8 B loads.
    q_lds_stage: bool = field(default=False, kw_only=True)
    # o_lds_stage (prototype): after one vmcnt(0)+barrier, stage O through the K
    #   buffer (per-wave, XOR-swizzled) and store it row-contiguous with 16 B per
    #   lane (full 128 B lines) instead of 16 row-scattered 8 B stores.
    o_lds_stage: bool = field(default=False, kw_only=True)
    # qkla_vdb (prototype): two V^T LDS slots. V(j) lives in slot j%2; V(j+1)
    #   (loaded in B) is written into the other slot at the end of PV(j), so the
    #   end barrier publishes it, the mid barrier is dropped, and no V registers
    #   are carried across tiles.
    #   1 = V(j+1) stores in PV(j)'s 2nd half, 2 = in B(j)'s 2nd half.
    qkla_vdb: int = field(default=0, kw_only=True)
    # qkla_split_wait (prototype, needs qkla_vdb + q_lds_stage): the prologue waits
    #   only for Q + K(0) before QK(0); K(1) and V(0) stay in flight across QK(0)
    #   and are drained (and V(0) published) by a second barrier before tile 0.
    qkla_split_wait: bool = field(default=False, kw_only=True)
    # q_full_line (prototype, needs qkla_split_wait + k_read_b128): each Q dwordx4
    #   load covers 8 rows x 128 B (whole cache lines) instead of 16 rows x 64 B;
    #   staged through both V slots (free until V(0) is published after QK(0)).
    q_full_line: bool = field(default=False, kw_only=True)
    # q_batch_wait (prototype, needs q_lds_stage): drain the Q loads (vmcnt 0)
    #   before issuing the prologue K/V loads.
    q_batch_wait: bool = field(default=False, kw_only=True)
    qkla_rhythm_pv: int = field(default=0, kw_only=True)
    # kv_segments: split each work item's KV range into this many contiguous
    #   segments, one CTA each, on the default grid. Each segment writes its
    #   unnormalized FP32 partial O plus its running max / denominator to a
    #   workspace, and a small combine kernel merges them into ``o``. Raises the CTA
    #   count for short query lengths over long KV (chunked prefill), where one CTA
    #   per (query block, head) leaves most CUs idle. 1 (default) emits nothing new.
    kv_segments: int = field(default=1, kw_only=True)
    # xcdlb (prototype, compare-only): the XCD-local load-balanced persistent
    #   decode. wi % 8 is the XCD, wi // 8 its stream (cta_id // 8 + round * 38 at
    #   NP = 304); 4 Q heads per XCD, longest query block first, 38-worker snake.
    xcdlb: bool = field(default=False, kw_only=True)
    # qkla_sub_pin (prototype): opaque s - m subtract at every qkla_chunk_rev (rev >= 2
    # already does it), so SLP cannot pair two into a v_pk_add_f32.
    qkla_sub_pin: bool = field(default=False, kw_only=True)
    # qkla_lhalf (prototype): keep each lane half's partial row sum in the qkla_chunked
    # loops (no per-tile ds_bpermute); the halves are added once after the loops. Exact:
    # alpha is identical on both halves (the row max is reduced across them).
    qkla_lhalf: bool = field(default=False, kw_only=True)
    # qkla_lds_after_mfma (prototype): in each MFMA's chunk region, order the address
    # VALU first, then the MFMA, then the LDS ops (sched_group_barrier), so an LDS op
    # follows an MFMA (0-cycle issue gap) instead of a VALU (8 cycles). Covers the
    # QK(j+1) K reads (steps >= 1), the V(j+1) stores (moved into the MFMA region) and
    # the PV V reads (pack 0 read right after the last QK MFMA).
    qkla_lds_after_mfma: bool = field(default=False, kw_only=True)
    # qkla_vld_start / qkla_vld_stride (prototype): V(j+1) load item k issues in QK(j+1)
    # chunk start + stride*k (default 0 / 1 = the first chunks, unchanged). Spreads the
    # CU-wide VMEM issue burst at the tile start.
    qkla_vld_start: int = field(default=0, kw_only=True)
    qkla_vld_stride: int = field(default=1, kw_only=True)
    # qkla_prio_bal (prototype, 0 = off): s_setprio(qkla_prio_bal) at the tile start
    # (QK(j+1) region), s_setprio(0) when the wave enters PV(j). The wave that reaches PV
    # first drops priority, so the lagging wave on the SIMD wins arbitration and catches up.
    qkla_prio_bal: int = field(default=0, kw_only=True)
    # qkla_max_ahead (prototype): the unmasked main loop computes S(j+1)'s row max in
    # PV(j)'s tail chunks (after S(j+1)'s scale) and carries it, so tile j+1's head skips
    # the max chain + ds_bpermute and starts alpha/rescale/exps at its first chunk.
    # fmax is order-exact, so results are bit-identical. Masked tiles keep their own max.
    qkla_max_ahead: bool = field(default=False, kw_only=True)
    # qkla_even (prototype, 0 = off): target VALU issue cycles per MFMA gap. PV(j) is
    # filled by an earliest-deadline-first scheduler over fine-grained units (s-m 4,
    # exp 16, row-sum add 4, bf16 round 8, pack word 4, K DMA row 8, S scale 4), each
    # element stage in a later chunk than its previous stage (no exp hazards); the
    # QK(j+1) pack-ahead work is split the same way and costed by its VALU cycles.
    qkla_even: int = field(default=0, kw_only=True)
    # qkla_even_mix (prototype, needs qkla_even): in QK(j+1)'s head, interleave the O
    # rescale (64 x v_mul) with the pack-ahead stages by fraction done, so the 16-cycle
    # exps don't cluster in B's second half.
    qkla_even_mix: bool = field(default=False, kw_only=True)
    # wg_timing (prototype, measurement only, persistent grid): every wave writes
    #   16 i32 to a trailing ``ts_ptr`` at [cta*8 + wave]: s_memrealtime at start
    #   (lo, hi) and after its last item's stores drained (lo, hi), HW_ID, XCC_ID,
    #   cta, wave; then s_memtime (shader clock) at start (lo, hi) and end (lo, hi).
    wg_timing: bool = field(default=False, kw_only=True)

    @property
    def fastdiv_divisors(self) -> tuple[str, ...]:
        if self.xcdlb and self.persistent and self.runtime_shape:
            return ("nqb",)  # the xcdlb batch split divides its stream by NQB
        return super().fastdiv_divisors


    @classmethod
    def supported_persist_decodes(cls) -> frozenset[str]:
        # The persistent body takes the shape at runtime, so only decodes with a
        # runtime-operand form (no gqa_pair*, whose NP equals the baked work count).
        return frozenset(
            {
                "auto",
                PersistQbMajor.name,
                PersistHkvMajor.name,
                PersistBtHkvMinor.name,
                PersistHqMinorSwz.name,
            }
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.emit_lse) is not bool:
            raise ValueError(
                f"emit_lse must be a bool, got {type(self.emit_lse).__name__}"
            )
        # Reject, never coerce: bool("false") is True, so a truthiness read would
        # silently build the CFVST kernel the caller asked to turn off.
        if self.use_cfvst is not None and type(self.use_cfvst) is not bool:
            raise ValueError(
                f"use_cfvst must be None or a bool, got {type(self.use_cfvst).__name__}"
            )
        if type(self.lds_num_buffers) is not int:
            raise ValueError(
                "lds_num_buffers must be an int, got "
                f"{type(self.lds_num_buffers).__name__}"
            )
        if type(self.v_reg_prefetch) is not bool:
            raise ValueError(
                "v_reg_prefetch must be a bool, got "
                f"{type(self.v_reg_prefetch).__name__}"
            )

    def resolved_use_cfvst(self) -> bool:
        """Resolved conflict-free-V decision (``None`` -> :func:`_use_cfvst`)."""
        if self.use_cfvst is None:
            return _use_cfvst(self.head_size, self.dtype)
        return bool(self.use_cfvst)

    def resolved_v_row_pad(self) -> int:
        """Resolved V^T row pad (``None`` -> :func:`_v_row_pad`).

        Read by both the builder's ``V_lds`` alloc and the :func:`_lds_bytes` budget, so
        the two cannot drift."""
        if self.v_row_pad is None:
            return _v_row_pad(self.head_size, self.dtype, self.block_n)
        return int(self.v_row_pad)

    def resolved_use_v_swizzle(self) -> bool:
        """Resolved V^T bank-conflict swizzle. Gated on the cfvst path (only the
        transposed-V store has a V_lds to swizzle), and within it independent of
        ``v_row_pad`` so the pad sweep never silently toggles it. ``None`` -> policy
        (:func:`_use_v_swizzle`)."""
        if not self.resolved_use_cfvst():
            return False
        if self.use_v_swizzle is None:
            return _use_v_swizzle(self.head_size, self.dtype)
        return bool(self.use_v_swizzle)

    def resolved_use_exp2_fast(self) -> bool:
        """Resolved exp2_fast decision (``None`` -> :func:`_use_exp2_fast`)."""
        if self.use_exp2_fast is None:
            return _use_exp2_fast(self.head_size, self.dtype, self.persistent)
        return bool(self.use_exp2_fast)

    def resolved_waves_per_eu(self) -> int:
        """Emitted ``amdgpu-waves-per-eu``.

        A plain read of the INHERITED field, kept as a named accessor because the
        builder, the name tag and :func:`supports_attention_dense` all need the same
        int and the shared field is typed loosely enough to be worth normalizing in
        one place. ``dispatch.attention.gfx942_dense._base_spec`` fills it from
        :func:`_tuned_waves_per_eu`, so the shipped value tracks the measured policy
        without this class restating it."""
        return int(self.waves_per_eu)

    @property
    def runtime_shape(self) -> bool:
        """Whether the body reads the problem shape -- batch, seqlen_q, seqlen_kv and
        both head counts -- *only* from its kernel params, baking it nowhere -- so ONE
        compiled kernel serves every shape and those fields drop out of the cache key.
        This is what collapses the AOT batch x seqlen x heads instance explosion.

        Both grids are on the runtime path. The persistent grid derives its work-item
        count and fold constants from the params in the prologue, and its per-item
        decode divides through host-computed fast-division params
        (:attr:`fastdiv_divisors`), so the grid-stride loop stays multiply-and-shift.
        Sliding-window is on the runtime path: its KV-loop prune reads only the query
        block, the window and the runtime trip count, and its zero-trip guard in
        :func:`supports_attention_dense` runs per launch, ahead of the cache lookup.
        ``ragged``/``varlen``/``paged`` never reach the builder
        (:func:`supports_attention_dense` rejects them) but stay in the predicate,
        identical to the gfx950 twin, so a later admit lands already excluded.

        Every other knob that forks the body -- :func:`_use_exp2_fast` included --
        is a function of compile-time config only, never of the problem shape, so
        none of them can leak a per-shape body past this predicate. Keep it that
        way: a policy that reads a shape field would make two shapes share a cache
        key AND a kernel name while lowering to different IR, which neither the
        name assert nor this predicate can catch."""
        return not (self.ragged or self.varlen or self.paged)

    @property
    def runtime_param_fields(self) -> tuple[str, ...]:
        """The shape fields this body reads *only* from its kernel params, and so
        the fields ``attention_dense_cache_key`` may drop.

        The tuple also opens the ABI: the body declares one i32 param per field, in
        this order, after ``scale`` (then the fast-division params of
        :attr:`runtime_kernarg_fields`), and :func:`attention_dense_signature` /
        :func:`attention_dense_runtime_args` read the same tuple -- so the kernel and
        its launchers cannot skew. Kept in sync with :meth:`kernel_name` by hand,
        which drops ``sq``/``sk`` (via :meth:`_shape_name_parts`), ``hq``/``kv`` (via
        :meth:`_head_name_parts`) and ``b{batch}`` for exactly this tuple. A field
        declared here but still baked anywhere would be a cache collision serving one
        shape's binary to another."""
        if not self.runtime_shape:
            return ()
        return ("batch", "seqlen_q", "seqlen_kv", "num_query_heads", "num_kv_heads")

    def _shape_name_parts(self) -> tuple[str, ...]:
        return () if self.runtime_shape else super()._shape_name_parts()

    def _head_name_parts(self) -> tuple[str, ...]:
        return () if self.runtime_shape else super()._head_name_parts()

    def kernel_name(self) -> str:
        """The gfx942 kernel symbol: the shared name plus everything THIS body bakes.

        Overriding the base method rather than post-processing its result in a free
        function is what makes identity correct BY CONSTRUCTION -- every consumer
        (builder, ``_DENSE_LAUNCHER_CACHE``, dispatch's ``kernel_name_override``, any
        harness) reads the symbol off the object and cannot forget to re-apply the
        arch suffix.

        ``AttentionDenseSpec.kernel_name()`` omits both ``batch`` and ``waves_per_eu``,
        and this kernel bakes both:

          * ``batch`` sizes the buffer-resource extents, so two specs differing only in
            it collide in a name-keyed cache and a B>1 launch is served the B=1 binary
            -- an out-of-bounds read.
          * ``waves_per_eu`` is emitted as the ``amdgpu-waves-per-eu`` kernel attribute
            and changes register allocation, so the two compile to different binaries.

        The gfx942-private fields append through :func:`_tuning_name_tags`, which is
        EMPTY at the shipped configuration -- so the default gfx942 name is
        byte-identical to the pre-subclass one and no golden moves -- and non-empty,
        uniquely, for any other configuration.

        The K row-group pad needs NO tag: it also changes the emitted K_lds layout and
        do_qk addressing, but it lives in the shared ``lds_k_group_pad`` field and the
        base ``kernel_name()`` already emits a ``kpad{N}`` token for it on the packed
        path. A second gfx942 tag would restate the same fact under another name.

        Under ``runtime_shape`` the ``b{batch}`` token drops, for the same reason the
        base drops ``sq``/``sk`` and ``hq``/``kv`` (via :meth:`_shape_name_parts` /
        :meth:`_head_name_parts`): batch is a kernel param there, it sizes nothing
        baked, and keeping it would give two specs that share ONE cache key two
        DIFFERENT symbol names -- which the ``assert art.kernel_name ==
        spec.kernel_name()`` in :func:`run_attention_dense_torch` would then trip on
        the second shape served from the cache.
        """
        batch_part = "" if self.runtime_shape else f"_b{self.batch}"
        return (
            f"{super().kernel_name()}_gfx942{batch_part}"
            f"_wpe{self.resolved_waves_per_eu()}{_tuning_name_tags(self)}"
        )


def _as_gfx942_spec(spec: AttentionDenseSpec) -> Gfx942AttentionDenseSpec:
    """Promote a shared spec to the gfx942 spec at its shipped defaults.

    Every public entry point below runs its argument through this, so a caller that
    only knows the arch-neutral :class:`AttentionDenseSpec` -- dispatch, which never
    sets a gfx942-private knob, and every shape-only helper -- keeps working
    unchanged, while a caller that wants a private lever passes the subclass. The
    promotion is value-preserving and the added fields land on their defaults, so the
    promoted spec's :meth:`~Gfx942AttentionDenseSpec.kernel_name` is byte-identical to
    the name the pre-subclass code produced for the same shape.

    Copies the BASE field set specifically (not ``spec``'s own), so promoting is
    total: it cannot fail on a subclass that carries fields this one does not know.
    """
    if isinstance(spec, Gfx942AttentionDenseSpec):
        return spec
    if type(spec) is not AttentionDenseSpec:
        raise TypeError(
            "cannot promote a concrete architecture spec to gfx942: "
            f"{type(spec).__name__}"
        )
    return Gfx942AttentionDenseSpec(
        **{f.name: getattr(spec, f.name) for f in _dataclass_fields(AttentionDenseSpec)}
    )


__all__ = [
    "AttentionDenseSpec",
    "Gfx942AttentionDenseSpec",
    "supports_attention_dense",
    "bf16_cfvst_supported",
    "build_attention_dense",
    "attention_dense_grid",
    "attention_dense_block",
    "attention_dense_signature",
    "gfx942_kernel_name",
    "run_attention_dense_torch",
]


def _tuning_name_tags(spec: "Gfx942AttentionDenseSpec") -> str:
    """Name suffix for every gfx942-private field that is NOT at its shipped value.

    Empty string at the shipped configuration -- which is the property that keeps the
    goldens and every cached kernel name byte-identical -- and non-empty, uniquely, for
    any other configuration. The tri-state fields are compared against what the POLICY
    would have produced, not against the raw ``None``, so ``use_cfvst=True`` on the
    config the policy already turns on is (correctly) the same kernel and the same
    name.

    Deliberately tagged on "differs from the default", not on "is live in this
    build": at D64 a non-default ``lds_row_pad`` emits identical IR (D64 takes the
    per-GROUP pad instead), so this over-tags there and costs one duplicate compile.
    That direction is free; the other direction -- under-tagging a knob that DOES
    move IR -- silently serves a stale binary out of ``_DENSE_LAUNCHER_CACHE``, which
    is the bug this whole tagging scheme exists to prevent.

    ``waves_per_eu`` needs no tag here: it is a base field and its value is already
    emitted as the ``wpe{N}`` token by
    :meth:`Gfx942AttentionDenseSpec.kernel_name` itself.
    """
    parts: list[str] = []
    # Both IR-changing controls have independent cache/name tags. Neither
    # changes the default symbol, which has one K buffer and no V prefetch.
    if _use_kdb(spec):
        parts.append("kdb1")
    if _use_vpf(spec):
        parts.append("vpf1")
    if spec.lds_row_pad != _DEFAULT_LDS_ROW_PAD:
        parts.append(f"krowpad{spec.lds_row_pad}")
    vp = spec.resolved_v_row_pad()
    if vp != _v_row_pad(spec.head_size, spec.dtype, spec.block_n):
        parts.append(f"vrowpad{vp}")
    cfvst = spec.resolved_use_cfvst()
    if cfvst != _use_cfvst(spec.head_size, spec.dtype):
        parts.append("cfvst1" if cfvst else "cfvst0")
    swz = spec.resolved_use_v_swizzle()
    if swz != (spec.resolved_use_cfvst() and _use_cfvst(spec.head_size, spec.dtype)):
        parts.append("vswz1" if swz else "vswz0")
    e2f = spec.resolved_use_exp2_fast()
    if e2f != _use_exp2_fast(spec.head_size, spec.dtype, spec.persistent):
        parts.append("e2f1" if e2f else "e2f0")
    if spec.iglp != _DEFAULT_IGLP:
        parts.append("iglp1" if spec.iglp else "iglp0")
    if spec.iglp_mode != _DEFAULT_IGLP_MODE:
        parts.append(f"iglpm{spec.iglp_mode}")
    if spec.pv_loop_order != _DEFAULT_PV_LOOP_ORDER:
        parts.append("pvkmaj")
    if spec.o_store_width != _DEFAULT_O_STORE_WIDTH:
        parts.append(f"osw{spec.o_store_width}")
    if spec.pv_priority:
        parts.append(f"prio{spec.pv_priority}")
    if spec.pv_sched_fence_mask is not None:
        parts.append(f"fence{spec.pv_sched_fence_mask:x}")
    if spec.causal_diag_split:
        parts.append("dsplit")
    if spec.window_wave_skip:
        parts.append("wskip")
    if spec.emit_lse:
        parts.append("lse")
    if spec.k_lds_swizzle:
        parts.append("kswz")
    if spec.k_read_b128:
        parts.append("kb128")
    if spec.tile_interleave:
        parts.append("tilv")
    if spec.tilv_v_write_qk:
        parts.append("vwq")
    if spec.tilv_rescale_skip:
        parts.append("askip")
    if spec.qk_lookahead:
        parts.append("qkla")
    if spec.qkla_pack_ahead:
        parts.append(f"pa{spec.qkla_pack_ahead}")
    if spec.qkla_rescale_plain:
        parts.append("rsp")
    if spec.qkla_read_ahead:
        parts.append(f"ra{spec.qkla_read_ahead}")
    if spec.qkla_qk_tail:
        parts.append(f"qt{spec.qkla_qk_tail}")
    if spec.qkla_rhythm_b or spec.qkla_rhythm_pv:
        parts.append(f"rh{spec.qkla_rhythm_b}x{spec.qkla_rhythm_pv}")
    if spec.qkla_pv_spread:
        parts.append("pvs")
    if spec.qkla_pv_fence:
        parts.append(f"pvf{spec.qkla_pv_fence}")
    if spec.qkla_chunked:
        parts.append("chk")
    if spec.qkla_loads_b:
        parts.append("ldb")
    if spec.qkla_chunk_fence:
        parts.append(f"cf{spec.qkla_chunk_fence}")
    if spec.qkla_chunk_rev:
        parts.append(f"cr{spec.qkla_chunk_rev}")
    if spec.q_lds_stage:
        parts.append("qls")
    if spec.o_lds_stage:
        parts.append("ols")
    if spec.qkla_vdb:
        parts.append(f"vdb{spec.qkla_vdb}")
    if spec.qkla_split_wait:
        parts.append("sw")
    if spec.q_full_line:
        parts.append("qfl")
    if spec.q_batch_wait:
        parts.append("qbw")
    if spec.kv_segments != 1:
        parts.append(f"kvs{spec.kv_segments}")
    if spec.xcdlb:
        parts.append("xcdlb")
    if spec.qkla_sub_pin:
        parts.append("subpin")
    if spec.qkla_lhalf:
        parts.append("lhalf")
    if spec.qkla_lds_after_mfma:
        parts.append("lam")
    if spec.qkla_vld_start or spec.qkla_vld_stride != 1:
        parts.append(f"vld{spec.qkla_vld_start}s{spec.qkla_vld_stride}")
    if spec.qkla_prio_bal:
        parts.append(f"pb{spec.qkla_prio_bal}")
    if spec.qkla_max_ahead:
        parts.append("maxa")
    if spec.qkla_even:
        parts.append(f"even{spec.qkla_even}")
    if spec.qkla_even_mix:
        parts.append("mix")
    if spec.wg_timing:
        parts.append("wgt")
    return "".join(f"_{p}" for p in parts)


def gfx942_kernel_name(spec: AttentionDenseSpec) -> str:
    """The gfx942 kernel symbol for ``spec``.

    A thin shim over :meth:`Gfx942AttentionDenseSpec.kernel_name`, which owns the
    derivation; this exists so a caller holding a plain
    :class:`AttentionDenseSpec` can ask for the gfx942 symbol without knowing about
    the subclass (dispatch's ``kernel_name_override`` and the harnesses do exactly
    that). Promotion lands the private fields on their shipped defaults, so the
    result is byte-identical to the pre-subclass name for the same shape.
    """
    return _as_gfx942_spec(spec).kernel_name()


# In-scope for the gfx942 dense-prefill port. supports_attention_dense rejects
# everything outside this AND every mode the builder cannot emit, so a True result
# from it means build_attention_dense succeeds. That equivalence is what keeps the
# dispatch arm from selecting a spec it cannot build.
_SUPPORTED_DTYPES = ("bf16", "fp16")
_SUPPORTED_HEAD_SIZES = (64, 128)

# Elements moved into LDS by ONE async-DMA instruction: 64 lanes x dwords=1 (4 B)
# / 2 B per element. wave64 and a 2-byte dtype are the only cases this kernel emits
# (supports_attention_dense gates dtype to bf16/fp16); an fp8 extension must
# re-derive this or the row stride and the LDS budget go wrong together.
_DMA_ELEMS_PER_INSTR = 64 * 4 // 2  # = 128


def _rows_per_instr(head_size: int) -> int:
    """K/V rows packed into one async-DMA instruction (1 at D128, 2 at D64).

    Single definition of the packing rule that BOTH the LDS row stride and the
    loader's addressing derive from; if those two ever disagree, the padded stride
    and the DMA disagree about row adjacency, which corrupts silently.
    """
    return _DMA_ELEMS_PER_INSTR // head_size


def _use_exp2_fast(head_size: int, dtype: str, persistent: bool) -> bool:
    """Whether softmax uses ``exp2_fast`` (one v_exp_f32, no range-reduction guard).

    Enabled for all configs EXCEPT bf16 head_dim=128 on the DEFAULT grid (see the
    guard below). exp2_fast is a strict VALU reduction and the dominant P2 lever on
    the (post-P1) VALU-bound path, and is always numerically safe here -- both softmax
    args (alpha's m_i - m_new and p's s - m_new) are <= 0, exactly exp2_fast's
    precondition, independent of head_size and dtype. So this is a pure perf gate,
    never a correctness one.

    Default-grid bf16 D128 guard. exp2_fast is a net win only where the kernel is
    VALU-bound, and on this config the grid is what decides that. The P4 persistent
    grid keeps one CTA resident over many work items, so the softmax VALU stays the
    critical path and exp2_fast wins there uniformly. On the default grid (one CTA per
    work item) bf16 D128 is occupancy/latency-bound instead, and exp2_fast's register/
    schedule shift pays for itself nowhere on that path -- so it reverts to plain
    exp2, its original perf. fp16 D128 -- byte-identical across the exp2_fast boundary
    -- is unaffected on either grid, which pins the regression to this kernel.
    head_dim != 128 and fp16 are unaffected.

    This predicate reads only compile-time config, never the problem shape: an earlier
    revision cut on ``seqlen < 4096`` instead, which made the emitted body a function
    of ``seqlen_q`` and so collided with the runtime-shape cache identity. Re-measuring
    across both grids showed the seqlen term was a proxy for which grid the dispatch
    layer picked at that shape, not an effect of the sequence length: there is no
    seqlen at which exp2_fast wins on the default grid, nor one at which it loses on
    the persistent grid. Keep any future term compile-time for the same reason.

    Scope: measured on gfx942 against the develop dense builder of the time (causal,
    square Sq==Skv), both grids forced, with fp16-D128 / bf16-D64 / fp16-D64 controls.
    Full-mask square shapes take the same cut on the same mechanism (identical
    kernel/softmax), not separately measured. The gfx950 dense kernel makes an
    independent exp2_fast decision (its own builder) and is not covered here. This is
    a gfx942 / ROCm-7.2.2 sweep snapshot taken BEFORE the softmax scale moved onto the
    fp32 scores (one extra multiply per score; bf16-D128 default-grid VGPR use rose),
    and it was not re-measured after that change -- re-measure via the dense sweep
    harness on a toolchain or kernel change rather than treating it as fixed.
    """
    if dtype == "bf16" and head_size == 128 and not persistent:
        return False
    return True


def _use_cfvst(head_size: int, dtype: str) -> bool:
    """Whether to feed V through the P1 conflict-free perm_b32 store-path transpose.

    D128 only (both dtypes). Measured on gfx942:
      * D64 (any dtype): cfvst REGRESSES it -- D64 is VGPR-bound, not LDS-bound
        (plan §6.1), so the register round-trip + perm temps raise pressure without
        relieving the actual bottleneck.
      * D128 bf16: once the P cast is the NaN-free pair pack (:func:`_pack_p_bf16`),
        cfvst fits the waves-per-eu=2 cap without spilling (about 220 VGPR on
        LLVM 23), and replaces the naive path's 128 element-wise ds_read_u16 per tile
        with 32 ds_read_b64, as on fp16.
    Gate on rows-per-DMA == 1 (D128) so the LDS budget and the body agree.
    """
    del dtype  # both 2-byte dtypes take the same transposed-V path
    return _rows_per_instr(head_size) == 1


def _pack_p_bf16(b: IRBuilder, elems):
    """Round four f32 probabilities to bf16 (nearest-even) and pack them as the PV
    B-operand ``<4 x bf16>``.

    P = exp2(s - m) is finite and in [0, 1], so the NaN-preserving select of a
    generic f32 -> bf16 conversion is dead weight: each element is rounded with
    ``bits + 0x7FFF + ((bits >> 16) & 1)``, and one ``perm_b32`` takes the two high
    halves of a pair. Bit-identical to the generic conversion for every finite
    input; gfx942 has no packed f32 -> bf16 convert instruction.
    """
    words = []
    for h in range(0, len(elems), 2):
        rounded = []
        for e in elems[h : h + 2]:
            bits = b.bitcast(e, I32)
            lsb = b.land(b.lshr(bits, b.const_i32(16)), b.const_i32(1))
            rounded.append(b.add(b.add(bits, b.const_i32(0x7FFF)), lsb))
        # Bytes 2-3 of each operand: low half <- first element, high <- second.
        words.append(b.perm_b32(rounded[0], rounded[1], b.const_i32(0x03020706)))
    return b.bitcast(b.vec_pack(words, I32), VectorType(BF16, len(elems)))


def _pack_p_bf16_word(b: IRBuilder, e0, e1):
    """One i32 word of :func:`_pack_p_bf16` (elements e0, e1), same ops."""
    rounded = []
    for e in (e0, e1):
        bits = b.bitcast(e, I32)
        lsb = b.land(b.lshr(bits, b.const_i32(16)), b.const_i32(1))
        rounded.append(b.add(b.add(bits, b.const_i32(0x7FFF)), lsb))
    return b.perm_b32(rounded[0], rounded[1], b.const_i32(0x03020706))


def _use_kdb(spec: "Gfx942AttentionDenseSpec") -> bool:
    """Whether K uses two LDS slots rather than one (opt-in, off by default)."""
    return spec.lds_num_buffers == 2


def _use_vpf(spec: "Gfx942AttentionDenseSpec") -> bool:
    """Whether to issue next-tile V VMEM loads before the current PV (opt-in)."""
    return spec.v_reg_prefetch


def bf16_cfvst_supported(spec: Gfx942AttentionDenseSpec) -> bool:
    """Whether the KDB/VPF pipeline is implemented for this problem's BF16 CFVST
    body: D128 top-left causal attention without a sliding window or sinks, with or
    without ``causal_diag_split``.

    The single definition of the KDB/VPF cohort, gated by
    :func:`supports_attention_dense`. The pipeline additionally needs the persistent
    grid, resolved CFVST and ``block_n=64``.
    """
    return (
        spec.dtype == "bf16"
        and spec.head_size == 128
        and spec.causal
        and not spec.causal_bottom_right
        and spec.sliding_window == 0
        and not spec.use_sinks
    )


def _v_swizzle_width(block_n: int) -> int:
    """V^T row width (elements) the XOR bank-conflict swizzle needs: the smallest pow2
    >= max(64, block_n). SINGLE source of truth -- the derived pad (:func:`_v_row_pad`),
    the emitted swizzle mask (``V_LDROW // 4 - 1``), and the swizzle-engages test all
    take the width from here, so they cannot drift. 64 (not 128) is the minimum that
    spreads all 32 banks: only bits 2..5 of the multiple-of-4 column reach the bank
    field, so a 4-bit key saturates it and a 5th bit only doubles the row for no gain.

    Chosen over the sibling ``attention_tiled_2d`` V^T scheme, which spreads the token
    axis with an 8-element ROTATION (``_v_t_pad``, ~2 KB at D128) behind an explicit
    ``SWIZZLE_VLDS`` knob that auto-disables on an LDS-headroom check. The pow2+XOR form
    here is a bijection that leaves the read conflict-free (1-way) rather than merely
    reduced, at the cost of a pow2 row; the auto-disable is not replicated because
    :func:`supports_attention_dense` rejects an over-budget spec upstream instead of
    silently dropping the swizzle.
    """
    return max(64, 1 << (block_n - 1).bit_length())


def _use_v_swizzle(head_size: int, dtype: str) -> bool:
    """Policy for the V^T bank-conflict swizzle: rides the :func:`_use_cfvst` gate,
    since only the transposed-V (D128) store has a V_lds to swizzle."""
    return _use_cfvst(head_size, dtype)


def _v_row_pad(head_size: int, dtype: str, block_n: int) -> int:
    """V^T row (token-axis) pad, in elements, on the cfvst path (D128); 8 (no
    swizzle, original layout) everywhere else. On the cfvst path the row is widened to
    :func:`_v_swizzle_width` so the XOR swizzle (``col' = key ^ ((dim & (V_LDROW//4-1))
    << 2)``) stays in-bounds and engages at EVERY tile width (pad 64->0, 32->32,
    128->0). V footprint ``head_size * width * 2`` scales with the tile (16 KB at
    D128/block_n=64, not block_n-invariant); total LDS 33 KB -- below develop's +8-pad
    layout (35 KB) and the old fixed-width-128 row (49 KB), but still ~1 KB over the
    32 KB 2-WG/CU ceiling, so that door needs the register floor cut, not this widening.
    Whether the swizzle is actually EMITTED is a
    separate, independently-sweepable decision (:meth:`resolved_use_v_swizzle`), so the
    pad value and the swizzle no longer silently toggle each other.
    """
    if not _use_cfvst(head_size, dtype):
        return 8
    return _v_swizzle_width(block_n) - block_n


def _tuned_waves_per_eu(head_size: int, dtype: str) -> int:
    """Tuned ``amdgpu-waves-per-eu`` per config (P3 occupancy).

    Default 2 (one WG/CU = 2 waves/SIMD at this kernel's ~178-220 VGPR on ROCm 10 /
    LLVM 23). The one override is **bf16 D64 -> 4**, which forces the allocator from
    215 VGPR down to ~117 (0 spill). D64's LDS is only 16 KB (allows 4 WGs), so the
    smaller per-wave budget lets a SECOND WG co-reside (2 WG/CU), and the bf16 ``.1k``
    schedule is serialized enough at wpe=2 that its HBM latency is EXPOSED -- the
    extra resident WG hides it. Measured on MI300X it is a large win at S512 and S8192
    and neutral at S256 and S2048, so there is no seqlen gate.

    Why not the other configs (all measured, all kept at 2):
      * fp16 D64: wpe=3 already reaches 2 WG/CU at 116 VGPR / 0 spill, but its wpe=2
        schedule ALREADY hides the latency via ILP (the 214 VGPR is scheduler headroom
        spent on in-flight LDS reads, plan §6.1) -- halving per-wave registers loses
        more ILP than the 2nd WG buys (S2048 81.5 -> 79.8, slightly negative).
      * D128 (either dtype): LDS-bound at ~33-34 KB, so 2 x > 64 KB -- no wpe value
        reaches a 2nd WG/CU. Occupancy there is an LDS-footprint problem (P3 K/V-pad
        or single-buffer work), not a waves-per-eu knob.

    Consumed by the gfx942 dispatch base spec (``_base_spec``) so the kernel_name
    ``wpe{N}`` tag and the emitted ``amdgpu-waves-per-eu`` attribute always agree.
    """
    if dtype == "bf16" and head_size == 64:
        return 4
    return 2


def _lds_row_stride(spec: "Gfx942AttentionDenseSpec") -> int:
    """K_lds / V_lds row stride in ELEMENTS for ``spec``'s head size.

    D128 packs ONE row per async-DMA instruction (64 lanes x 2 elems = 128 elems =
    one D128 row), so the row can carry the bank-conflict pad
    (``spec.lds_row_pad``). D64 packs TWO rows per instruction, which requires a
    contiguous UNPADDED stride -- a padded row would not be adjacent to the next one,
    so the single instruction could not cover both. D64 therefore pads between
    row-GROUPS instead (``spec.lds_k_group_pad`` / :func:`_k_group_stride`), and
    ``spec.lds_row_pad`` is inert there.

    Shared by the builder and :func:`supports_attention_dense` so the budget check
    cannot drift from the actual allocation.
    """
    if _rows_per_instr(spec.head_size) == 1:
        return spec.head_size + spec.lds_row_pad
    return spec.head_size


def _k_group_pad_active(spec: AttentionDenseSpec) -> bool:
    """Whether the D64 K-LDS row-group bank-conflict pad is EMITTED for ``spec``.

    ADOPTED, not pending. There is NO gfx942-private switch: the pad amount is the
    SHARED ``spec.lds_k_group_pad``, which gfx950 owns and gfx942 reuses verbatim.
    Both arches pad between async-DMA row-GROUPS by the same formula
    (``rows_per_instr * head_size + pad``, see :func:`_k_group_stride`) for the same
    reason, so this is one lever with one field rather than a per-arch duplicate.
    gfx942 derived the pad from a measured ~2x ON/OFF ablation, gfx950 from a
    bank/phase model, and both landed on the same 8-element default.

    Active only on the PACKED path (``rows_per_instr > 1``, i.e. D64 here): D128
    already carries a per-row K pad via :func:`_lds_row_stride`, so it must never
    be re-padded. ``lds_k_group_pad=0`` reproduces the unpadded layout for A/B --
    which is also how the pad's ~2x is priced, no module-level probe constant needed.

    What the pad does
    -----------------
    With a non-zero pad the D64 ``K_lds`` takes a *row-group boundary* pad, so the
    ``do_qk`` K reads drop from a full 32-way bank conflict to 4-way (matching D128)
    WITHOUT losing async-DMA efficiency: the contiguous 2-rows-per-instruction DMA
    still writes one whole group per instruction, and the pad sits at the group
    boundary it never touches. This is the "swizzled LDS layout" arm of the
    hypothesis, and it is strictly better than the other arm that was considered --
    ``ROWS_PER_INSTR=1`` plus half the DMA efficiency.

    MECHANISM (corrected -- a proven negative, do not re-derive)
    -----------------------------------------------------------
    The static codegen A/B is IDENTICAL with the pad on and off: VGPR count, the
    LDS-instruction stream and the ``s_nop`` padding are all unchanged. The
    once-hypothesised "the pad frees registers -> more ILP -> compounding win"
    mechanism is therefore DEBUNKED. The win is purely eliminated bank-conflict
    REPLAY -- the same ``ds_read`` replayed ~8x at 32-way instead of ~4-way -- which
    is a runtime effect and is invisible to any static probe. Re-running a static
    A/B to "explain" this lever will find nothing; only a timed run prices it.

    Takes a plain :class:`AttentionDenseSpec` rather than the gfx942 subclass because
    the field it reads is SHARED (``spec.lds_k_group_pad``), not a gfx942-private
    knob -- precisely because gfx950 emits the identical layout from the identical
    formula. The narrower annotation is the signal for which of the two pads a
    reader is looking at.
    """
    return spec.lds_k_group_pad > 0 and _rows_per_instr(spec.head_size) > 1


def _k_group_stride(spec: AttentionDenseSpec) -> int:
    """K_lds physical group stride in ELEMENTS when the row-group pad is active.

    A group is ROWS_PER_INSTR (=2 at D64) contiguous rows written by one async-DMA
    instruction; the pad sits at the group boundary (never DMA-touched). Identical
    to the gfx950 builder's ``LDROW = K_GROUP * D + spec.lds_k_group_pad``. Reads only
    SHARED fields, for the reason given in :func:`_k_group_pad_active`."""
    return _rows_per_instr(spec.head_size) * spec.head_size + spec.lds_k_group_pad


def _lds_bytes(spec: "Gfx942AttentionDenseSpec") -> int:
    """LDS footprint: one or two K buffers and one V buffer.

    K keeps the natural ``[token, dim]`` layout with :func:`_lds_row_stride` (async
    DMA target). V is TRANSPOSED to ``[dim, token]`` for the P1 conflict-free store, so
    its footprint is ``D * (block_n + resolved_v_row_pad)`` rather than
    ``block_n * row_stride``. 2 bytes/element is exact for every dtype in
    ``_SUPPORTED_DTYPES`` (bf16/fp16) and
    must be revisited if a narrower or wider element type is added. Shared with the
    budget check in :func:`supports_attention_dense` so the two cannot drift.

    When :func:`_k_group_pad_active` (the D64 bank-conflict pad), K instead takes the
    2-row-group layout ``K_lds[1, block_n // rows_per_instr, _k_group_stride]`` --
    +``spec.lds_k_group_pad`` elements per group (D128 and the kpad-off D64 path are
    unchanged).

    ``resolved_v_row_pad`` is read HERE and in the builder's ``V_lds`` allocation; both
    sites go through the same resolver, so the budget cannot silently under-count the
    allocation. ``v_reg_prefetch`` changes VGPR scheduling only, not LDS size.
    """
    if _k_group_pad_active(spec):
        rpi = _rows_per_instr(spec.head_size)
        k_bytes = (spec.block_n // rpi) * _k_group_stride(spec) * 2
    else:
        k_bytes = spec.block_n * _lds_row_stride(spec) * 2
    k_bytes *= spec.lds_num_buffers  # Only K is double-buffered.
    if spec.resolved_use_cfvst():
        # V transposed to [dim, token+pad] for the conflict-free store (D128).
        v_bytes = spec.head_size * (spec.block_n + spec.resolved_v_row_pad()) * 2
        if getattr(spec, "qkla_vdb", False):
            v_bytes *= 2
    else:
        # V keeps the natural [token, dim] async-DMA layout (D64, naive read).
        v_bytes = spec.block_n * _lds_row_stride(spec) * 2
    return k_bytes + v_bytes


def supports_attention_dense(
    spec: AttentionDenseSpec,
    *,
    arch: str = "gfx942",
) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for one gfx942 dense-prefill config.

    This is the SINGLE source of truth for what :func:`build_attention_dense` can
    emit: every rejection the builder would make is made here first, so a True
    result implies the build succeeds. Dispatch gates on this, which lets an
    out-of-scope request fall through to another candidate instead of selecting
    this arm and failing at build time.

    That contract covers the gfx942-private fields too, not just the shared shape:
    an illegal :class:`Gfx942AttentionDenseSpec` is rejected HERE, with a structured
    reason, rather than only blowing up inside the builder (or -- worse -- building a
    config that is known to spill or to read past its tile).

    Accepts a plain :class:`AttentionDenseSpec`, which is promoted to the gfx942
    subclass at its shipped defaults; that keeps a caller that only knows the shared
    spec (dispatch, the descriptor path) working unchanged.

    In scope for this port: gfx942, bf16/fp16, D64/D128, MHA/GQA including
    non-power-of-2 groups, causal or full, the default grid AND the P4 persistent
    grid-stride variant, ``block_n`` dividing the ``block_m`` query tile, within the
    LDS budget and 32-bit addressing, sliding-window (KV-loop prune + window mask), and
    attention sinks (alone or with sliding-window). varlen / ragged are later
    follow-ups (rejected below).
    """
    if arch != "gfx942":
        return False, f"kernels.gfx942.attention_dense is gfx942-only (got {arch})"
    # Type FIRST: every check below dereferences a spec attribute, so a non-spec
    # argument would raise AttributeError out of this (bool, str) API instead of
    # returning the structured rejection the contract promises.
    if not isinstance(spec, AttentionDenseSpec):
        return False, f"spec must be an AttentionDenseSpec, got {type(spec).__name__}"
    # Bottom-right causal shifts the diagonal by seqlen_kv - seqlen_q (a runtime value
    # on the runtime-shape path). Only the plain causal band has that shifted form:
    # the window prune is anchored on the top-left diagonal.
    if spec.causal_bottom_right and spec.sliding_window:
        return False, (
            "gfx942 attention_dense: causal_bottom_right is not supported with "
            "sliding_window>0"
        )
    try:
        spec = _as_gfx942_spec(spec)
    except TypeError as exc:
        return False, str(exc)
    if spec.dtype not in _SUPPORTED_DTYPES:
        return (
            False,
            f"gfx942 attention_dense supports {_SUPPORTED_DTYPES}, got {spec.dtype}",
        )
    if spec.head_size not in _SUPPORTED_HEAD_SIZES:
        return False, (
            f"gfx942 attention_dense scope is D{list(_SUPPORTED_HEAD_SIZES)} "
            f"(D256 is served by its own wide-atom candidates), got D{spec.head_size}"
        )
    # --- Shared preflight: dataclass re-validation, positive extents, block_n
    # dividing the query tile, and the 32-bit extent bounds. All four are properties
    # of the base spec with the same verdict for every dense body, so they live in
    # kernels.common next to the spec rather than being replicated per arch. The
    # gfx942-private knobs and the LDS budget are NOT in there -- they need this
    # body's tile math, and are checked below.
    ok, why = check_dense_spec_preflight(spec)
    if not ok:
        return False, why

    # --- Mode scope. The body implements the default-grid AND the P4 persistent
    # grid-stride variant, both uniform dense self-attention. Checked HERE and not
    # only in the builder so that support() and build() agree on exactly one set of
    # specs. Persistent needs Sq % block_m == 0 (the grid-stride work count
    # W = (Sq // block_m) * Hq * B floors); ragged (the ceil case) is rejected
    # just below, and the seqlen_q check under "Tuning struct" then makes the floor
    # exact for every spec that reaches the persistent builder.
    if spec.varlen:
        return False, "gfx942 attention_dense: varlen not yet supported"
    if spec.ragged:
        # The factory sets ragged for any graph whose lengths leave a partial tile;
        # name the length and the tile rather than calling an ordinary graph ragged.
        return False, (
            f"gfx942 attention_dense needs whole tiles: {partial_tile_reason(spec)} "
            f"(the partial-tile path is not yet supported)"
        )
    # sliding_window is supported (KV-loop prune + window mask); the shared spec
    # __post_init__ re-run above enforces its constraints (W % block_n, causal).
    # use_sinks is supported here (m/l seed per work item). The shared __post_init__
    # still rejects it combined with paged or varlen.

    # --- gfx942-private sweep knobs. Validated here rather than only in the builder
    # because the module contract is support() => build(): a knob that only the
    # builder rejects is the same dispatch fall-through hole every other check in
    # this function exists to close. Every branch below is inert at the shipped
    # defaults, so the shipped behaviour is unchanged.
    #
    # block_m: the wave count is block_m // 32, a FLOOR. At a non-multiple of 32 the
    # emitted waves under-cover the query tile and the top block_m % 32 rows of every
    # block are never computed and never written -- a silent wrong answer with no
    # fault (the same invariant the module-scope _BLOCK_M guard binds for the
    # default). block_m // 32 * 64 is the CTA thread count, capped at 1024 by the
    # hardware workgroup limit.
    if spec.block_m <= 0 or spec.block_m % 32 != 0:
        return False, (
            f"block_m must be a positive multiple of 32, got "
            f"{spec.block_m}: the body derives WAVES = block_m // 32, which floors, "
            f"so a non-multiple leaves the top rows of every query block uncovered by "
            f"any wave -- never computed and never written"
        )
    if spec.block_m // 32 * 64 > 1024:
        return False, (
            f"block_m={spec.block_m} needs a "
            f"{spec.block_m // 32 * 64}-thread CTA, past the 1024-thread workgroup "
            f"maximum"
        )
    # Q is read with a plain global_load_vN (no buffer bound), and the epilogue stores
    # the same rows, so a query tile that runs past seqlen_q reads and writes out of
    # bounds. Implied by the dataclass (seqlen_q % 256 == 0) at the default block_m.
    if spec.seqlen_q % spec.block_m != 0:
        return False, (
            f"seqlen_q={spec.seqlen_q} must be a multiple of "
            f"block_m={spec.block_m}: the last query tile would otherwise "
            f"run past seqlen_q, and Q/O are addressed without a bounds check"
        )
    # Pads, in elements. The QK / cfvst-V reads are smem_load_vN(n=4) = 8-byte
    # payloads, which the lowering stamps `align 8` on unconditionally; the column
    # index inside a row is always a multiple of 4 elements, so keeping the PITCH a
    # multiple of 4 elements is what keeps that alignment contract true. A pitch that
    # breaks it keeps the ds_read and silently returns the wrong data or faults --
    # the same failure mode the spec's lds_k_group_pad % 8 check guards for ds_read_b128.
    for _pad_name, _pad in (
        ("lds_row_pad", spec.lds_row_pad),
        ("v_row_pad", spec.resolved_v_row_pad()),
    ):
        if _pad < 0 or _pad % 4 != 0:
            return False, (
                f"{_pad_name} must be a non-negative multiple of 4 elements "
                f"(8 bytes) so the LDS pitch stays aligned for the n=4 ds_read, got "
                f"{_pad}"
            )
    _wpe = spec.resolved_waves_per_eu()
    if _wpe <= 0:
        return False, f"waves_per_eu must be positive, got {_wpe}"
    # cfvst forced ON where the policy says OFF. The case the policy excludes is
    # known-bad, not merely untuned: D64 (rows-per-DMA > 1) is VGPR-bound, so the
    # register round-trip regresses it AND the transposed store no longer matches the
    # packed async-DMA layout the naive path relies on. Turning cfvst OFF is always
    # legal (it is the naive path), so only the ON direction is gated. See _use_cfvst.
    if spec.resolved_use_cfvst() and not _use_cfvst(spec.head_size, spec.dtype):
        return False, (
            f"use_cfvst=True is rejected at D{spec.head_size}/{spec.dtype}: "
            f"the conflict-free-V store is measured-negative there (D64 is VGPR-bound "
            f"and regresses), and it is only tile-exact on the rows-per-DMA==1 path "
            f"-- see _use_cfvst"
        )
    # use_v_swizzle forced ON where there is no transposed-V buffer to swizzle. Only the
    # cfvst path builds V_lds[dim, token]; on the naive path the flag is a no-op, so
    # reject an explicit True there rather than silently ignoring it.
    if spec.use_v_swizzle is True and not spec.resolved_use_cfvst():
        return False, (
            f"use_v_swizzle=True is rejected at D{spec.head_size}/{spec.dtype}: "
            f"the XOR bank-conflict swizzle only exists on the conflict-free-V "
            f"(D128) path -- there is no transposed V_lds to swizzle otherwise"
        )
    # Performance-only codegen knobs: ranges, plus the one scheduling conflict.
    if spec.lds_num_buffers not in (1, 2):
        return False, "lds_num_buffers must be 1 or 2"
    # These are independent opt-in scheduling levers, but both currently share
    # the same validated BF16 D128 CFVST scope. In particular, block_n=64 only:
    # the other tile sizes have not been validated for these pipeline modes.
    if (_use_kdb(spec) or _use_vpf(spec)) and not (
        spec.persistent
        and bf16_cfvst_supported(spec)
        and spec.resolved_use_cfvst()
        and spec.block_n == 64
    ):
        return False, (
            "NOT_YET_IMPLEMENTED: lds_num_buffers=2 or v_reg_prefetch=True requires "
            "persistent=True, bf16 D128 top-left causal CFVST at block_n=64, without "
            "sliding_window or sinks"
        )
    if spec.iglp_mode not in (0, 1):
        return False, f"iglp_mode must be 0 or 1, got {spec.iglp_mode}"
    if spec.iglp_mode != _DEFAULT_IGLP_MODE and not spec.iglp:
        return False, "iglp_mode only applies with iglp=True"
    if spec.pv_loop_order not in _PV_LOOP_ORDERS:
        return False, (
            f"pv_loop_order must be one of {_PV_LOOP_ORDERS}, "
            f"got {spec.pv_loop_order!r}"
        )
    if spec.o_store_width not in _O_STORE_WIDTHS:
        return False, (
            f"o_store_width must be one of {_O_STORE_WIDTHS}, "
            f"got {spec.o_store_width}"
        )
    if spec.o_store_width != _DEFAULT_O_STORE_WIDTH and spec.dtype == "fp16":
        return False, (
            f"o_store_width={spec.o_store_width} is bf16-only: narrower fp16 stores "
            f"are not bit-identical to the width-4 output"
        )
    if not 0 <= spec.pv_priority <= 3:
        return False, f"pv_priority must be in 0..3, got {spec.pv_priority}"
    if spec.pv_sched_fence_mask is not None:
        if not 0 <= spec.pv_sched_fence_mask <= _SCHED_BARRIER_MASK_MAX:
            return False, (
                f"pv_sched_fence_mask must be in 0..{_SCHED_BARRIER_MASK_MAX:#x}, "
                f"got {spec.pv_sched_fence_mask}"
            )
        if spec.iglp:
            return False, (
                "pv_sched_fence_mask and iglp are exclusive: iglp owns the loop "
                "schedule"
            )
    if spec.window_wave_skip and not (spec.causal and spec.sliding_window):
        return False, "window_wave_skip applies only to a causal sliding window"
    if spec.causal_diag_split and (not spec.causal or spec.sliding_window):
        return False, (
            "causal_diag_split applies only to causal attention without a sliding "
            "window"
        )
    if spec.k_read_b128 and not (
        spec.head_size == 128 and spec.resolved_use_cfvst() and spec.lds_row_pad % 8 == 0
    ):
        return False, (
            "k_read_b128 (prototype) needs D128 with the cfvst V path and a 16 B-aligned "
            "K row (lds_row_pad % 8 == 0)"
        )
    if spec.tile_interleave and not (
        _use_kdb(spec) and _use_vpf(spec) and spec.block_n == 64 and spec.head_size == 128
        and spec.resolved_use_cfvst() and not spec.iglp and spec.pv_sched_fence_mask is None
        and spec.pv_priority == 0
        and (not spec.k_read_b128 or spec.qk_lookahead) and not spec.sliding_window
        and not spec.window_wave_skip
    ):
        return False, (
            "tile_interleave (prototype) needs the BN64/D128 cfvst KDB+VPF pipeline "
            "without iglp, a PV fence/priority, a sliding window, or k_read_b128 "
            "outside qk_lookahead"
        )
    if (spec.tilv_v_write_qk or spec.tilv_rescale_skip) and not spec.tile_interleave:
        return False, "tilv_v_write_qk / tilv_rescale_skip need tile_interleave"
    if spec.qk_lookahead and not (spec.tilv_v_write_qk and not spec.tilv_rescale_skip):
        return False, "qk_lookahead (prototype) needs tilv_v_write_qk without tilv_rescale_skip"
    if (
        spec.qkla_pack_ahead or spec.qkla_rescale_plain or spec.qkla_read_ahead
        or spec.qkla_qk_tail or spec.qkla_rhythm_b or spec.qkla_rhythm_pv
        or spec.qkla_pv_spread or spec.qkla_pv_fence or spec.qkla_chunked
        or spec.qkla_loads_b
    ) and not spec.qk_lookahead:
        return False, (
            "qkla_pack_ahead / qkla_rescale_plain / qkla_read_ahead / qkla_qk_tail "
            "need qk_lookahead"
        )
    if spec.qkla_chunk_rev >= 3 and (not spec.qkla_chunked or spec.qkla_qk_tail):
        return False, "qkla_chunk_rev 3 needs qkla_chunked without qkla_qk_tail"
    if spec.q_lds_stage and not (
        (spec.qk_lookahead or spec.tilv_v_write_qk)
        and spec.resolved_use_cfvst()
        and (_use_kdb(spec) or _use_vpf(spec))
        and spec.head_size == 128
        and spec.kv_segments == 1
        and spec.head_size * (spec.block_n + spec.resolved_v_row_pad()) * 2
        >= (spec.block_m // 32) * 32 * 32 * 2
    ):
        return False, (
            "q_lds_stage needs the D128 CFVST pipeline with qk_lookahead or "
            "tilv_v_write_qk (V published after the prologue barrier) and a V "
            "buffer of at least 2 KB per wave"
        )
    if spec.qkla_vdb not in (0, 1, 2):
        return False, "qkla_vdb must be 0, 1 or 2"
    if spec.qkla_vdb and not (
        spec.qk_lookahead and spec.qkla_chunked and spec.qkla_chunk_rev != 3
        and spec.tilv_v_write_qk and spec.resolved_use_cfvst()
    ):
        return False, (
            "qkla_vdb needs qk_lookahead + qkla_chunked (not rev 3) + tilv_v_write_qk "
            "on the cfvst V path"
        )
    if spec.qkla_lds_after_mfma and not (
        spec.qk_lookahead and spec.qkla_chunked and spec.qkla_vdb == 2
        and spec.qkla_chunk_rev == 1 and not spec.qkla_qk_tail
    ):
        return False, "qkla_lds_after_mfma (prototype) needs qkla_chunked rev 1, qkla_vdb 2, no QK tail"
    if (spec.qkla_vld_start or spec.qkla_vld_stride != 1) and not (
        spec.qkla_chunked and spec.qkla_loads_b and spec.qkla_vld_stride >= 1
        and spec.qkla_vld_start >= 0
        and spec.qkla_vld_start + spec.qkla_vld_stride * 3 < 16
    ):
        return False, "qkla_vld_* (prototype): qkla_chunked + qkla_loads_b, last V load before chunk 16"
    if spec.qkla_prio_bal and not (
        spec.qkla_chunked and 1 <= spec.qkla_prio_bal <= 3 and spec.qkla_chunk_fence != 1
    ):
        return False, "qkla_prio_bal (prototype): 1..3, qkla_chunked, not with qkla_chunk_fence=1"
    if spec.qkla_max_ahead and not (
        spec.qk_lookahead and spec.qkla_chunked and spec.qkla_chunk_rev == 1
        and not spec.qkla_qk_tail and spec.qkla_vdb
    ):
        return False, "qkla_max_ahead (prototype) needs qkla_chunked rev 1, no QK tail, qkla_vdb"
    if spec.qkla_even and not (
        spec.qk_lookahead and spec.qkla_chunked and spec.qkla_chunk_rev == 1
        and not spec.qkla_qk_tail and spec.qkla_vdb == 2 and 8 <= spec.qkla_even <= 64
        and spec.dtype == "bf16" and not spec.qkla_max_ahead
    ):
        return False, "qkla_even (prototype): bf16, qkla_chunked rev 1, qkla_vdb 2, no QK tail, 8..64"
    if spec.qkla_even_mix and not spec.qkla_even:
        return False, "qkla_even_mix (prototype) needs qkla_even"
    if spec.qkla_lhalf and not (spec.qk_lookahead and spec.qkla_chunked):
        return False, "qkla_lhalf (prototype) needs qk_lookahead + qkla_chunked"
    if spec.qkla_split_wait and not (spec.qkla_vdb and spec.q_lds_stage):
        return False, "qkla_split_wait (prototype) needs qkla_vdb + q_lds_stage"
    if spec.q_full_line and not (spec.qkla_split_wait and spec.k_read_b128):
        return False, "q_full_line (prototype) needs qkla_split_wait + k_read_b128"
    if spec.q_batch_wait and not spec.q_lds_stage:
        return False, "q_batch_wait (prototype) needs q_lds_stage"
    if spec.o_lds_stage and not (
        spec.head_size == 128
        and not spec.emit_lse
        and spec.kv_segments == 1
        and not _k_group_pad_active(spec)
        and spec.block_n * _lds_row_stride(spec) * 2 * spec.lds_num_buffers
        >= (spec.block_m // 32) * 32 * 64 * 2
    ):
        return False, (
            "o_lds_stage needs D128, no emit_lse, no kv_segments, and a K buffer of "
            "at least 4 KB per wave"
        )
    if not 0 <= spec.qkla_qk_tail <= 16:
        return False, "qkla_qk_tail must be in [0, 16] (the K DMA follows the tail in PV)"
    if not 0 <= spec.qkla_pack_ahead < spec.block_n // 8:
        return False, f"qkla_pack_ahead must be in [0, {spec.block_n // 8 - 1}]"
    if spec.qkla_read_ahead < 0:
        return False, "qkla_read_ahead must be >= 0"
    if spec.k_lds_swizzle and not (spec.k_read_b128 and spec.lds_row_pad == 0):
        return False, (
            "k_lds_swizzle (prototype) needs k_read_b128 and lds_row_pad=0 "
            "(the swizzle replaces the K row pad)"
        )
    ok, why = check_kv_segments(spec)
    if not ok:
        return False, why
    if spec.xcdlb and not (
        spec.persistent
        and spec.num_persistent == 304
        and spec.num_query_heads == 32
        and spec.num_kv_heads in (8, 16)
    ):
        return False, "xcdlb (prototype) needs persistent NP=304, Hq=32, Hkv 8/16"
    if spec.wg_timing and not spec.persistent:
        return False, "wg_timing (prototype) is persistent-grid only"
    # The swizzle mask (V_LDROW//4 - 1) only tiles a pow2 >= 64 row; an explicit
    # v_row_pad that leaves V_LDROW non-pow2 while the swizzle is on would emit an
    # out-of-bounds column. Reject loudly (set use_v_swizzle=False to sweep the pad).
    if spec.resolved_use_v_swizzle():
        _vldrow = spec.block_n + spec.resolved_v_row_pad()
        if _vldrow < 64 or (_vldrow & (_vldrow - 1)) != 0:
            return False, (
                f"use_v_swizzle needs a pow2 V^T row >= 64 but block_n="
                f"{spec.block_n} + v_row_pad gives V_LDROW={_vldrow}; leave "
                f"v_row_pad=None (derived) or set use_v_swizzle=False to sweep the pad"
            )

    # --- Wave/tile divisibility, mirrored from the builder so support() and build()
    # agree on exactly one set of specs (the module contract at the top of this file).
    # The condition below is ALSO enforced in _build_attention_dense_single_buffer;
    # without it here, a spec support() accepted would die in the builder with a
    # ValueError --
    # precisely the dispatch fall-through hole this gate exists to close.
    _waves = spec.block_m // 32
    _rpi = _rows_per_instr(spec.head_size)
    if spec.block_n % _waves != 0 or (spec.block_n // _waves) % _rpi != 0:
        return False, (
            f"block_n={spec.block_n} over {_waves} waves gives ROWS_PER_WAVE="
            f"{spec.block_n // _waves}, not a multiple of ROWS_PER_INSTR={_rpi} "
            f"(D={spec.head_size}); the async K/V DMA would skip rows"
        )
    # cfvst assigns one 2x2 (token-pair x dim-pair) block per thread with no tail
    # guard, so the CTA thread count must divide the block count. Gated on the cfvst
    # predicate: the naive-V path never reads these quantities, and checking it
    # unconditionally would reject specs the naive path builds fine.
    if spec.resolved_use_cfvst():
        _threads = _waves * 64
        _vblocks = (spec.block_n // 2) * (spec.head_size // 2)
        if _vblocks % _threads != 0:
            return False, (
                f"cfvst: V 2x2 block count {_vblocks} (block_n={spec.block_n}, "
                f"D={spec.head_size}) is not a multiple of the {_threads}-thread CTA"
            )

    # --- LDS budget. Without this, an over-budget tile reaches comgr and fails with
    # an opaque CODEGEN_BC_TO_RELOCATABLE abort instead of a structured reason.
    lds_bytes = _lds_bytes(spec)
    from ..common.attention_arch import attention_lds_capacity_bytes

    capacity = attention_lds_capacity_bytes(arch)
    if lds_bytes > capacity:
        return False, (
            f"K_lds+V_lds needs {lds_bytes} B at block_n={spec.block_n}, "
            f"D={spec.head_size}, which exceeds the {arch} LDS capacity ({capacity} B)"
        )

    # Sliding-window + causal: the last query block's window can start past
    # seqlen_kv (start_tile >= n_up), giving a zero-trip KV loop -> l == 0 ->
    # rcp(0) -> NaN. Same class as the block_m % block_n gate above; reject.
    # Sinks make the zero-trip loop well-defined: l stays at the l0 = 1 seed and
    # o at zero, so the output is 0, which is softmax([sink]) applied to no values.
    if spec.sliding_window and spec.causal and not spec.use_sinks:
        _n_q = spec.seqlen_q // spec.block_m
        # Floor division, so the guard reads the same tile count the KV loop uses
        # (n_ktiles = Skv // BN). The shared spec enforces seqlen_kv % block_n == 0,
        # so floor and ceil coincide for every spec that reaches here.
        _n_ktiles = spec.seqlen_kv // spec.block_n
        _n_per = spec.block_m // spec.block_n
        _swt = spec.sliding_window // spec.block_n
        if (_n_q - 1) * _n_per - _swt >= _n_ktiles:
            return False, (
                f"sliding_window={spec.sliding_window}: last query block's window "
                f"starts at tile {(_n_q - 1) * _n_per - _swt}, past seqlen_kv tile "
                f"count {_n_ktiles} -> zero-trip KV loop -> NaN (well-defined "
                f"with use_sinks)"
            )
    return True, ""


def _xcdlb_decode(shape, wi):
    """xcdlb prototype (port of the Oct-24 attention_dense (13).py decode).

    The original looped ``stream = cta_id // 8 .. step 38`` per XCD ``cta_id % 8``;
    at NP = 304 that is exactly ``wi // 8`` / ``wi % 8`` of the grid-stride loop
    (``wi = cta_id + round * 304``), and ``wi < W`` iff ``stream < B * NQB * 4``
    (Hq = 32). Returns ``(qb, hq, bt, hkv=...)`` for ``_run_work_item``."""
    b = shape.b
    c = b.const_i32
    owner = b.mod(wi, c(8))  # XCD: 4 Q heads, 1 (Hkv 8) or 2 (Hkv 16) kv heads
    stream = b.div(wi, c(8))
    nqb = shape.val("nqb")
    per_bt = b.mul(nqb, c(4))  # this XCD's work per batch
    base_jobs = b.div(per_bt, c(38))
    n_long = b.mod(per_bt, c(38))
    n_short = b.sub(c(38), n_long)
    # floor(stream / (NQB*4)) == floor((stream >> 2) / NQB), via the nqb fastdiv.
    bt = shape.div(b.lshr(stream, c(2)), "nqb")
    local = b.sub(stream, b.mul(bt, per_bt))
    rnd = b.div(local, c(38))
    slot = b.mod(local, c(38))
    # Capacity-aware 38-worker causal snake, short/long slot permutation and
    # partial final round, as in the original.
    base_pos = b.select(
        b.cmp_lt(slot, n_long), b.add(n_short, slot), b.sub(slot, n_long)
    )
    even = b.cmp_eq(b.mod(rnd, c(2)), c(0))
    full_rank = b.add(
        b.mul(rnd, c(38)), b.select(even, base_pos, b.sub(c(37), base_pos))
    )
    part_rank = b.add(
        b.mul(base_jobs, c(38)),
        b.select(even, slot, b.sub(b.sub(n_long, c(1)), slot)),
    )
    rank = b.select(b.cmp_lt(rnd, base_jobs), full_rank, part_rank)
    # Longest query block first, rotating the XCD's 4 local Q heads.
    qb = b.sub(b.sub(nqb, c(1)), b.lshr(rank, c(2)))
    lhq = b.mod(rank, c(4))
    hq = b.add(b.mul(owner, c(4)), lhq)
    hkv = b.select(
        b.cmp_eq(shape.val("num_kv_heads"), c(8)),
        owner,
        b.add(b.mul(owner, c(2)), b.lshr(lhq, c(1))),
    )
    return qb, hq, bt, hkv


def build_attention_dense(
    spec: Gfx942AttentionDenseSpec,
    *,
    arch: str = "gfx942",
) -> KernelDef:
    """Emit the gfx942 dense flash-attention prefill kernel for ``spec``.

    Delegates to :func:`_build_attention_dense_single_buffer`, which builds the
    default single-buffer body and the independently selected BF16 D128 K-double-
    buffer / V-register-prefetch variants. It retains CFVST, exp2_fast, fused
    rescale, per-config waves-per-eu and persistent grid-stride. Every scope
    restriction --
    including the modes deferred to later follow-ups -- lives in
    :func:`supports_attention_dense`, which is consulted here, so this function is a
    thin gate-plus-delegate and cannot reject a spec that ``supports`` accepted.

    The gfx942-private sweep knobs are FLAT FIELDS on
    :class:`Gfx942AttentionDenseSpec`, so the signature here is the same
    ``(spec, *, arch)`` every other arch's builder takes. A plain
    :class:`AttentionDenseSpec` is promoted to the subclass at its shipped defaults,
    so an existing caller that never mentions those knobs gets byte-identical IR and
    a byte-identical kernel name.

    :raises NotImplementedError: ``arch`` is not gfx942.
    :raises ValueError: ``spec`` is outside the supported set; the message
        carries :func:`supports_attention_dense`'s structured reason (out-of-scope
        dtype or head size, a mode deferred to a later phase, a ``block_n`` that does
        not divide the query tile, an illegal knob override, an over-budget LDS
        footprint, or an extent past the 32-bit addressing limit).
    """
    if arch != "gfx942":
        raise NotImplementedError(
            f"kernels.gfx942.attention_dense is gfx942-only (got {arch})"
        )
    ok, why = supports_attention_dense(spec, arch=arch)
    if not ok:
        raise ValueError(f"unsupported gfx942 attention_dense spec: {why}")
    return _build_attention_dense_single_buffer(_as_gfx942_spec(spec))


def _emit_o_store(
    b, o, q_row_byte, d_half, o_acc, rcp_l, dtype, d_tiles, width, row_has_key=None
):
    """O = acc / l, ``width`` contiguous head-dim elements per global store.

    ``row_has_key`` (LSE path only) zeroes rows that see no key; ``None`` emits the
    unchanged store sequence.
    """
    for dt in range(d_tiles):
        for g in range(4):
            for c in range(0, 4, width):
                d0 = b.add(b.const_i32(dt * 32 + g * 8 + c), d_half)
                addr = b.add(q_row_byte, d0)
                vals = [
                    b.cast_f32_to(
                        b.fmul(b.vec_extract(o_acc[dt], g * 4 + kk), rcp_l), dtype
                    )
                    for kk in range(c, c + width)
                ]
                packed = b.vec_pack(vals, dtype)
                if row_has_key is not None:
                    packed = zero_keyless_row(b, packed, row_has_key, dtype, width)
                b.global_store_vN(o, addr, packed, width, align=2 * width)


def _build_attention_dense_single_buffer(
    spec: "Gfx942AttentionDenseSpec",
) -> KernelDef:
    """gfx942 dense prefill body: 32x32x8 atom + K-loop doubling.

    The opt-in BF16 D128 persistent-grid pipeline exposes K LDS double
    buffering (lds_num_buffers=2) and V register prefetch (v_reg_prefetch=True)
    as independent tuning knobs, off by default. At (1, False) the original single-buffer
    body is unchanged. Each tile computes QK then softmax/PV.
    The per-work-item compute is factored into
    ``_run_work_item(qb, hq, bt)``, shared by the default grid (one CTA per work item)
    and the P4 persistent grid-stride path (one CTA strides over many). Perf levers
    are all on: conflict-free V/cfvst (P1, D128), exp2_fast + fused rescale (P2),
    per-config waves-per-eu (P3), persistent grid-stride + qb/hkv-major decode (P4).
    The transposed-QK architecture (S^T=K@Q^T so P feeds PV lane-locally) is kept
    from the gfx950 sibling — only the MFMA atom and K/V fragment widths change.
    """
    B, Sq, Skv = spec.batch, spec.seqlen_q, spec.seqlen_kv
    Hq, Hkv, D = spec.num_query_heads, spec.num_kv_heads, spec.head_size
    causal = spec.causal
    dtype = _DTYPE_IR[spec.dtype]

    BLOCK_M = spec.block_m
    WAVES = BLOCK_M // 32  # 8
    BN = spec.block_n

    N_SUB = BN // 32  # key 32-tiles per KV tile
    D_TILES = D // 32  # head-dim 32-tiles
    K_STEPS = D // 8  # QK K=8 doubled steps (was D//16 on gfx950)
    KK_STEPS = BN // 8  # PV K=8 doubled steps (was BN//16)
    gqa = Hq // Hkv
    stride_q_tok = Hq * D
    stride_k_tok = Hkv * D
    ROWS_PER_INSTR = _rows_per_instr(D)  # 1 for D128, 2 for D64
    if spec.pv_loop_order == "k_major":
        PV_STEPS = [(dt, kk) for kk in range(KK_STEPS) for dt in range(D_TILES)]
    else:
        PV_STEPS = [(dt, kk) for dt in range(D_TILES) for kk in range(KK_STEPS)]
    PV_PRIORITY = spec.pv_priority
    PV_FENCE_MASK = spec.pv_sched_fence_mask
    DIAG_SPLIT = spec.causal_diag_split

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = WAVES * 64
    # The RESOLVED waves-per-eu, so the emitted attribute and the wpe{N} token in the
    # name above are the same number by construction (a name/binary mismatch here is
    # what makes the name-keyed launcher cache serve a stale HSACO).
    b.kernel.attrs["waves_per_eu"] = spec.resolved_waves_per_eu()

    q = b.param(
        "q_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    k = b.param(
        "k_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    v = b.param(
        "v_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    o = b.param(
        "o_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    scale = b.param("scale", F32)
    # Problem shape as kernel params, so ONE compiled binary serves every
    # (batch, seqlen_q, seqlen_kv, num_query_heads, num_kv_heads). Declared right
    # after scale, one i32 per ``spec.runtime_kernarg_fields`` entry in that tuple's
    # order -- attention_dense_signature and attention_dense_runtime_args read the
    # same tuple, so the kernarg order cannot skew between kernel and launcher. On
    # the persistent grid the tuple also carries the work decode's fast-division
    # magic/shift pairs (see the grid-stride loop below).
    if spec.runtime_shape:
        rt = {f: b.param(f, I32) for f in spec.runtime_kernarg_fields}
        batch_p, seqlen_q_p, seqlen_kv_p = rt["batch"], rt["seqlen_q"], rt["seqlen_kv"]
        hq_p, hkv_p = rt["num_query_heads"], rt["num_kv_heads"]
        # Uniform (SGPR) values, computed once: the token strides feed per-lane
        # address math inside the KV loop, the GQA ratio one divide per CTA.
        gqa_p = b.div(hq_p, hkv_p)
        stride_q_p = b.mul(hq_p, b.const_i32(D))
        stride_k_p = b.mul(hkv_p, b.const_i32(D))
        # The runtime shape as the work decodes read it (DecodeShape).
        shape_values = {
            "batch": batch_p,
            "seqlen_q": seqlen_q_p,
            "seqlen_kv": seqlen_kv_p,
            "num_query_heads": hq_p,
            "num_kv_heads": hkv_p,
            "gqa": gqa_p,
        }
    else:
        batch_p = seqlen_q_p = seqlen_kv_p = None
        hq_p = hkv_p = gqa_p = stride_q_p = stride_k_p = None
        shape_values = None
    # Per-query-head sink logits [Hq], after every runtime kernarg field (the gfx950
    # ABI order). Declared only when use_sinks, so the non-sink ABI and IR are
    # unchanged.
    if spec.use_sinks:
        sinks = b.param(
            "sink_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
        )

    # The baked branch emits each stride constant in place, at its use: the IR
    # builder is side-effecting, so sharing one const node would renumber every SSA
    # value after it and move the goldens of any baked path.
    def _stride_q():
        return stride_q_p if spec.runtime_shape else b.const_i32(stride_q_tok)

    def _stride_k():
        return stride_k_p if spec.runtime_shape else b.const_i32(stride_k_tok)

    if spec.emit_lse:
        lse = b.param(
            "lse_ptr",
            PtrType(F32, "global"),
            noalias=True,
            writeonly=True,
            align=4,
        )
    # KV-segment split: per-segment FP32 partial O [B*Sq, Hq, S, D] and running
    # max (log2 domain) / denominator [B*Sq, Hq, S], merged by the combine kernel.
    KVS = spec.kv_segments
    if KVS > 1:
        segment_params = declare_segment_params(b)
    if spec.wg_timing:
        ts_p = b.param("ts_ptr", PtrType(I32, "global"), noalias=True, writeonly=True, align=16)
    qk_scale = b.fmul(scale, b.const_f32(LOG2E))

    tid = b.thread_id_x()
    # In the CFVST KDB/VPF pipeline, the wave index is uniform across a wave.
    # Scalarize at full EXEC before /64 so subsequent V buffer offsets can
    # stay on the SGPR/SALU path; leave all nonpipeline IR unchanged.
    if _use_kdb(spec) or _use_vpf(spec):
        wave = b.div(b.to_sgpr_u32(tid), b.const_i32(64))
    else:
        wave = b.div(tid, b.const_i32(64))
    lane = b.mod(tid, b.const_i32(64))
    lane_m = b.mod(lane, b.const_i32(32))
    lane_h = b.div(lane, b.const_i32(32))
    d_base = b.mul(lane_h, b.const_i32(4))  # K=8 half-split: lane_h in {0,1} -> col 0/4
    neg_inf = b.const_f32(-1e30)

    # ---- CTA-invariant LDS + buffer setup (allocated ONCE per CTA; for the
    # persistent grid it is REUSED across every work item the CTA strides over, and
    # for the default grid it is the single tile) ----
    # Padded for D128 (one row per DMA instr), unpadded for D64 (two rows per
    # instr need a contiguous stride) -- see _lds_row_stride. Shared with the
    # LDS-budget check in supports_attention_dense so the two cannot drift. The one
    # exception is when _k_group_pad_active(spec): D64 K_lds then takes the padded
    # [1, block_n // rows_per_instr, _k_group_stride] 2-row-group layout below.
    USE_CFVST = spec.resolved_use_cfvst()  # P1 conflict-free V: D128 only
    # bf16 P is packed by _pack_p_bf16 on the transposed-V path (the configuration
    # whose register budget it was sized for); D64 keeps the generic conversion.
    PACK_P_BF16 = USE_CFVST and spec.dtype == "bf16"
    USE_KDB = _use_kdb(spec)
    KSWZ = spec.k_lds_swizzle
    KB128 = spec.k_read_b128
    USE_VPF = _use_vpf(spec)
    USE_PIPELINE = USE_KDB or USE_VPF
    K_BUFS = spec.lds_num_buffers
    LDROW = _lds_row_stride(spec)
    # Hypothesis #3 (D64 K-LDS bank conflicts): K_lds with a row-group boundary pad,
    # sized by the shared spec.lds_k_group_pad (0 disables it -- that is the A/B probe).
    KPAD_D64 = _k_group_pad_active(spec)
    if KPAD_D64:
        K_GROUP = ROWS_PER_INSTR  # rows per async-DMA instruction (=2 at D64)
        K_GROUP_STRIDE = _k_group_stride(spec)  # rows_per_instr*D + pad elems
        K_lds = b.smem_alloc(
            dtype, [1, BN // K_GROUP, K_GROUP_STRIDE], name_hint="Klds"
        )
    else:
        K_lds = b.smem_alloc(dtype, [K_BUFS, BN, LDROW], name_hint="Klds")
    if USE_CFVST:
        # V stored TRANSPOSED (P1 conflict-free V): [dim, token] with token inner and
        # padded by resolved_v_row_pad, so the PV A-operand read is a contiguous
        # ds_read_b64 rather than P0's 4 element-wise ds_read_u16. Filled by the
        # perm_b32 store path below. This is one of exactly TWO reads that SIZE V_lds --
        # the other is the _lds_bytes budget -- and both take it from the same
        # resolver on the same spec, so the budget cannot under-count this allocation.
        V_LDROW = BN + spec.resolved_v_row_pad()
        V_lds = b.smem_alloc(
            dtype, [2 if spec.qkla_vdb else 1, D, V_LDROW], name_hint="VldsT"
        )
        # V^T bank-conflict swizzle (col' = key ^ ((dim & V_SWZ_MASK) << 2)). Whether it
        # is emitted is an independent spec decision (resolved_use_v_swizzle), NOT
        # derived from the row width, so the pad-value sweep cannot silently toggle it;
        # supports_attention_dense has already rejected swizzle-on with a non-pow2 / <64
        # row, so the mask (derived from the row so store and read can't drift) is always
        # in-bounds here. 4 bits (15) saturate the 32-bank field at width 64, 31 at 128.
        SWZ_V = spec.resolved_use_v_swizzle()
        V_SWZ_MASK = V_LDROW // 4 - 1
    else:
        # V keeps the natural [token, dim] async-DMA layout (D64: VGPR-bound, cfvst
        # regresses it -- see _use_cfvst); read element-wise in read_v. No V^T buffer to
        # swizzle here; default the flags the read_v / _cfvst_store_v closures capture so
        # a future call on this path is a no-op, not a NameError.
        SWZ_V = False
        V_SWZ_MASK = 0
        V_lds = b.smem_alloc(dtype, [1, BN, LDROW], name_hint="Vlds")

    n_ktiles = Skv // BN
    n_per = BLOCK_M // BN
    # Sliding-window left-context length. SW == 0 is full causal (the byte-identical
    # always-on path -- every SW-gated block below is elided so the emitted IR is
    # unchanged). SW > 0 attends keys k in (q - SW, q]; the KV loop is pruned to skip
    # tiles fully below the window (see start_tile in _run_work_item) and do_mask
    # applies the extra lower bound. The shared spec __post_init__ guarantees
    # SW % BN == 0 and causal, so SWt is exact. NOTE: named SW/SWt, not W -- the
    # persistent grid-stride path below rebinds `W` to the work-item count
    # (NQB*Hq*B), and _run_work_item closes over it with late binding; reusing `W`
    # here would make the window logic read that work count on the persistent path.
    SW = spec.sliding_window
    SWt = SW // BN  # window length in KV tiles (0 when disabled)
    # Bottom-right causal: query row q sees keys k <= q + (seqlen_kv - seqlen_q).
    # The offset is read from the shape params on the runtime-shape path, so one
    # binary serves every chunk length. Off (plain top-left causal) emits nothing.
    BR = causal and spec.causal_bottom_right

    def _diag_off():
        if spec.runtime_shape:
            return b.sub(seqlen_kv_p, seqlen_q_p)
        return b.const_i32(Skv - Sq)

    # ---- async DMA loaders (arch-neutral; width=1 = CDNA3 legal) ----
    K_LDROW_BYTES = LDROW * 2
    ROWS_PER_WAVE = BN // WAVES
    # The D64 loader steps ROWS_PER_INSTR rows per DMA instruction with a FLOOR
    # (`range(ROWS_PER_WAVE // ROWS_PER_INSTR)`), and one instruction writes
    # 64 lanes x 4 B = 2 unpadded D64 rows starting at row0*K_LDROW_BYTES -- so row0
    # must be even. A non-multiple would silently skip the tail rows, leaving
    # uninitialized LDS that do_qk/read_v then consume as garbage (no fault). Holds
    # today because block_n % 32 == 0 makes BN//8 a multiple of 4; bind it rather
    # than rely on the dataclass validator staying that strict.
    # Raised, not asserted: `python -O` strips asserts, and skipped DMA rows are a
    # SILENT wrong answer (uninitialized LDS consumed as garbage, no fault) -- the
    # same reasoning as the `_BLOCK_M % 32` wave-coverage guard at module scope.
    # Mirrored in supports_attention_dense so support() and build() cannot disagree.
    if BN % WAVES != 0 or ROWS_PER_WAVE % ROWS_PER_INSTR != 0:
        raise ValueError(
            f"block_n={BN} over {WAVES} waves gives ROWS_PER_WAVE={ROWS_PER_WAVE}, "
            f"not a multiple of ROWS_PER_INSTR={ROWS_PER_INSTR} (D={D}); the async "
            f"DMA would skip rows"
        )
    zero_soff = b.const_i32(0)
    K_lds_addr = b.smem_addr_of(K_lds)
    # cfvst feeds V via buffer_load + perm_b32 + smem_store (no async-DMA handle); the
    # naive D64 path still lands V through async_buffer_load_lds, which needs the base.
    V_lds_addr = None if USE_CFVST else b.smem_addr_of(V_lds)
    # num_records must track THIS launch's real extent: under runtime_shape one
    # kernel serves every (seqlen, head count), and a baked bound would silently drop
    # valid reads for a larger one (buffer loads past num_records return 0 with no
    # fault). The baked branch keeps const_i32(B*Skv*...) verbatim.
    _kv_elem_bytes = Hkv * D * 2
    if spec.runtime_shape:
        kv_bytes = b.mul(b.mul(b.mul(batch_p, seqlen_kv_p), hkv_p), b.const_i32(D * 2))
        k_rsrc = b.buffer_rsrc(k, kv_bytes)
        v_rsrc = b.buffer_rsrc(v, kv_bytes)
    else:
        k_rsrc = b.buffer_rsrc(k, b.const_i32(B * Skv * _kv_elem_bytes))
        v_rsrc = b.buffer_rsrc(v, b.const_i32(B * Skv * _kv_elem_bytes))

    # ---- conflict-free V (P1): perm_b32 store-path transpose into V_lds[dim, token] ----
    # Load V naturally [token, dim] (coalesced VMEM over the contiguous dim axis),
    # transpose each 2x2 f16 block in-thread with perm_b32 (no cross-lane, no lgkmcnt;
    # CK transpose_vectors masks), then write ONE contiguous 2-half ds_write per dim
    # row into V_lds[dim, token]. The consumer read_v then reads 4 consecutive tokens
    # at a fixed dim as a single ds_read_b64. Lifted from the proven
    # attention_tiled_2d cfvst vehicle, after a cross-part correctness disproof of its
    # parked sign-flip symptom (procedure recorded in the optimization plan).
    # The block loop is statically unrolled (V_ITEMS is small: 4 at D128/BN64), which
    # is well under the tiled_2d full-unroll IR-explosion threshold.
    THREADS = WAVES * 64
    if USE_CFVST:
        V_TOK_PAIRS = BN // 2
        V_DIM_PAIRS = D // 2
        V_BLOCKS = V_TOK_PAIRS * V_DIM_PAIRS
        # tid-strided block assignment must tile the 2x2 grid exactly (no partial
        # last item, so no per-item guard).
        #
        # CFVST-PATH-ONLY. Computing this unconditionally would reject specs the
        # NAIVE V path builds fine: at D64/block_n=32 with a 1024-thread CTA,
        # V_BLOCKS=512 < THREADS=1024 and the check fires even though `read_v` never
        # touches V_BLOCKS there (D64 is naive-V by _use_cfvst). Raised, not
        # asserted, for the reason stated at the DMA row-split guard above:
        # `python -O` strips asserts, and this one guards a silent wrong answer (an
        # unguarded tail item would read past the tile).
        if V_BLOCKS % THREADS != 0:
            raise ValueError(
                f"cfvst: V 2x2 block count {V_BLOCKS} (block_n={BN}, D={D}) is not a "
                f"multiple of THREADS={THREADS}; a tail item would need a bounds guard"
            )
        V_ITEMS = V_BLOCKS // THREADS
    else:
        V_TOK_PAIRS = V_DIM_PAIRS = V_BLOCKS = V_ITEMS = 0

    VDB = spec.qkla_vdb
    V_SLOT = {"r": None, "w": None}  # qkla_vdb: current read / write V^T slot

    def _cfvst_store_v(payload):
        """perm_b32 2x2 transpose + contiguous ds_write publishing V_lds[dim, token].

        CTA-invariant: reads only the per-tile payload and writes V_lds (a fixed
        allocation), so it is shared by every work item."""
        for d0, t0, x0, x1 in payload:
            # Each output i32 holds 2 CONSECUTIVE tokens (t0, t0+1) at one fixed dim.
            row_d0 = b.perm_b32(x0, x1, b.const_i32(0x01000504))  # (V[t0,d0], V[t1,d0])
            row_d1 = b.perm_b32(
                x0, x1, b.const_i32(0x03020706)
            )  # (V[t0,d0+1], V[t1,d0+1])
            d1 = b.add(d0, b.const_i32(1))
            if SWZ_V:
                # Emit the (and, shl 2, xor) triple as separate ops in that exact order
                # instead of nesting sub-expressions inside b.shl/b.xor argument lists: a
                # C++ builder mirror may evaluate call arguments in a different order and
                # renumber SSA, tripping a parity gate. Same order here = same IR.
                m0 = b.land(d0, b.const_i32(V_SWZ_MASK))
                s0 = b.shl(m0, b.const_i32(2))
                c0 = b.xor(t0, s0)
                m1 = b.land(d1, b.const_i32(V_SWZ_MASK))
                s1 = b.shl(m1, b.const_i32(2))
                c1 = b.xor(t0, s1)
            else:
                c0, c1 = t0, t0
            b.smem_store_vN(
                V_lds,
                [b.const_i32(0) if V_SLOT["w"] is None else V_SLOT["w"], d0, c0],
                b.bitcast(row_d0, VectorType(dtype, 2)),
                2,
            )
            b.smem_store_vN(
                V_lds,
                [b.const_i32(0) if V_SLOT["w"] is None else V_SLOT["w"], d1, c1],
                b.bitcast(row_d1, VectorType(dtype, 2)),
                2,
            )

    def read_v(dt, kk):
        """PV A-operand = V^T[dim, key]. dim=dt*32+lane_m, keys=kk*8+lane_h*4+{0..3}.

        cfvst (D128): the 4 keys are CONTIGUOUS in the transposed V_lds[dim, token],
        so this is ONE ds_read_b64 vs the naive path's 4 element-wise ds_read_u16.
        The values delivered to the MFMA are bit-identical between the two paths (same
        (dim, key) mapping); only the LDS layout and read width differ.
        The XOR swizzle takes this read from 4-way (the pre-swizzle +8-pad develop
        layout) to 1-way / conflict-free -- that is the reviewable win, not removal of a
        pre-existing 32-way jam (the 32-way state only appears if the row is widened to a
        32-dword multiple, and the swizzle then digs back out of it). The residual bank
        conflicts live on the STORE (8-way -> 4-way: within a wave t0 is constant and
        d0 = 2*lane, so (d0 & mask) takes only 8 values), so further conflict work
        belongs there, not on this read.
        naive (D64): element-wise from V_lds[key, dim] (bank-heavy, but D64 is
        VGPR-bound under cfvst, not LDS-read-bound)."""
        if USE_CFVST and V_SLOT["r"] is not None and SWZ_V:
            # qkla_vdb: one per-tile base (slot + row lane_m + (d_base ^ swizzle)),
            # then (kk*8) by XOR (kk*8 and d_base use disjoint bits, so
            # (kk*8 + d) ^ M == (kk*8) ^ (d ^ M)); dt rows fold into immediates.
            if V_SLOT.get("rb") is None:
                m = b.land(lane_m, b.const_i32(V_SWZ_MASK))
                sw = b.shl(m, b.const_i32(2))
                V_SLOT["rb"] = b.add(
                    b.add(
                        b.mul(V_SLOT["r"], b.const_i32(D * V_LDROW)),
                        b.mul(lane_m, b.const_i32(V_LDROW)),
                    ),
                    b.xor(d_base, sw),
                )
            vb = V_SLOT["rb"]
            vx = vb if kk == 0 else b.xor(vb, b.const_i32(kk * 8))
            e = b.add(vx, b.const_i32(dt * 32 * V_LDROW))
            return b.smem_load_vN(V_lds, b.const_i32(0), b.const_i32(0), e,
                                  dtype=dtype, n=4)
        if USE_CFVST:
            dim_row = b.add(b.const_i32(dt * 32), lane_m)
            key0 = b.add(b.const_i32(kk * 8), d_base)
            col = key0
            if SWZ_V:
                m = b.land(dim_row, b.const_i32(V_SWZ_MASK))
                s = b.shl(m, b.const_i32(2))
                col = b.xor(key0, s)
            return b.smem_load_vN(
                V_lds, b.const_i32(0) if V_SLOT["r"] is None else V_SLOT["r"],
                dim_row, col, dtype=dtype, n=4,
            )
        dim_col = b.add(b.const_i32(dt * 32), lane_m)
        elems = []
        for j in range(4):
            key = b.add(b.add(b.const_i32(kk * 8), d_base), b.const_i32(j))
            vv = b.smem_load_vN(V_lds, b.const_i32(0), key, dim_col, dtype=dtype, n=1)
            elems.append(b.vec_extract(vv, 0))
        return b.vec_pack(elems, dtype)

    def do_pv(o_acc_in, p_packs):
        out = list(o_acc_in)
        for dt, kk in PV_STEPS:
            out[dt] = mfma_32x32x8_for_dtype(
                b, dtype, read_v(dt, kk), p_packs[kk], out[dt]
            )
        return out

    def _run_work_item(qb, hq, bt, hkv=None, seg=None):
        """Emit the full attention for ONE (query-block, query-head, batch) work item.

        The default grid calls this once with the CTA's block ids; the persistent grid
        calls it once per grid-stride iteration with the decoded work item. Everything
        that depends on the work-item coordinates lives here (Q/K base offsets, the Q
        packs, the tile loaders, the causal clamp, the KV loop, the O epilogue); the
        LDS buffers, buffer resources, and the coordinate-free helpers above are
        CTA-invariant and closed over.

        ``hkv`` is the kv head. The persistent grid passes it from its decode, so the
        grid-stride loop never divides by the runtime ``gqa``; the default grid leaves
        it ``None`` and takes the one divide per CTA here."""
        if hkv is None:
            hkv = b.div(hq, gqa_p if spec.runtime_shape else b.const_i32(gqa))
        diag_off = _diag_off() if BR else None
        q_tok0 = b.add(b.mul(qb, b.const_i32(BLOCK_M)), b.mul(wave, b.const_i32(32)))
        # NOT hoisted into a local shared with o_base below: the IR builder is
        # side-effecting, so `b.const_i32(Sq)` EMITS a node at the point it is
        # written. Hoisting moves that node earlier and lets the second use CSE
        # away -- semantically identical, but it renumbers every SSA value after it
        # and moves the golden hashes of any baked path. Emit in place on the baked
        # branch.
        q_base = b.add(
            b.mul(
                b.mul(bt, seqlen_q_p if spec.runtime_shape else b.const_i32(Sq)),
                _stride_q(),
            ),
            b.mul(hq, b.const_i32(D)),
        )
        k_base = b.add(
            b.mul(
                b.mul(bt, seqlen_kv_p if spec.runtime_shape else b.const_i32(Skv)),
                _stride_k(),
            ),
            b.mul(hkv, b.const_i32(D)),
        )

        # Q packs (QK B-operand), loaded unscaled. qk_scale = softmax_scale *
        # log2(e) is applied to the fp32 MFMA scores in do_qk instead: rounding
        # Q * qk_scale back to bf16/fp16 before the MFMA would cost up to 2^-8
        # relative error per element (bf16), and the score error it causes grows
        # with the scale.
        q_tok = b.add(q_tok0, lane_m)
        q_packs = []
        if KB128 and not spec.q_lds_stage:
            # Paired-step D permutation: lane half h holds the 8 contiguous dims
            # 16t + 8h .. +7 and splits them over steps 2t (first 4) and 2t+1 (last
            # 4). K uses the same map, so every dim is still summed exactly once.
            q_row = b.add(q_base, b.mul(q_tok, _stride_q()))
            q_h8 = b.mul(lane_h, b.const_i32(8))
            for t in range(K_STEPS // 2):
                col = b.add(b.const_i32(t * 16), q_h8)
                v8 = b.global_load_vN(q, b.add(q_row, col), dtype, 8, align=16)
                for half in range(2):
                    q_packs.append(
                        b.vec_pack(
                            [b.vec_extract(v8, half * 4 + i) for i in range(4)], dtype
                        )
                    )
        elif spec.q_lds_stage and spec.q_full_line:
            # Whole-line Q: lane = (row i*8 + lane/8, dims h*64 + (lane%8)*8) ->
            # 8 rows x 128 B per load; q_raw[h*4 + i].
            q_raw = []
            lane_q = b.optimization_barrier(lane)  # keep per item (no hoist/spill)
            q_lrow = b.div(lane_q, b.const_i32(8))
            q_lcol = b.mul(b.mod(lane_q, b.const_i32(8)), b.const_i32(8))
            for h in range(2):
                for i in range(4):
                    row = b.add(b.add(q_tok0, b.const_i32(i * 8)), q_lrow)
                    col = b.add(b.const_i32(h * 64), q_lcol)
                    addr = b.add(b.add(q_base, b.mul(row, _stride_q())), col)
                    q_raw.append(b.global_load_vN(q, addr, dtype, 8, align=16))
        elif spec.q_lds_stage:
            # Row-contiguous Q: lane = (row i*16 + lane/4, 8 dims at quarter p,
            # chunk lane%4) -> 16 rows x 64 B per load. _stage_q transposes it.
            q_raw = []
            q_lrow = b.div(lane, b.const_i32(4))
            q_lcol = b.mul(b.mod(lane, b.const_i32(4)), b.const_i32(8))
            for p in range(4):
                for i in range(2):
                    row = b.add(b.add(q_tok0, b.const_i32(i * 16)), q_lrow)
                    col = b.add(b.const_i32(p * 32), q_lcol)
                    addr = b.add(b.add(q_base, b.mul(row, _stride_q())), col)
                    q_raw.append(b.global_load_vN(q, addr, dtype, 8, align=16))
        else:
            for ks in range(K_STEPS):
                col = b.add(b.const_i32(ks * 8), d_base)
                addr = b.add(b.add(q_base, b.mul(q_tok, _stride_q())), col)
                q_packs.append(b.global_load_vN(q, addr, dtype, 4, align=8))

        def _async_load_rows(rsrc, lds_base, tile_key0, buf_off, kswz):
            """D128 async DMA, one K row per instruction; yields after each row so a
            caller can spread the issue (tile_interleave). _async_load drains it, so
            the default emission is op-for-op unchanged."""
            for r in range(ROWS_PER_WAVE):
                row = b.add(b.mul(wave, b.const_i32(ROWS_PER_WAVE)), b.const_i32(r))
                row_lds_off = b.zext(b.mul(row, b.const_i32(K_LDROW_BYTES)), I64)
                if buf_off is not None:
                    row_lds_off = b.add(buf_off, row_lds_off)
                row_base = b.smem_ptr_add(lds_base, row_lds_off)
                gkey = b.add(tile_key0, row)
                gcol = b.mul(lane, b.const_i32(2))
                voff = b.add(b.add(k_base, b.mul(gkey, _stride_k())), gcol)
                if kswz:
                    # Lane writes LDS col 2*lane; fetch the logical col that the
                    # 16 B-chunk XOR swizzle maps there (XOR is its own inverse).
                    # Applied to the whole per-tile offset (its row/tile part is a
                    # multiple of the 128-elem row, so only the lane col changes),
                    # which keeps one lane base instead of 8 hoisted per-row ones.
                    km = b.land(row, b.const_i32(7))
                    ks_ = b.shl(km, b.const_i32(3))
                    voff = b.xor(voff, ks_)
                b.async_buffer_load_lds_addr(
                    rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 1
                )
                yield

        def _async_load(rsrc, lds_base, tile_key0, group_pad=False, kbuf=None, kswz=False):
            # Only the pipelined K path passes kbuf; the default emission is
            # unchanged (no extra IR operations and the same LDS addresses).
            buf_off = (
                b.zext(b.mul(kbuf, b.const_i32(BN * LDROW * 2)), I64)
                if kbuf is not None
                else None
            )
            if ROWS_PER_INSTR == 1:
                for _ in _async_load_rows(rsrc, lds_base, tile_key0, buf_off, kswz):
                    pass
            else:
                lanes_per_row = D // 2
                sub_row = b.div(lane, b.const_i32(lanes_per_row))
                col = b.mul(b.mod(lane, b.const_i32(lanes_per_row)), b.const_i32(2))
                for it in range(ROWS_PER_WAVE // ROWS_PER_INSTR):
                    row0 = b.add(
                        b.mul(wave, b.const_i32(ROWS_PER_WAVE)),
                        b.const_i32(it * ROWS_PER_INSTR),
                    )
                    if group_pad:
                        # 2-row-group boundary pad (the D64 K-pad probe): the DMA still
                        # writes one whole group (ROWS_PER_INSTR contiguous rows)
                        # per instruction, but consecutive groups are spaced by the
                        # padded K_GROUP_STRIDE so do_qk's krow reads land 4-way (not
                        # 32-way) bank-conflicted. group = row0 // ROWS_PER_INSTR.
                        grp = b.add(
                            b.mul(
                                wave,
                                b.const_i32(ROWS_PER_WAVE // ROWS_PER_INSTR),
                            ),
                            b.const_i32(it),
                        )
                        row_lds_off = b.zext(
                            b.mul(grp, b.const_i32(K_GROUP_STRIDE * 2)), I64
                        )
                    else:
                        row_lds_off = b.zext(
                            b.mul(row0, b.const_i32(K_LDROW_BYTES)), I64
                        )
                    row_base = b.smem_ptr_add(lds_base, row_lds_off)
                    gkey = b.add(b.add(tile_key0, row0), sub_row)
                    voff = b.add(b.add(k_base, b.mul(gkey, _stride_k())), col)
                    b.async_buffer_load_lds_addr(
                        rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 1
                    )

        def load_tile(tile_idx):
            # cfvst (D128): K only; V is fed through the perm_b32 store path below
            # (async DMA would land V in the WRONG [token, dim] orientation for a
            # conflict-free read). Naive (D64): K and V both via async DMA, natural
            # [token, dim].
            tk0 = b.mul(tile_idx, b.const_i32(BN))
            _async_load(k_rsrc, K_lds_addr, tk0, group_pad=KPAD_D64, kswz=KSWZ)
            if not USE_CFVST:
                _async_load(v_rsrc, V_lds_addr, tk0)

        def load_pipeline_k(kbuf, tile_idx):
            """Issue K(tile_idx) DMA into the selected K LDS slot."""
            tk0 = b.mul(tile_idx, b.const_i32(BN))
            _async_load(k_rsrc, K_lds_addr, tk0, kbuf=kbuf, kswz=KSWZ)

        def load_pipeline_k_steps(kbuf, tile_idx):
            """load_pipeline_k one DMA row per step (D128), for tile_interleave."""
            tk0 = b.mul(tile_idx, b.const_i32(BN))
            buf_off = b.zext(b.mul(kbuf, b.const_i32(BN * LDROW * 2)), I64)
            return _async_load_rows(k_rsrc, K_lds_addr, tk0, buf_off, KSWZ)

        def _cfvst_load_v(tile_idx):
            """Issue the cfvst VMEM loads and keep each thread's V tile in VGPRs.

            Returns a payload of ``(d0, t0, x0, x1)`` per unrolled item: ``x0``/``x1``
            are the i32-packed ``<2 x elem>`` loads of token rows ``t0``/``t0+1`` at
            the contiguous dim pair ``(d0, d0+1)``. Loads go through ``v_rsrc`` so an
            out-of-range token is hardware-clamped to 0 (matches the async path)."""
            tile_tok0 = b.mul(tile_idx, b.const_i32(BN))
            payload = []
            for it in range(V_ITEMS):
                blk = b.add(b.mul(b.const_i32(it), b.const_i32(THREADS)), tid)
                # dim-pair is the fastest-varying coord so adjacent lanes issue
                # coalesced VMEM loads over the natural [token, dim] layout.
                tg = b.div(blk, b.const_i32(V_DIM_PAIRS))
                dg = b.mod(blk, b.const_i32(V_DIM_PAIRS))
                t0 = b.mul(tg, b.const_i32(2))
                d0 = b.mul(dg, b.const_i32(2))
                gk0 = b.add(tile_tok0, t0)
                gk1 = b.add(gk0, b.const_i32(1))
                # byte offset of (token, d0); contiguous dim pair -> one 2-half load.
                eoff0 = b.add(b.add(k_base, b.mul(gk0, _stride_k())), d0)
                eoff1 = b.add(b.add(k_base, b.mul(gk1, _stride_k())), d0)
                x0 = b.buffer_load_vN(
                    v_rsrc, b.mul(eoff0, b.const_i32(2)), zero_soff, dtype, 2
                )
                x1 = b.buffer_load_vN(
                    v_rsrc, b.mul(eoff1, b.const_i32(2)), zero_soff, dtype, 2
                )
                payload.append((d0, t0, b.bitcast(x0, I32), b.bitcast(x1, I32)))
            return payload

        def _cfvst_load_v_pipeline(tile_idx):
            values = []
            for _ in _cfvst_load_v_pipeline_steps(tile_idx, values):
                pass
            return values

        def _cfvst_load_v_pipeline_steps(tile_idx, values):
            """Load packed V with one SGPR tile/wave base and a VGPR lane offset.

            For BF16 D128 / BN64, V_DIM_PAIRS=64 and
              blk = it * (WAVES * 64) + tid,
              d0 = 2 * lane,
              t0 = 2 * wave + 2 * it * WAVES.
            The original unrolled loader rebuilt per-thread byte addresses and
            div/mod coordinates for each item. Use buffer soffset for the uniform
            tile/wave part and buffer voffset for the reusable lane offset.
            Reconstruct V^T publication coordinates after PV rather than keeping
            them live across the prefetch window.
            """
            if D != 128 or V_DIM_PAIRS != 64:
                raise ValueError("pipeline V SGPR addressing requires D=128")
            wave_key0 = b.add(
                b.mul(tile_idx, b.const_i32(BN)),
                b.mul(wave, b.const_i32(2)),
            )
            # Do the uniform-to-scalar conversion just once per tile.
            soff0 = b.to_sgpr_u32(
                b.mul(b.add(k_base, b.mul(wave_key0, _stride_k())), b.const_i32(2))
            )
            stride_bytes = b.mul(_stride_k(), b.const_i32(2))
            lane_byte = b.mul(lane, b.const_i32(4))
            for it in range(V_ITEMS):
                soff_it = (
                    soff0
                    if it == 0
                    else b.add(
                        soff0,
                        b.mul(stride_bytes, b.const_i32(it * 2 * WAVES)),
                    )
                )
                x0 = b.buffer_load_vN(v_rsrc, lane_byte, soff_it, dtype, 2)
                x1 = b.buffer_load_vN(
                    v_rsrc, lane_byte, b.add(soff_it, stride_bytes), dtype, 2
                )
                values.append((b.bitcast(x0, I32), b.bitcast(x1, I32)))
                yield

        def do_qk(kbuf=None):
            """S^T = K@Q^T via the 32x32x8 atom (K-doubled)."""
            s_reg = []
            for nsub in range(N_SUB):
                acc = b.zero_vec_f32(16)
                krow = b.add(b.const_i32(nsub * 32), lane_m)
                if KPAD_D64:
                    # 2-row-group padded K_lds[1, BN//2, 2*D+pad]: decompose the
                    # logical krow into (group = krow >> 1, within = krow & 1); the
                    # padded group stride spreads consecutive krow across 4 (not 32)
                    # banks (the D64 K-pad probe). col stays within the D-wide row.
                    k_group = b.lshr(krow, b.const_i32(1))
                    k_within_off = b.mul(b.land(krow, b.const_i32(1)), b.const_i32(D))
                if KB128:
                    k_h8 = b.mul(lane_h, b.const_i32(8))
                    if KSWZ:
                        # One base per lane: A = slot + row(lane_m) + (8h ^ 8(r & 7)),
                        # flat in elements. Slot and row offsets are multiples of the
                        # 128-elem row, so XOR with the in-row step constant 16(t & 3)
                        # never reaches them; t bit 2 (+64) and nsub (+32 rows) are adds
                        # that fold into ds_read offsets.
                        k_swz = b.shl(b.land(lane_m, b.const_i32(7)), b.const_i32(3))
                        k_lo = b.xor(k_h8, k_swz)
                        k_a = b.add(b.mul(lane_m, b.const_i32(LDROW)), k_lo)
                        if kbuf is not None:
                            k_a = b.add(b.mul(kbuf, b.const_i32(BN * LDROW)), k_a)
                    for t in range(K_STEPS // 2):
                        if KSWZ:
                            k_x = k_a if t % 4 == 0 else b.xor(k_a, b.const_i32((t % 4) * 16))
                            col = b.add(
                                k_x, b.const_i32(nsub * 32 * LDROW + (t // 4) * 64)
                            )
                            k8 = b.smem_load_vN(K_lds, b.const_i32(0), b.const_i32(0),
                                                col, dtype=dtype, n=8)
                        else:
                            col = b.add(b.const_i32(t * 16), k_h8)
                            k8 = b.smem_load_vN(
                                K_lds, b.const_i32(0) if kbuf is None else kbuf,
                                krow, col, dtype=dtype, n=8)
                        for half in range(2):
                            k_pack = b.vec_pack(
                                [b.vec_extract(k8, half * 4 + i) for i in range(4)], dtype
                            )
                            acc = mfma_32x32x8_for_dtype(
                                b, dtype, k_pack, q_packs[2 * t + half], acc
                            )
                for ks in range(K_STEPS if not KB128 else 0):
                    col = b.add(b.const_i32(ks * 8), d_base)
                    if KPAD_D64:
                        k_pack = b.smem_load_vN(
                            K_lds,
                            b.const_i32(0) if kbuf is None else kbuf,
                            k_group,
                            b.add(k_within_off, col),
                            dtype=dtype,
                            n=4,
                        )
                    else:
                        k_pack = b.smem_load_vN(
                            K_lds,
                            b.const_i32(0) if kbuf is None else kbuf,
                            krow,
                            col,
                            dtype=dtype,
                            n=4,
                        )
                    acc = mfma_32x32x8_for_dtype(b, dtype, k_pack, q_packs[ks], acc)
                # Into the log2 domain in fp32, so masking, the row max and
                # exp2(s - m) all see the same rounded score (s - m <= 0).
                s_reg.append(
                    [b.fmul(b.vec_extract(acc, i), qk_scale) for i in range(16)]
                )
            return s_reg

        def do_mask(s_reg, tile_idx, lower=False, upper=True):
            # upper: causal bound (ktok <= query_tok). lower: sliding-window bound
            # (ktok > query_tok - W), only emitted when lower=True. With the default
            # lower=False/upper=True the emitted IR is exactly the pre-window causal
            # mask, so the W == 0 path stays byte-identical. Mirrors the gfx950 dense
            # do_mask (gfx950/attention_dense.py). The mask is on the QK output S
            # (N_SUB x 16 regs) and is independent of the doubled 32x32x8 K/PV loops.
            if not causal:
                return
            tile_key0 = b.mul(tile_idx, b.const_i32(BN))
            query_tok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
            if BR:
                query_tok = b.add(query_tok, diag_off)
            win_lo = b.sub(query_tok, b.const_i32(SW)) if lower else None
            for nsub in range(N_SUB):
                sub_base = b.add(tile_key0, b.const_i32(nsub * 32))
                for i in range(16):
                    ktok = b.add(sub_base, _mfma_32x32_c_row(b, lane, i))
                    if upper:
                        s_reg[nsub][i] = b.select(
                            b.cmp_le(ktok, query_tok), s_reg[nsub][i], neg_inf
                        )
                    if lower:
                        s_reg[nsub][i] = b.select(
                            b.cmp_gt(ktok, win_lo), s_reg[nsub][i], neg_inf
                        )

        # n_up: causal clamps the KV loop to the diagonal tile of this query block.
        # The window does not move the upper diagonal, only the lower edge, so n_up
        # is unchanged by W.
        # BN is a power of two, so the runtime divide lowers to a shift. The causal
        # clamp below needs no further edit -- it consumes n_ktiles_c, so converting
        # the trip count carries it.
        n_ktiles_c = (
            b.div(seqlen_kv_p, b.const_i32(BN))
            if spec.runtime_shape
            else b.const_i32(n_ktiles)
        )
        if BR:
            # Last visible key of the block is qb*BLOCK_M + BLOCK_M - 1 + diag_off;
            # ceil it to a KV-tile count (all terms are non-negative).
            n_up = b.add(
                b.div(
                    b.add(
                        b.mul(qb, b.const_i32(BLOCK_M)),
                        b.add(diag_off, b.const_i32(BLOCK_M - 1)),
                    ),
                    b.const_i32(BN),
                ),
                b.const_i32(1),
            )
            n_up = b.select(b.cmp_lt(n_up, n_ktiles_c), n_up, n_ktiles_c)
        elif causal:
            n_up = b.add(b.mul(qb, b.const_i32(n_per)), b.const_i32(n_per))
            n_up = b.select(b.cmp_lt(n_up, n_ktiles_c), n_up, n_ktiles_c)
        else:
            n_up = n_ktiles_c

        # start_tile: sliding-window lower bound on the KV loop. Tiles fully below
        # the window (< diag_start - Wt) are all -inf, so the loop skips them. Uses
        # the per-work-item qb, so the persistent grid-stride path is covered too.
        # W == 0 emits NOTHING here; the loop's lower bound const is created inline
        # below at its original position so the W == 0 IR stays byte-identical (an
        # earlier const_i32 would shift the shared SSA counter and rename o0..o3).
        if SW:
            _diag_start = b.mul(qb, b.const_i32(n_per))
            _lo_raw = b.sub(_diag_start, b.const_i32(SWt))
            start_tile = b.select(
                b.cmp_gt(_lo_raw, b.const_i32(0)), _lo_raw, b.const_i32(0)
            )
            # Persistent grid: bound start_tile by n_up - 1 so LLVM can prove the
            # KV loop runs at least once. With the runtime trip count it cannot see
            # that on its own, and it then clones the epilogue and the grid-stride
            # latch for a zero-trip edge, which costs the outer loop its schedule.
            # A no-op on every supported shape: the sliding-window guard in
            # supports_attention_dense keeps the last (latest-starting) query
            # block's start_tile below n_ktiles, every start_tile is below its own
            # diagonal, and n_up >= 1 since seqlen_kv >= BN. The default grid runs
            # one item per CTA, so it keeps its IR unchanged.
            if spec.persistent:
                _last = b.sub(n_up, b.const_i32(1))
                start_tile = b.select(b.cmp_lt(start_tile, _last), start_tile, _last)

        # ---- online-softmax main loop (non-pipelined, single buffer) ----
        if spec.use_sinks:
            # The sink is a virtual key with logit sink[hq] and zero value: seed the
            # running max with it (log2 domain) and the denominator with exp2(0) = 1.
            # Seeded once per work item before the loop, never inside the doubled
            # 32x32x8 K steps. The loop's l_i*alpha rescales the seed on every tile,
            # including the first, so no tile-0 fix-up is needed.
            sink_h = b.global_load(sinks, hq, dtype, align=2)
            m0 = b.fmul(b.cast_to_f32(sink_h), b.const_f32(LOG2E))
            l0 = b.const_f32(1.0)
        else:
            m0 = neg_inf
            l0 = b.const_f32(0.0)
        o0 = [b.zero_vec_f32(16) for _ in range(D_TILES)]
        iter_args = [("m", m0), ("l", l0)] + [
            (f"o{dt}", o0[dt]) for dt in range(D_TILES)
        ]

        def softmax_pv(s, j, m_i, l_i, o_acc, masked):
            """One KV tile's mask -> online softmax -> PV update, shared by the
            single-buffer ``tile_compute`` and the KDB/VPF pipeline so both
            honour the same exp2, fused-rescale and PV scheduling knobs. Returns
            the updated ``(m, l, o_acc)``; the caller owns loads and barriers."""
            # Every visited tile gets the causal upper bound; W > 0 additionally
            # applies the window lower bound (redundant no-op compares on tiles
            # fully inside the window). W == 0 -> lower=False, byte-identical.
            # do_mask early-returns when non-causal, so upper is only read on the
            # causal path (where it is the causal bound); the arg is kept for
            # signature parity with the gfx950 do_mask, which does use upper=False.
            # masked=False only for below-diagonal tiles of the diagonal split.
            if masked is True:
                do_mask(s, j, lower=(SW > 0), upper=True)
            elif masked:
                # (lower, upper) pair from the sliding-window phase split.
                lower, upper = masked
                if lower or upper:
                    do_mask(s, j, lower=lower, upper=upper)

            # tile max over keys (both lane-halves) for this query.
            local_max = neg_inf
            for nsub in range(N_SUB):
                for i in range(16):
                    local_max = b.fmax(local_max, s[nsub][i])
            tile_max = b.fmax(local_max, b.warp_shuffle_xor(local_max, 32))
            m_new = b.fmax(m_i, tile_max)
            # P2: exp2_fast (llvm.amdgcn.exp2.f32 -> one v_exp_f32) drops the ~5-VALU
            # guarded range reduction that plain exp2 (llvm.exp2.f32) emits. Safe
            # here: both softmax args are always <= 0 -- alpha's m_i - m_new (m_new =
            # max(m_i, tile_max) >= m_i) and p's s - m_new (m_new >= tile_max >= every
            # s) -- exactly exp2_fast's precondition (no overflow; v_exp_f32 flushes
            # large negatives to 0). Cuts ~99 VALU/tile at D128, the dominant
            # MFMA-starving residual once conflict-free V (P1) lands. Enabled by
            # _use_exp2_fast for every config except bf16 D128 on the default grid,
            # which reverts to plain exp2; rationale + grid gate in that docstring.
            exp2 = b.exp2_fast if spec.resolved_use_exp2_fast() else b.exp2
            alpha = exp2(b.fsub(m_i, m_new))

            # P2 fused rescale: compute each exp2 inline, accumulate l_local, and
            # cast->pack it into the PV B-operand in ONE pass. The f32 p value dies
            # after its two uses (the l_local fadd + the dtype cast) instead of
            # staying live across the whole l_local reduction AND a separate cast/pack
            # pass. Peak live f32 p regs drop from N_SUB*16 (=32 at D128) to ~one
            # pack, freeing ~28 VGPR (plan §6.1).
            #
            # The kk->(nsub,i) map is the PV B-operand relayout (key=kk*8+lane_h*4+j
            # maps to QK C-reg nsub=kk//4, i=(kk%4)*4+j -- same lane_h split, no
            # cross-lane). KK_STEPS==N_SUB*4, so this covers every p element exactly
            # once, and i steps 0,1,2,3 / 4,5,6,7 / ... within each nsub -- the
            # identical accumulation order as a nested (nsub,i) loop, so l_local and
            # the packed casts are numerically bit-identical: pure live-range relief.
            l_local = b.const_f32(0.0)
            p_packs = []
            for kk in range(KK_STEPS):
                nsub = kk // 4
                base_i = (kk % 4) * 4
                elems = []
                for j2 in range(4):
                    pv = exp2(b.fsub(s[nsub][base_i + j2], m_new))
                    l_local = b.fadd(l_local, pv)
                    elems.append(pv if PACK_P_BF16 else b.cast_f32_to(pv, dtype))
                if PACK_P_BF16:
                    p_packs.append(_pack_p_bf16(b, elems))
                else:
                    p_packs.append(b.vec_pack(elems, dtype))
            l_tile = b.fadd(l_local, b.warp_shuffle_xor(l_local, 32))
            l_new = b.fadd(b.fmul(l_i, alpha), l_tile)

            o_acc = [
                b.vec_pack(
                    [b.fmul(b.vec_extract(o_acc[dt], i), alpha) for i in range(16)], F32
                )
                for dt in range(D_TILES)
            ]
            if PV_FENCE_MASK is not None:
                b.sched_barrier(PV_FENCE_MASK)
            if PV_PRIORITY:
                b.s_setprio(PV_PRIORITY)
            o_acc = do_pv(o_acc, p_packs)
            if PV_PRIORITY:
                b.s_setprio(0)
            return m_new, l_new, o_acc

        def tile_ilv(kbuf, j, m_i, l_i, o_acc, make_v_steps, make_k_steps,
                     v_store_steps=None):
            """tile_interleave: QK, online softmax and PV of one unmasked tile, with
            VALU placed between independent MFMAs in program order (in-order issue
            only overlaps what is interleaved).

            make_v_steps(values) -> generator issuing the V(j+1) register prefetch
            (no LDS), appending its (x0, x1) payload to ``values``; it goes in the QK
            nsub1 gaps, or in the PV gaps with tilv_v_write_qk. make_k_steps() ->
            generator issuing the K(j+1) LDS DMA one row per PV MFMA from the second P
            pack on, after every K read of the tile (K reads alias the K DMA; the alias
            scopes only separate K from V). v_store_steps (tilv_v_write_qk) writes
            V(j) to LDS in the QK nsub0 gaps, followed by lgkmcnt(0) + barrier.
            Returns (m_new, l_new, o_acc, v_next_values)."""
            VWQ = v_store_steps is not None

            def qk_chain(nsub, between=None):
                acc = b.zero_vec_f32(16)
                krow = b.add(b.const_i32(nsub * 32), lane_m)
                for ks in range(K_STEPS):
                    col = b.add(b.const_i32(ks * 8), d_base)
                    k_pack = b.smem_load_vN(K_lds, kbuf, krow, col, dtype=dtype, n=4)
                    acc = mfma_32x32x8_for_dtype(b, dtype, k_pack, q_packs[ks], acc)
                    if between is not None:
                        between(ks)
                return acc

            if VWQ:
                def store_v(ks):
                    if ks % 4 == 0:
                        next(v_store_steps, None)
                acc0 = qk_chain(0, store_v)
                for _ in v_store_steps:
                    pass
                # Publish V(j): every wave's writes land before any wave's PV(j).
                b.sync_lds_only()
            else:
                acc0 = qk_chain(0)
            # QK nsub1 chain with nsub0's scale + partial max between its MFMAs.
            s0 = [None] * 16
            mx = [neg_inf]
            v_next = []
            v_steps_qk = None if VWQ else make_v_steps(v_next)

            def scale_max0(ks):
                s0[ks] = b.fmul(b.vec_extract(acc0, ks), qk_scale)
                mx[0] = b.fmax(mx[0], s0[ks])
                if v_steps_qk is not None and ks % 4 == 1:
                    next(v_steps_qk, None)
            acc1 = qk_chain(1, scale_max0)
            if v_steps_qk is not None:
                for _ in v_steps_qk:
                    pass
            s1 = []
            for i in range(16):
                s1.append(b.fmul(b.vec_extract(acc1, i), qk_scale))
                mx[0] = b.fmax(mx[0], s1[i])
            s = [s0, s1]
            local_max = mx[0]
            tile_max = b.fmax(local_max, b.warp_shuffle_xor(local_max, 32))
            m_new = b.fmax(m_i, tile_max)
            exp2 = b.exp2_fast if spec.resolved_use_exp2_fast() else b.exp2
            alpha = exp2(b.fsub(m_i, m_new))

            def make_pack(elems):
                return _pack_p_bf16(b, elems) if PACK_P_BF16 else b.vec_pack(elems, dtype)

            def exp_elem(l_acc, kk, j2):
                pv = exp2(b.fsub(s[kk // 4][(kk % 4) * 4 + j2], m_new))
                l_acc[0] = b.fadd(l_acc[0], pv)
                return pv if PACK_P_BF16 else b.cast_f32_to(pv, dtype)

            if spec.tilv_rescale_skip:
                # alpha == 1 on every lane: O * 1 == O, so skipping it is exact.
                # Wave-uniform; each O[dt] rescale sits in its own small branch so the
                # MFMAs and exps around it stay unduplicated.
                same = b.wave_all(
                    b.select(
                        b.fcmp("oeq", alpha, b.const_f32(1.0)),
                        b.const_i32(1),
                        b.const_i32(0),
                    )
                )
                need = b.cmp_eq(same, b.const_i32(0))

            def first_pack(o_in, v0):
                """Pack 0, O rescale (O[0] before, O[1..3] between the pack-0
                MFMAs) and pack 1's exps between them. Returns (o, l_acc, pack1)."""
                o = list(o_in)
                l_acc = [b.const_f32(0.0)]

                def rescale(dt):
                    def mul():
                        return b.vec_pack(
                            [b.fmul(b.vec_extract(o[dt], i), alpha) for i in range(16)],
                            F32,
                        )
                    if not spec.tilv_rescale_skip:
                        return mul()
                    br = b.scf_if_else(need, result_types=[VectorType(F32, 16)])
                    with br as (then_ctx, else_ctx):
                        with then_ctx:
                            b.scf_yield(mul())
                        with else_ctx:
                            b.scf_yield(o[dt])
                    return br.results[0]
                pack0 = make_pack([exp_elem(l_acc, 0, j2) for j2 in range(4)])
                o[0] = rescale(0)
                elems = []
                for dt in range(D_TILES):
                    o[dt] = mfma_32x32x8_for_dtype(b, dtype, v0[dt], pack0, o[dt])
                    if dt + 1 < D_TILES:
                        o[dt + 1] = rescale(dt + 1)
                    elems.append(exp_elem(l_acc, 1, dt))
                return o, l_acc, make_pack(elems)

            o_acc = list(o_acc)
            v0 = [read_v(dt, 0) for dt in range(D_TILES)]
            v_cur = [read_v(dt, 1) for dt in range(D_TILES)]
            o_acc, l_acc, pack = first_pack(o_acc, v0)
            vals = []
            v_steps = make_v_steps(vals) if VWQ else None
            k_steps = make_k_steps()
            for kk in range(1, KK_STEPS):
                nxt = kk + 1 < KK_STEPS
                v_nxt = [read_v(dt, kk + 1) for dt in range(D_TILES)] if nxt else None
                elems = []
                for dt in range(D_TILES):
                    o_acc[dt] = mfma_32x32x8_for_dtype(b, dtype, v_cur[dt], pack, o_acc[dt])
                    if nxt:
                        elems.append(exp_elem(l_acc, kk + 1, dt))
                    if 1 <= kk <= 2:
                        next(k_steps, None)
                    if v_steps is not None and kk == 3:
                        next(v_steps, None)
                if nxt:
                    pack = make_pack(elems)
                    v_cur = v_nxt
            for _ in k_steps:
                pass
            if v_steps is not None:
                for _ in v_steps:
                    pass
                v_next = vals
            l_local = l_acc[0]
            l_tile = b.fadd(l_local, b.warp_shuffle_xor(l_local, 32))
            l_new = b.fadd(b.fmul(l_i, alpha), l_tile)
            return m_new, l_new, o_acc, v_next

        QK_READ_AHEAD = spec.qkla_read_ahead or 3

        def qk_raw_steps(kbuf, out, preyield=False):
            """qk_lookahead: unscaled QK MFMA chain as a generator that yields after
            each (K ds_read, MFMA) pair. The two key subtiles alternate; each still
            accumulates ks in order, so the scores match do_qk exactly. Appends the
            N_SUB accumulators to ``out`` when exhausted."""
            accs = [b.zero_vec_f32(16) for _ in range(N_SUB)]
            krows = [b.add(b.const_i32(nsub * 32), lane_m) for nsub in range(N_SUB)]
            order = [(ks, nsub) for ks in range(K_STEPS) for nsub in range(N_SUB)]

            if KB128:
                # Paired-step map (as do_qk): one ds_read_b128 per (t, nsub) feeds
                # steps ks = 2t, 2t+1. KSWZ: one lane base, 3 v_xor per subtile.
                k_h8 = b.mul(lane_h, b.const_i32(8))
                if KSWZ:
                    k_swz = b.shl(b.land(lane_m, b.const_i32(7)), b.const_i32(3))
                    k_a = b.add(
                        b.mul(lane_m, b.const_i32(LDROW)), b.xor(k_h8, k_swz)
                    )
                    k_a = b.add(b.mul(kbuf, b.const_i32(BN * LDROW)), k_a)
                k8s = {}

                def read8(t, nsub):
                    if KSWZ:
                        k_x = k_a if t % 4 == 0 else b.xor(
                            k_a, b.const_i32((t % 4) * 16)
                        )
                        col = b.add(
                            k_x, b.const_i32(nsub * 32 * LDROW + (t // 4) * 64)
                        )
                        return b.smem_load_vN(K_lds, b.const_i32(0), b.const_i32(0),
                                              col, dtype=dtype, n=8)
                    col = b.add(b.const_i32(t * 16), k_h8)
                    return b.smem_load_vN(K_lds, kbuf, krows[nsub], col,
                                          dtype=dtype, n=8)

                def read(i):
                    ks, nsub = order[i]
                    key = (ks // 2, nsub)
                    if ks % 2 == 0:
                        k8s[key] = read8(*key)
                    return key, ks % 2

                def pack_of(entry):
                    key, half = entry
                    k8 = k8s[key]
                    return b.vec_pack(
                        [b.vec_extract(k8, half * 4 + i) for i in range(4)], dtype
                    )
            else:
                def read(i):
                    ks, nsub = order[i]
                    col = b.add(b.const_i32(ks * 8), d_base)
                    return b.smem_load_vN(K_lds, kbuf, krows[nsub], col, dtype=dtype, n=4)

                def pack_of(entry):
                    return entry

            packs = [read(i) for i in range(min(QK_READ_AHEAD, len(order)))]
            if preyield:
                yield  # rev 3: the first K reads issue before the carried MFMAs
            for i, (ks, nsub) in enumerate(order):
                if i + QK_READ_AHEAD < len(order):
                    packs.append(read(i + QK_READ_AHEAD))
                accs[nsub] = mfma_32x32x8_for_dtype(
                    b, dtype, pack_of(packs[i]), q_packs[ks], accs[nsub]
                )
                yield
            out.extend(accs)

        def tile_la(j, m_i, l_i, o_acc, s_acc, v_store_steps, qk_next, masked,
                    make_v_steps, make_k_steps):
            """qk_lookahead: tile j from its carried raw scores ``s_acc``. Before the
            mid barrier, QK(j+1) (``qk_next``) MFMAs cover V(j)'s LDS write, the
            score scale/mask/max, alpha, the O rescale and pack 0's exps. After it,
            PV(j) per P pack with the next pack's exps, the K(j+2) DMA rows and the
            V(j+1) loads between its MFMAs. Returns (m_new, l_new, o_acc, v_next)."""
            exp2 = b.exp2_fast if spec.resolved_use_exp2_fast() else b.exp2
            st = {}
            PACK_AHEAD = spec.qkla_pack_ahead
            QK_TAIL = spec.qkla_qk_tail
            RESCALE_PLAIN = spec.qkla_rescale_plain

            def make_pack(elems):
                return _pack_p_bf16(b, elems) if PACK_P_BF16 else b.vec_pack(elems, dtype)

            l_acc = [b.const_f32(0.0)]

            def exp_elem(kk, j2):
                s = st["s"]
                pv = exp2(b.fsub(s[kk // 4][(kk % 4) * 4 + j2], st["m"]))
                l_acc[0] = b.fadd(l_acc[0], pv)
                return pv if PACK_P_BF16 else b.cast_f32_to(pv, dtype)

            def head():
                # V(j) LDS write first: its payload registers die early.
                for _ in v_store_steps:
                    yield
                s = [[None] * 16 for _ in range(N_SUB)]
                mx = neg_inf
                for nsub in range(N_SUB):
                    for i in range(16):
                        s[nsub][i] = b.fmul(b.vec_extract(s_acc[nsub], i), qk_scale)
                        if not masked:
                            mx = b.fmax(mx, s[nsub][i])
                        if i % 2 == 1:
                            yield
                if masked:
                    do_mask(s, j, lower=False, upper=True)
                    yield
                    for nsub in range(N_SUB):
                        for i in range(16):
                            mx = b.fmax(mx, s[nsub][i])
                        yield
                tile_max = b.fmax(mx, b.warp_shuffle_xor(mx, 32))
                m_new = b.fmax(m_i, tile_max)
                alpha = exp2(b.fsub(m_i, m_new))
                st.update(s=s, m=m_new, alpha=alpha)
                yield
                o = list(o_acc)
                for dt in range(D_TILES):
                    for half in range(2):
                        if half == 0:
                            elems = []
                        for i in range(8 * half, 8 * half + 8):
                            prod = b.fmul(b.vec_extract(o[dt], i), alpha)
                            if RESCALE_PLAIN:
                                # Opaque to SLP: no v_pk_mul_f32 pairing.
                                prod = b.optimization_barrier(prod)
                            elems.append(prod)
                        yield
                    o[dt] = b.vec_pack(elems, F32)
                st["o"] = o
                st["pack"] = make_pack([exp_elem(0, j2) for j2 in range(4)])
                yield
                if PACK_AHEAD:
                    packs = [st["pack"]]
                    for kk in range(1, PACK_AHEAD + 1):
                        elems = []
                        for j2 in range(4):
                            elems.append(exp_elem(kk, j2))
                            if j2 % 2 == 1:
                                yield
                        packs.append(make_pack(elems))
                    st["packs"] = packs

            hv = head()
            if PACK_AHEAD or QK_TAIL:
                # Pace the head evenly over the QK(j+1) MFMAs kept above the mid
                # barrier so none of it drains after them (the step count matches
                # head()'s yields).
                n_ticks = K_STEPS * N_SUB - QK_TAIL
                n_head = (
                    V_ITEMS + 8 * N_SUB + ((1 + N_SUB) if masked else 0)
                    + 1 + 2 * D_TILES + 1 + 2 * PACK_AHEAD
                )
                done = 0
                for i in range(n_ticks):
                    next(qk_next)
                    want = (i + 1) * n_head // n_ticks
                    while done < want:
                        next(hv, None)
                        done += 1
            else:
                for _ in qk_next:
                    next(hv, None)
            for _ in hv:
                pass
            if spec.qkla_rhythm_b:
                # 1 MFMA then N VALU, for every MFMA left above the mid barrier.
                for _ in range(K_STEPS * N_SUB - QK_TAIL):
                    b.sched_group_barrier(0x008, 1, 0)
                    b.sched_group_barrier(0x002, spec.qkla_rhythm_b, 0)
            # Keep the head (O rescale, exps) above the mid barrier: LLVM otherwise
            # sinks it into the first PV packs, where the PV MFMAs wait on it.
            b.sched_barrier(0)
            # Publish V(j): every wave's writes land before any wave's PV(j).
            b.sync_lds_only()

            o = st["o"]
            pack = st["pack"]
            vals = []
            v_steps = make_v_steps(vals)
            k_steps = make_k_steps()
            v_cur = [read_v(dt, 0) for dt in range(D_TILES)]
            if spec.qkla_pv_spread:
                # QK tail on every other PV MFMA; K DMA rows after the last one.
                tail_at = {2 * i + 1 for i in range(QK_TAIL)}
                k_lo = max(1, -(-(2 * QK_TAIL) // D_TILES))
                # exps of packs PA+1..KK-1 spread evenly over PV MFMA slots
                # [0, span); slot <= 4p-1 holds for every item of pack p.
                items = [
                    (pk, j2) for pk in range(PACK_AHEAD + 1, KK_STEPS) for j2 in range(4)
                ]
                span = D_TILES * (KK_STEPS - 1)
                exp_at = {}
                for i, it in enumerate(items):
                    exp_at.setdefault(i * span // len(items), []).append(it)
                packs = list(st["packs"]) if PACK_AHEAD else [st["pack"]]
                packs += [None] * (KK_STEPS - len(packs))
                pend = {}
                for kk in range(KK_STEPS):
                    nxt = kk + 1 < KK_STEPS
                    v_nxt = (
                        [read_v(dt, kk + 1) for dt in range(D_TILES)] if nxt else None
                    )
                    pack = packs[kk]
                    if pack is None:
                        raise AssertionError(f"pv_spread: pack {kk} not built in time")
                    for dt in range(D_TILES):
                        g = kk * D_TILES + dt
                        o[dt] = mfma_32x32x8_for_dtype(b, dtype, v_cur[dt], pack, o[dt])
                        if g in tail_at:
                            next(qk_next, None)
                        for pk, j2 in exp_at.get(g, ()):
                            pend.setdefault(pk, []).append(exp_elem(pk, j2))
                            if len(pend[pk]) == 4:
                                packs[pk] = make_pack(pend.pop(pk))
                        if k_lo <= kk <= k_lo + 1:
                            next(k_steps, None)
                        if kk == 3:
                            next(v_steps, None)
                        if spec.qkla_pv_fence == 2:
                            b.sched_barrier(0)
                    if spec.qkla_pv_fence == 1:
                        b.sched_barrier(0)
                    if nxt:
                        v_cur = v_nxt
            # K(j+2) DMA rows: PV packs 1..2, or right after the QK tail.
            k_lo = max(1, -(-QK_TAIL // D_TILES))
            for kk in (range(0) if spec.qkla_pv_spread else range(KK_STEPS)):
                nxt = kk + 1 < KK_STEPS
                ahead = nxt and kk + 1 <= PACK_AHEAD
                v_nxt = [read_v(dt, kk + 1) for dt in range(D_TILES)] if nxt else None
                elems = []
                for dt in range(D_TILES):
                    o[dt] = mfma_32x32x8_for_dtype(b, dtype, v_cur[dt], pack, o[dt])
                    if QK_TAIL and kk * D_TILES + dt < QK_TAIL:
                        next(qk_next, None)
                    if nxt and not ahead:
                        elems.append(exp_elem(kk + 1, dt))
                    if k_lo <= kk <= k_lo + 1:
                        next(k_steps, None)
                    if kk == 3:
                        next(v_steps, None)
                if nxt:
                    pack = st["packs"][kk + 1] if ahead else make_pack(elems)
                    v_cur = v_nxt
            for _ in k_steps:
                pass
            for _ in v_steps:
                pass
            for _ in qk_next:  # exhaust: publishes the S(j+1) accumulators
                pass
            l_tile = b.fadd(l_acc[0], b.warp_shuffle_xor(l_acc[0], 32))
            l_new = b.fadd(b.fmul(l_i, st["alpha"]), l_tile)
            return st["m"], l_new, o, vals

        # Estimated issue cycles per unit, used only to size qkla_chunked chunks.
        C_VST, C_MX, C_MSK, C_ALPHA, C_RS = 20, 2, 4, 28, 4
        C_EXP, C_PACK, C_VRD, C_KDMA, C_VLD, C_SSC = 24, 40, 8, 12, 16, 4
        N_KDMA = 8  # K DMA rows per wave per tile (D128 / BN64)
        QKLA_CARRY = 2  # rev 3: PV MFMAs (the last ones of the last pack) carried

        def carried_dts():
            return list(range(D_TILES - QKLA_CARRY, D_TILES))

        def zero_vpack():
            return b.bitcast(b.vec_pack([b.const_i32(0)] * 2, I32), VectorType(BF16, 4))

        def tile_chunked(j, m_i, l_i, o_acc, s_sc, v_store_steps, qk_next, s_out,
                         masked, make_v_steps, make_k_steps, pend=None, pend_sink=None,
                         v_pub=None, m_ahead=None, ma_sink=None):
            """qkla_chunked: tile j from its carried scaled scores ``s_sc``. Every
            MFMA is followed by a fenced chunk of about the region's mean estimated
            cycles. Returns (m_new, l_new, o_acc, v_next, s_next) with S(j+1) scaled."""
            exp2 = b.exp2_fast if spec.resolved_use_exp2_fast() else b.exp2

            def fence():
                # A hard scheduling boundary between chunks. s_setprio 0 keeps the
                # default wave priority; sched_barrier(0) alone did not hold.
                if spec.qkla_chunk_fence == 1:
                    b.s_setprio(0)
                else:
                    b.sched_barrier(0)

            PA = spec.qkla_pack_ahead
            QT = spec.qkla_qk_tail
            LAM = spec.qkla_lds_after_mfma
            EVEN = spec.qkla_even
            # qkla_even: B pacing costs = VALU issue cycles (V loads use no VALU)
            C_VST_B = 16 if EVEN else C_VST
            C_VLD_B = 4 if EVEN else C_VLD

            def lds_after_mfma():
                # this region: all VALU (addresses, perms), then the MFMA, then LDS
                b.sched_group_barrier(0x002, 64, 0)
                b.sched_group_barrier(0x008, 1, 0)
                b.sched_group_barrier(0x080, 16, 0)
            REV = spec.qkla_chunk_rev
            CARRY = REV == 3
            PLAN = REV in (2, 3, 4)  # fixed per-chunk PV plan
            DEFER_ADD = REV in (4, 5)  # row-sum add delayed past the next exp

            def s_scale(raw, i):
                x = b.vec_extract(raw, i)
                if REV == 5:
                    # side-effecting pin: the multiply stays in the chunk it was put in
                    # (plain fences let the compiler hoist it into an earlier chunk)
                    x = b.inline_asm("", "=v,0", [x], F32, sideeffect=True)
                return b.optimization_barrier(b.fmul(x, qk_scale))
            o_acc = list(o_acc)
            st = {}
            l_acc = [b.const_f32(0.0)]

            def make_pack(elems):
                return _pack_p_bf16(b, elems) if PACK_P_BF16 else b.vec_pack(elems, dtype)

            def exp_elem(kk, j2):
                sv = st["s"]
                # Pinned (opaque) so it stays in the chunk it was placed in.
                diff = b.fsub(sv[kk // 4][(kk % 4) * 4 + j2], st["m"])
                if REV >= 2 or spec.qkla_sub_pin:
                    # opaque: keeps two subtracts from pairing into v_pk_add_f32
                    diff = b.optimization_barrier(diff)
                pv = b.optimization_barrier(exp2(diff))
                if DEFER_ADD and st.get("defer_add"):
                    # rev 4: add the PREVIOUS exp into the row sum now (same order),
                    # so no VALU reads this exp's result right after it issues
                    if st.get("pend_add") is not None:
                        l_acc[0] = b.fadd(l_acc[0], st["pend_add"])
                    st["pend_add"] = pv
                else:
                    l_acc[0] = b.fadd(l_acc[0], pv)
                return pv if PACK_P_BF16 else b.cast_f32_to(pv, dtype)

            # ---- above the mid barrier: head paced over the QK(j+1) MFMAs ----
            vals = []
            v_steps = make_v_steps(vals)
            k_steps = make_k_steps()
            LB_V = spec.qkla_loads_b
            # qkla_max_ahead: carried S(j) row max in (unmasked tiles), S(j+1) max out
            USE_MA_IN = m_ahead if (m_ahead is not None and not (masked and causal)) else None
            MA_OUT = ma_sink is not None and not masked
            ma = {"acc": neg_inf, "i": 0, "done": False}

            def head():
                for _ in v_store_steps:
                    yield C_VST
                sv = [[b.vec_extract(s_sc[n], i) for i in range(16)] for n in range(N_SUB)]
                mx = neg_inf
                MSK = masked and causal
                if MSK:
                    # do_mask's causal bound, one element at a time (same selects).
                    tile_key0 = b.mul(j, b.const_i32(BN))
                    query_tok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
                    if BR:
                        query_tok = b.add(query_tok, diag_off)
                for n in range(N_SUB if USE_MA_IN is None else 0):
                    if MSK:
                        sub_base = b.add(tile_key0, b.const_i32(n * 32))
                    for i in range(16):
                        if MSK:
                            ktok = b.add(sub_base, _mfma_32x32_c_row(b, lane, i))
                            sv[n][i] = b.select(b.cmp_le(ktok, query_tok), sv[n][i], neg_inf)
                        mx = b.fmax(mx, sv[n][i])
                        if not REV:
                            yield C_MX + (C_MSK if MSK else 0)
                        elif i % 2 == 1:
                            # Two elements per unit (one v_max3); the opaque running
                            # max keeps max3 fusion from pulling it into the next chunk.
                            yield 2 * (C_MX + (C_MSK if MSK else 0))

                tile_max = (USE_MA_IN if USE_MA_IN is not None
                            else b.fmax(mx, b.warp_shuffle_xor(mx, 32)))
                m_new = b.fmax(m_i, tile_max)
                alpha = exp2(b.fsub(m_i, m_new))
                st.update(s=sv, m=m_new, alpha=alpha)
                yield C_ALPHA
                if EVEN and spec.qkla_even_mix:
                    yield from mixed_rescale_packs(alpha)
                    return
                o = list(o_acc)
                for dt in range(D_TILES):
                    elems = []
                    for i in range(16):
                        # Opaque to SLP: plain v_mul_f32 (a v_pk_mul_f32 cannot issue
                        # while this wave's own MFMA holds the pipe).
                        elems.append(
                            b.optimization_barrier(b.fmul(b.vec_extract(o[dt], i), alpha))
                        )
                        yield C_RS
                    o[dt] = b.vec_pack(elems, F32)
                st["o"] = o
                packs = []
                for kk in range(PA + 1):
                    if EVEN:
                        # interleaved across the pack's 4 elements: no exp hazards
                        ds = []
                        for j2 in range(4):
                            ds.append(b.optimization_barrier(
                                b.fsub(sv[kk // 4][(kk % 4) * 4 + j2], st["m"])))
                            yield 4
                        ps = []
                        for j2 in range(4):
                            ps.append(b.optimization_barrier(exp2(ds[j2])))
                            yield 16
                        rr = []
                        for j2 in range(4):
                            l_acc[0] = b.fadd(l_acc[0], ps[j2])
                            yield 4
                        for j2 in range(4):
                            rr.append(b.optimization_barrier(_bf16_round_bits(ps[j2])))
                            yield 8
                        words = []
                        for h2 in range(2):
                            words.append(b.perm_b32(rr[2 * h2], rr[2 * h2 + 1],
                                                    b.const_i32(0x03020706)))
                            yield 4
                        packs.append(b.bitcast(b.vec_pack(words, I32), VectorType(BF16, 4)))
                        continue
                    elems = []
                    for j2 in range(4):
                        elems.append(exp_elem(kk, j2))
                        yield C_EXP
                    packs.append(make_pack(elems))
                    yield C_PACK
                st["packs"] = packs

            def mixed_rescale_packs(alpha):
                """qkla_even_mix: O rescale and pack-ahead stages merged by fraction done."""
                sv = st["s"]
                o = list(o_acc)
                def g_rs():
                    for dt in range(D_TILES):
                        elems = []
                        for i in range(16):
                            elems.append(b.optimization_barrier(
                                b.fmul(b.vec_extract(o[dt], i), alpha)))
                            yield 4
                        o[dt] = b.vec_pack(elems, F32)
                    st["o"] = o
                packs = []
                def g_pk():
                    for kk in range(PA + 1):
                        ds = []
                        for j2 in range(4):
                            ds.append(b.optimization_barrier(
                                b.fsub(sv[kk // 4][(kk % 4) * 4 + j2], st["m"])))
                            yield 4
                        ps = []
                        for j2 in range(4):
                            ps.append(b.optimization_barrier(exp2(ds[j2])))
                            yield 16
                        for j2 in range(4):
                            l_acc[0] = b.fadd(l_acc[0], ps[j2])
                            yield 4
                        rr = [b.optimization_barrier(_bf16_round_bits(ps[j2])) for j2 in range(2)]
                        yield 16
                        rr += [b.optimization_barrier(_bf16_round_bits(ps[j2])) for j2 in range(2, 4)]
                        yield 16
                        words = [b.perm_b32(rr[2 * h2], rr[2 * h2 + 1], b.const_i32(0x03020706))
                                 for h2 in range(2)]
                        yield 8
                        packs.append(b.bitcast(b.vec_pack(words, I32), VectorType(BF16, 4)))
                    st["packs"] = packs
                gens = [[g_rs(), 0, D_TILES * 16 * 4], [g_pk(), 0, (PA + 1) * 136]]
                while gens:
                    g = min(gens, key=lambda x: x[1] / x[2])
                    c_ = next(g[0], None)
                    if c_ is None:
                        gens.remove(g)
                        continue
                    g[1] += c_
                    yield c_

            def _bf16_round_bits(e):
                bits = b.bitcast(e, I32)
                lsb = b.land(b.lshr(bits, b.const_i32(16)), b.const_i32(1))
                return b.add(b.add(bits, b.const_i32(0x7FFF)), lsb)

            def pv_even():
                """qkla_even: PV(j) chunks filled to ~spec.qkla_even VALU cycles by an
                earliest-deadline-first pick over fine-grained units."""
                T = spec.qkla_even
                o = st["o"]
                packs = st["packs"] + [None] * (KK_STEPS - PA - 1)
                n_s = KK_STEPS * D_TILES
                vr = {0: st.pop("vr0") if "vr0" in st else [read_v(dt, 0) for dt in range(D_TILES)]}
                next(qk_next, None)  # flush: QK(j+1) finished above the barrier
                s_raw = list(s_out)
                sv = st["s"]
                s_flat = [None] * (N_SUB * 16)
                U = []  # units: dict(cost, dead, deps, fn)

                def add_u(cost, dead, deps, fn):
                    U.append(dict(cost=cost, dead=dead, deps=deps, fn=fn, slot=None))
                    return len(U) - 1
                val = {}
                for pk in range(PA + 1, KK_STEPS):
                    dl = D_TILES * pk - 1  # pack pk complete one MFMA before its first use
                    rid = []
                    for j2 in range(4):
                        key = (pk, j2)
                        def f_sub(key=key):
                            pk_, j_ = key
                            val[("d",) + key] = b.optimization_barrier(
                                b.fsub(sv[pk_ // 4][(pk_ % 4) * 4 + j_], st["m"]))
                        def f_exp(key=key):
                            val[("p",) + key] = b.optimization_barrier(exp2(val[("d",) + key]))
                        def f_add(key=key):
                            l_acc[0] = b.fadd(l_acc[0], val[("p",) + key])
                        def f_rnd(key=key):
                            val[("r",) + key] = b.optimization_barrier(
                                _bf16_round_bits(val[("p",) + key]))
                        u_sub = add_u(4, dl - 3, [], f_sub)
                        u_exp = add_u(16, dl - 2, [u_sub], f_exp)
                        add_u(4, n_s - 1, [u_exp], f_add)
                        rid.append(add_u(8, dl - 1, [u_exp], f_rnd))
                    for h2 in range(2):
                        def f_word(pk=pk, h2=h2):
                            w = b.perm_b32(val[("r", pk, 2 * h2)], val[("r", pk, 2 * h2 + 1)],
                                           b.const_i32(0x03020706))
                            val[("w", pk, h2)] = w
                            if ("w", pk, 1 - h2) in val:
                                ws = [val[("w", pk, 0)], val[("w", pk, 1)]]
                                packs[pk] = b.bitcast(b.vec_pack(ws, I32), VectorType(BF16, 4))
                        add_u(4, dl, [rid[2 * h2], rid[2 * h2 + 1]], f_word)
                for r in range(N_KDMA):
                    add_u(8, min(n_s - 1, 2 * r + 1), [], lambda: next(k_steps, None))
                for e in range(N_SUB * 16):
                    def f_ssc(e=e):
                        n, i = divmod(e, 16)
                        s_flat[e] = b.optimization_barrier(
                            b.fmul(b.vec_extract(s_raw[n], i), qk_scale))
                    add_u(4, n_s - 1, [], f_ssc)
                done_u = set()
                for si in range(n_s):
                    kk, dt = divmod(si, D_TILES)
                    if packs[kk] is None:
                        raise AssertionError(f"qkla_even: pack {kk} late")
                    o[dt] = mfma_32x32x8_for_dtype(b, dtype, vr[kk][dt], packs[kk], o[dt])
                    if dt == 0 and kk + 1 < KK_STEPS:
                        vr[kk + 1] = [read_v(d, kk + 1) for d in range(D_TILES)]
                        if LAM:
                            lds_after_mfma()
                    fence()
                    # even spread: this chunk takes (remaining work) / (remaining chunks)
                    rem = sum(u["cost"] for u in U if u["slot"] is None)
                    cap = rem / (n_s - si)
                    ready = [i for i, u in enumerate(U) if u["slot"] is None
                             and all(U[d]["slot"] is not None and U[d]["slot"] < si for d in u["deps"])]
                    ready.sort(key=lambda i: (U[i]["dead"], i))
                    took = 0
                    for i in ready:
                        u = U[i]
                        must = u["dead"] <= si
                        if must or (took < cap and took + u["cost"] / 2 <= cap) or took == 0:
                            u["fn"]()
                            u["slot"] = si
                            done_u.add(i)
                            took += u["cost"]
                    if v_steps is not None and not LB_V and kk == 3:
                        next(v_steps, None)
                    fence()
                left = [i for i, u in enumerate(U) if u["slot"] is None]
                guard = 0
                while left and guard < 8:
                    for i in left:
                        u = U[i]
                        if all(U[d]["slot"] is not None for d in u["deps"]):
                            u["fn"]()
                            u["slot"] = n_s
                    left = [i for i, u in enumerate(U) if u["slot"] is None]
                    guard += 1
                if left:
                    raise AssertionError("qkla_even: unscheduled units")
                for _ in k_steps:
                    pass
                s_next = [b.vec_pack(s_flat[16 * n : 16 * n + 16], F32) for n in range(N_SUB)]
                l_tile = (l_acc[0] if spec.qkla_lhalf
                          else b.fadd(l_acc[0], b.warp_shuffle_xor(l_acc[0], 32)))
                l_new = b.fadd(b.fmul(l_i, st["alpha"]), l_tile)
                return st["m"], l_new, o, vals, s_next

            def pv_planned():
                """rev 2 (no QK tail): a fixed per-chunk plan for PV(j). exps spread
                evenly; each pair's bf16 word one chunk after its second exp (before
                the pack's deadline); K(j+2) DMA rows on chunks 0..7; V reads two packs
                ahead; S(j+1)'s scale fills the lightest chunks."""
                o = st["o"]
                packs = st["packs"] + [None] * (KK_STEPS - PA - 1)
                n_s = KK_STEPS * D_TILES - (QKLA_CARRY if CARRY else 0)
                last = n_s - 1
                VRAP = 1 if CARRY else 2
                plan = [[] for _ in range(n_s)]
                cost = [0] * n_s
                def put(si, unit, c):
                    plan[si].append(unit)
                    cost[si] += c
                items = [(pk, j2) for pk in range(PA + 1, KK_STEPS) for j2 in range(4)]
                span = D_TILES * (KK_STEPS - 1) - 2  # leave room for the last word
                for i, it in enumerate(items):
                    put(i * span // len(items), ("exp",) + it, C_EXP)
                exp_at = {}
                for si, units in enumerate(plan):
                    for u in units:
                        exp_at[(u[1], u[2])] = si
                for pk in range(PA + 1, KK_STEPS):
                    for h in range(2):
                        si = max(exp_at[(pk, 2 * h)], exp_at[(pk, 2 * h + 1)]) + 1
                        si = min(si, D_TILES * pk - 1)
                        put(si, ("word", pk, h), C_PACK // 2)
                for r in range(N_KDMA):
                    put(min(r, last), ("kdma",), 24)  # measured: ~5 instr per row
                # V reads: rev 4 reads pack p in two halves on separate chunks
                vhalf = {}  # chunk -> list of (pack, half)
                if REV == 4:
                    for pk in range(1, KK_STEPS):
                        if pk == 1:
                            slots_h = [None, 0]  # half 0 before PV, half 1 on chunk 0
                        else:
                            slots_h = [D_TILES * (pk - 2) + 1, D_TILES * (pk - 2) + 3]
                        for h, si in enumerate(slots_h):
                            if si is not None:
                                vhalf.setdefault(si, []).append((pk, h))
                                cost[si] += C_VRD // 2
                else:
                    for kk in range(KK_STEPS - VRAP):
                        cost[D_TILES * kk] += C_VRD
                for e in range(N_SUB * 16):
                    si = min(range(n_s), key=lambda k: (cost[k], k))
                    put(si, ("ssc", e), C_SSC)
                vr = {0: [read_v(dt, 0) for dt in range(D_TILES)]}
                HALF = D_TILES // 2
                if REV == 4:
                    vr[1] = [read_v(dt, 1) if dt < HALF else None for dt in range(D_TILES)]
                    st["defer_add"] = True
                    st["pend_add"] = None
                elif VRAP == 2 and KK_STEPS > 1:
                    vr[1] = [read_v(dt, 1) for dt in range(D_TILES)]
                next(qk_next, None)  # flush: QK(j+1) finished above the barrier
                s_raw = list(s_out)
                s_flat = [None] * (N_SUB * 16)
                ev = {}
                wl = {}
                vals_pv = []
                v_steps_pv = None if LB_V else make_v_steps(vals_pv)
                for si in range(n_s):
                    kk, dt = divmod(si, D_TILES)
                    if REV == 4:
                        for pk, h in vhalf.get(si, []):
                            row = vr.setdefault(pk, [None] * D_TILES)
                            for d in range(h * HALF, (h + 1) * HALF):
                                row[d] = read_v(d, pk)
                    elif dt == 0 and kk + VRAP < KK_STEPS:
                        vr[kk + VRAP] = [read_v(d, kk + VRAP) for d in range(D_TILES)]
                    if vr[kk][dt] is None:
                        raise AssertionError(f"qkla_chunked rev4: V read for pack {kk} late")
                    if packs[kk] is None:
                        raise AssertionError(f"qkla_chunked rev2: pack {kk} late")
                    o[dt] = mfma_32x32x8_for_dtype(b, dtype, vr[kk][dt], packs[kk], o[dt])
                    fence()
                    for u in plan[si]:
                        if u[0] == "exp":
                            ev[(u[1], u[2])] = exp_elem(u[1], u[2])
                        elif u[0] == "word":
                            pk, h = u[1], u[2]
                            if PACK_P_BF16:
                                wl.setdefault(pk, []).append(b.optimization_barrier(
                                    _pack_p_bf16_word(b, ev[(pk, 2 * h)], ev[(pk, 2 * h + 1)])
                                ))
                                if len(wl[pk]) == 2:
                                    packs[pk] = b.bitcast(
                                        b.vec_pack(wl.pop(pk), I32), VectorType(BF16, 4)
                                    )
                            elif h == 1:
                                packs[pk] = make_pack([ev[(pk, x)] for x in range(4)])
                        elif u[0] == "kdma":
                            next(k_steps, None)
                        elif u[0] == "ssc":
                            n, i = divmod(u[1], 16)
                            s_flat[u[1]] = b.optimization_barrier(
                                b.fmul(b.vec_extract(s_raw[n], i), qk_scale)
                            )
                    if v_steps_pv is not None and kk == 3:
                        next(v_steps_pv, None)
                    fence()
                for _ in k_steps:
                    pass
                if v_steps_pv is not None:
                    for _ in v_steps_pv:
                        pass
                s_next = [b.vec_pack(s_flat[16 * n : 16 * n + 16], F32) for n in range(N_SUB)]
                if st.get("pend_add") is not None:
                    l_acc[0] = b.fadd(l_acc[0], st["pend_add"])
                    st["pend_add"] = None
                l_tile = (l_acc[0] if spec.qkla_lhalf
                          else b.fadd(l_acc[0], b.warp_shuffle_xor(l_acc[0], 32)))
                l_new = b.fadd(b.fmul(l_i, st["alpha"]), l_tile)
                if CARRY:
                    if packs[KK_STEPS - 1] is None:
                        raise AssertionError("qkla_chunked rev3: last pack not built")
                    st["pend"] = [vr[KK_STEPS - 1][d] for d in carried_dts()] + [
                        packs[KK_STEPS - 1]
                    ]
                return st["m"], l_new, o, (vals_pv if v_steps_pv is not None else vals), s_next

            n_b = K_STEPS * N_SUB - QT + (QKLA_CARRY if CARRY else 0)
            # K(j+2) DMA stays below the barrier: a QK(j+1) K read after a K DMA row
            # waits vmcnt(0) (the scopes cannot tell the two slots apart).
            LB_K = False
            NVST = 0 if VDB else V_ITEMS  # qkla_vdb: no V(j) stores in B
            VST_B = v_pub is not None and spec.qkla_vdb == 2
            vst_b = v_pub(vals) if VST_B else None
            tot_b = (
                NVST * C_VST_B + (N_SUB * 16 * C_MSK if masked and causal else 0)
                + (N_SUB * 16 * C_MX if USE_MA_IN is None else 0) + C_ALPHA + D_TILES * 16 * C_RS
                + (PA + 1) * (4 * C_EXP + C_PACK)
                + (V_ITEMS * C_VLD_B if LB_V else 0) + (N_KDMA * C_KDMA if LB_K else 0)
                + (V_ITEMS * C_VST_B if VST_B else 0)
            )
            if spec.qkla_prio_bal:
                b.s_setprio(spec.qkla_prio_bal)  # QK(j+1) region: high priority
            hv = head()
            done = 0
            # head() yields: V stores, max pairs, alpha, rescale elements, pack exps+builds
            n_units = (NVST + (N_SUB * (8 if REV else 16) if USE_MA_IN is None else 0)
                       + 1 + D_TILES * 16 + (PA + 1) * (18 if EVEN else 5))
            used = 0
            if CARRY:
                next(qk_next)  # rev 3: QK(j+1)'s first K reads, no MFMA yet
                cdts = carried_dts()
            for i in range(n_b):
                if CARRY and i < QKLA_CARRY:
                    # rev 3: tile j-1's last PV MFMAs (registers only), before the
                    # head rescales O
                    dtc = cdts[i]
                    o_acc[dtc] = mfma_32x32x8_for_dtype(b, dtype, pend[i], pend[-1], o_acc[dtc])
                else:
                    next(qk_next)
                if LAM and i >= 1:
                    # V(j+1) stores join the MFMA region (their perms/addresses go
                    # before the MFMA, the ds_writes right after it)
                    if VST_B and i >= n_b // 2 and next(vst_b, "end") != "end":
                        done += C_VST_B
                    if i == n_b - 1:
                        st["vr0"] = [read_v(dt, 0) for dt in range(D_TILES)]
                    lds_after_mfma()
                fence()  # the MFMA stays at the head of its chunk
                # V(j+1) loads: rev < 2 one per chunk from the tile start; rev 2
                # spread evenly over B (they are only needed by the next tile)
                VS0, VSS = spec.qkla_vld_start, spec.qkla_vld_stride
                if LB_V and i >= VS0 and (i - VS0) % VSS == 0 and next(v_steps, "end") != "end":
                    done += C_VLD_B
                # qkla_vdb 2: V(j+1) -> slot (j+1)%2, one item per chunk in B's
                # 2nd half (its loads were issued in the first chunks)
                if VST_B and not LAM and i >= n_b // 2 and next(vst_b, "end") != "end":
                    done += C_VST_B
                first = bool(REV)
                # at least one unit per chunk, and never so many that fewer units
                # remain than MFMAs still to cover
                while first or (
                    done < (i + 1) * tot_b / n_b
                    and (not REV or n_units - used > n_b - i - 1)
                ):
                    first = False
                    c = next(hv, None)
                    if c is None:
                        break
                    done += c
                    used += 1
                fence()
            for _ in hv:
                pass
            if VST_B:
                for _ in vst_b:
                    pass
            b.sched_barrier(0)
            if spec.qkla_prio_bal:
                b.s_setprio(0)  # PV(j): yield arbitration to a wave still in QK(j+1)
            # Publish V(j): every wave's writes land before any wave's PV(j).
            # qkla_vdb: V(j) was published by the previous end barrier.
            if not VDB:
                b.sync_lds_only()

            # ---- below the mid barrier: PV + QK tail MFMAs, fenced chunks ----
            if EVEN and QT == 0:
                return pv_even()
            if PLAN and QT == 0:
                res = pv_planned()
                if CARRY and pend_sink is not None:
                    pend_sink.extend(st["pend"])
                return res
            o = st["o"]
            packs = st["packs"] + [None] * (KK_STEPS - PA - 1)
            slots = []
            for kk in range(KK_STEPS):
                for dt in range(D_TILES):
                    slots.append(("pv", kk, dt))
                    if kk * D_TILES + dt < 2 * QT and (kk * D_TILES + dt) % 2 == 0:
                        slots.append(("qk",))
            last_qk = max([i for i, x in enumerate(slots) if x[0] == "qk"], default=-1)
            first_of = {x[1]: i for i, x in reversed(list(enumerate(slots))) if x[0] == "pv"}
            items = [(pk, j2) for pk in range(PA + 1, KK_STEPS) for j2 in range(4)]
            # qkla_vdb: V(j+1) -> its own slot, one item per chunk in PV's 2nd half
            vst = v_pub(vals) if v_pub is not None and spec.qkla_vdb == 1 else None
            vst_live = [vst is not None]
            tot_pv = (
                len(items) * C_EXP + (KK_STEPS - PA - 1) * C_PACK
                + (KK_STEPS - 1) * C_VRD + (0 if LB_K else N_KDMA * C_KDMA)
                + (0 if LB_V else V_ITEMS * C_VLD) + N_SUB * 16 * C_SSC
                + (V_ITEMS * C_VST if vst is not None else 0)
                + (N_SUB * 16 * C_MX + 8 if MA_OUT else 0)
            )
            tgt = tot_pv / len(slots)
            s_raw = []
            s_flat = []
            vr = {0: st.pop("vr0") if "vr0" in st else [read_v(dt, 0) for dt in range(D_TILES)]}
            VRA = 2 if PLAN else 1  # V reads this many packs ahead
            if DEFER_ADD:
                st["defer_add"] = True
                st["pend_add"] = None
            if VRA == 2 and KK_STEPS > 1:
                vr[1] = [read_v(dt, 1) for dt in range(D_TILES)]
            ei = 0
            pend = []
            tail_done = last_qk < 0
            if tail_done:
                next(qk_next, None)  # flush: QK(j+1) finished above the barrier
            k_live = [True]
            v_live = [True]
            done = 0

            words = []
            ready = []  # rev 2: (pack, e0, e1) pairs whose rounding is not yet emitted
            wpk = {}

            def build_word(pk, e0, e1):
                wpk.setdefault(pk, []).append(
                    b.optimization_barrier(_pack_p_bf16_word(b, e0, e1))
                )
                if len(wpk[pk]) == 2:
                    packs[pk] = b.bitcast(b.vec_pack(wpk.pop(pk), I32), VectorType(BF16, 4))
                return C_PACK // 2

            def flush_pack(pk):
                c = 0
                while ready and ready[0][0] <= pk:
                    c += build_word(*ready.pop(0))
                return c

            def emit_exp():
                nonlocal ei
                pk, j2 = items[ei]
                ei += 1
                e = exp_elem(pk, j2)
                c = C_EXP
                if REV >= 2 and PACK_P_BF16:
                    # round the previous pair now, after this exp: its exp results
                    # have had time to finish
                    if ready:
                        c += build_word(*ready.pop(0))
                    pend.append(e)
                    if len(pend) == 2:
                        ready.append((pk, pend[0], pend[1]))
                        pend.clear()
                    return c
                pend.append(e)
                if PACK_P_BF16 and len(pend) == 2:
                    # Half of the pack: one word now, so the chain is not one lump.
                    words.append(
                        b.optimization_barrier(_pack_p_bf16_word(b, pend[0], pend[1]))
                    )
                    pend.clear()
                    c += C_PACK // 2
                    if len(words) == 2:
                        packs[pk] = b.bitcast(
                            b.vec_pack(list(words), I32), VectorType(BF16, 4)
                        )
                        words.clear()
                elif not PACK_P_BF16 and len(pend) == 4:
                    packs[pk] = make_pack(list(pend))
                    pend.clear()
                    c += C_PACK
                return c

            def emit_load(gen, live, cost):
                if live[0] and next(gen, "end") != "end":
                    return cost
                live[0] = False
                return 0

            def ma_unit():
                """qkla_max_ahead: one v_max3 pair of S(j+1) (once scaled), then the
                cross-half shuffle; 0 when nothing is ready."""
                if not MA_OUT or ma["done"]:
                    return 0
                if ma["i"] + 2 <= len(s_flat):
                    x0, x1 = s_flat[ma["i"]], s_flat[ma["i"] + 1]
                    ma["acc"] = b.optimization_barrier(b.fmax(b.fmax(ma["acc"], x0), x1))
                    ma["i"] += 2
                    return 2 * C_MX
                if ma["i"] == N_SUB * 16:
                    ma["out"] = b.fmax(ma["acc"], b.warp_shuffle_xor(ma["acc"], 32))
                    ma["done"] = True
                    return 8
                return 0

            for si, slot in enumerate(slots):
                if slot[0] == "pv":
                    kk, dt = slot[1], slot[2]
                    if dt == 0 and kk + VRA < KK_STEPS:
                        vr[kk + VRA] = [read_v(d, kk + VRA) for d in range(D_TILES)]
                        done += C_VRD
                    if packs[kk] is None and REV >= 2:
                        done += flush_pack(kk)
                    if packs[kk] is None:
                        raise AssertionError(f"qkla_chunked: pack {kk} late")
                    o[dt] = mfma_32x32x8_for_dtype(b, dtype, vr[kk][dt], packs[kk], o[dt])
                    if LAM and dt == 0 and kk + VRA < KK_STEPS:
                        lds_after_mfma()
                else:
                    next(qk_next)
                fence()  # the MFMA stays at the head of its chunk
                if si == last_qk:
                    next(qk_next, None)  # flush the S(j+1) accumulators
                    tail_done = True
                if tail_done and not s_raw:
                    s_raw = list(s_out)
                done_before = done
                # K(j+2) DMA: one row per chunk as soon as the tail's K reads are
                # done, so it lands well before the end barrier's vmcnt(0).
                if REV and tail_done and k_live[0]:
                    done += emit_load(k_steps, k_live, C_KDMA)
                if vst_live[0] and si >= len(slots) // 2:
                    done += emit_load(vst, vst_live, C_VST)
                # exps whose pack the next slot needs
                while ei < len(items):
                    pk = items[ei][0]
                    if first_of[pk] > si + 1:
                        break
                    done += emit_exp()
                # keep the exps on an even schedule up to the last pack's deadline
                due = min(len(items), -(-(si + 1) * len(items) // first_of[KK_STEPS - 1]))
                while ei < due:
                    done += emit_exp()
                if REV >= 2 and si + 1 < len(slots) and slots[si + 1][0] == "pv":
                    nk = slots[si + 1][1]
                    if packs[nk] is None:
                        # finish the next MFMA's pack one chunk ahead of it
                        done += flush_pack(nk)
                if REV >= 2 and ei >= len(items) and tail_done and s_raw:
                    # rev 2: S(j+1)'s scale split evenly over the remaining chunks
                    left = N_SUB * 16 - len(s_flat)
                    take = -(-left // (len(slots) - si)) if left else 0
                    for _ in range(take):
                        n, i = divmod(len(s_flat), 16)
                        s_flat.append(
                            s_scale(s_raw[n], i)
                        )
                        done += C_SSC
                # rev 1: K DMA handled above; V loads were issued in B
                one_load = bool(REV)

                def next_cost():
                    if ei < len(items):
                        return C_EXP + (C_PACK // 2 if PACK_P_BF16 and len(pend) == 1 else 0)
                    return C_SSC

                # add a unit only while it brings the chunk closer to the target
                while done + next_cost() / 2 < (si + 1) * tgt:
                    if not one_load and tail_done and k_live[0]:
                        c = emit_load(k_steps, k_live, C_KDMA)
                        if c:
                            done += c
                            one_load = True
                            continue
                    if not one_load and v_live[0]:
                        c = emit_load(v_steps, v_live, C_VLD)
                        if c:
                            done += c
                            one_load = True
                            continue
                    if ei < len(items):
                        done += emit_exp()
                        continue
                    if tail_done and len(s_flat) < N_SUB * 16 and s_raw:
                        n, i = divmod(len(s_flat), 16)
                        s_flat.append(
                            s_scale(s_raw[n], i)
                        )
                        done += C_SSC
                        continue
                    c_ma = ma_unit()
                    if c_ma:
                        done += c_ma
                        continue
                    c = (emit_load(k_steps, k_live, C_KDMA) if tail_done else 0) or emit_load(
                        v_steps, v_live, C_VLD
                    )
                    if not c:
                        break
                    done += c
                if REV and done == done_before:
                    # at least one unit per chunk: no MFMA without work after it
                    if ei < len(items):
                        done += emit_exp()
                    elif tail_done and len(s_flat) < N_SUB * 16 and s_raw:
                        n, i = divmod(len(s_flat), 16)
                        s_flat.append(
                            s_scale(s_raw[n], i)
                        )
                        done += C_SSC
                    else:
                        done += ma_unit()
                fence()
            while ei < len(items):
                emit_exp()
            flush_pack(KK_STEPS)
            for _ in k_steps:
                pass
            for _ in v_steps:
                pass
            if vst is not None:
                for _ in vst:
                    pass
            if not s_raw:
                s_raw = list(s_out)
            while len(s_flat) < N_SUB * 16:
                n, i = divmod(len(s_flat), 16)
                s_flat.append(
                    s_scale(s_raw[n], i)
                )
            while ma_unit():
                pass
            if MA_OUT:
                ma_sink.append(ma["out"])
            s_next = [b.vec_pack(s_flat[16 * n : 16 * n + 16], F32) for n in range(N_SUB)]
            if st.get("pend_add") is not None:
                l_acc[0] = b.fadd(l_acc[0], st["pend_add"])
                st["pend_add"] = None
            l_tile = (l_acc[0] if spec.qkla_lhalf
                      else b.fadd(l_acc[0], b.warp_shuffle_xor(l_acc[0], 32)))
            l_new = b.fadd(b.fmul(l_i, st["alpha"]), l_tile)
            return st["m"], l_new, o, vals, s_next

        def tile_compute(j, carry, masked):
            """Score, mask, online-softmax and PV for one staged tile; returns
            ``(m, l, *O)``."""
            m_i = carry[0]
            l_i = carry[1]
            o_acc = list(carry[2 : 2 + D_TILES])
            s = do_qk()
            m_new, l_new, o_acc = softmax_pv(s, j, m_i, l_i, o_acc, masked)
            return (m_new, l_new, *o_acc)

        # Wave-uniform skip of (wave, tile) pairs a sliding window fully masks.
        WAVE_SKIP = spec.window_wave_skip
        _skip_ctr = [0]

        def tile_body(j, carry, masked):
            m_i = carry[0]
            l_i = carry[1]
            o_acc = list(carry[2 : 2 + D_TILES])

            if spec.iglp:
                # Runbook lever 7: one canned-scheduler hint at the loop-body top.
                # Only meaningful on the cfvst path (in-loop ds_write to interleave).
                b.iglp_opt(spec.iglp_mode)

            if USE_CFVST:
                load_tile(j)  # K async DMA -> K_lds
                v_payload = _cfvst_load_v(
                    j
                )  # V register buffer_loads (natural [token,dim])
                # Drain BOTH the K async DMA and the V register loads (all vmcnt)
                # before the V perm+store reads those registers and before QK reads
                # K_lds.
                b.s_waitcnt(vmcnt=0)
                _cfvst_store_v(v_payload)  # perm_b32 -> ds_write V_lds[dim,token]
                # Publish K_lds (DMA) AND V_lds (ds_write) to all waves. This MUST
                # drain lgkmcnt for the V ds_write and cannot be a bare s_barrier:
                # gfx90a+ set FeatureBackOffBarrier, so LLVM would sink the lgkm wait
                # past the barrier to the ds_read consumer (the P0 read-after-barrier
                # race, now on the tile-START barrier because V publication is an
                # in-loop ds_write). sync_lds_only() emits the lgkm drain BEFORE the
                # barrier (CK Tile block_sync_lds). vmcnt is already 0, no full sync.
                b.sync_lds_only()
            else:
                # Naive (D64): K and V both async-DMA'd into their natural layout.
                # Drain the DMA (vmcnt) then a plain barrier -- V is not an in-loop
                # ds_write here, so the tile-start barrier needs no lgkm drain (the
                # DMA landed the data; the trailing sync_lds_only guards the
                # read-before-overwrite).
                load_tile(j)
                b.s_waitcnt(vmcnt=0)
                b.s_barrier_bare()

            if WAVE_SKIP:
                # Sliding window: skip the whole score/softmax/PV step for a wave
                # whose 32 query rows see no key of this tile -- every key is past
                # the causal diagonal or behind the window for all of them. The
                # running (m, l, O) state is exactly what a fully masked tile leaves.
                # The 0/1-trip loop is wave-uniform (readfirstlane), so the K/V
                # staging above and the trailing barrier stay in every wave.
                key0 = b.mul(j, b.const_i32(BN))
                past_diag = b.cmp_gt(key0, b.add(q_tok0, b.const_i32(31)))
                behind_window = b.cmp_le(
                    b.add(key0, b.const_i32(BN - 1)), b.sub(q_tok0, b.const_i32(SW))
                )
                trips = b.readfirstlane(
                    b.select(
                        b.lor(past_diag, behind_window), b.const_i32(0), b.const_i32(1)
                    )
                )
                _skip_ctr[0] += 1
                tag = _skip_ctr[0]
                sk = b.scf_for_iter(
                    b.const_i32(0),
                    trips,
                    b.const_i32(1),
                    [(f"wm{tag}", m_i), (f"wl{tag}", l_i)]
                    + [(f"wo{dt}_{tag}", o_acc[dt]) for dt in range(D_TILES)],
                    iv_name=f"ws{tag}",
                )
                with sk as (_iv, sc):
                    b.scf_yield(*tile_compute(j, sc, masked))
                m_new, l_new = sk.results[0], sk.results[1]
                o_acc = list(sk.results[2 : 2 + D_TILES])
            else:
                m_new, l_new, *o_acc = tile_compute(j, carry, masked)
            # Tile done; the next iteration refills the SINGLE K/V buffer (K via async
            # DMA, V via the perm_b32 ds_write), so every wave's LDS reads must have
            # LANDED (not just issued) before any wave starts writing. This MUST drain
            # lgkmcnt and cannot be a bare s_barrier: gfx90a+ set FeatureBackOffBarrier,
            # so SIInsertWaitcnts skips the conservative pre-barrier drain and places
            # the lgkm wait at the ds_read's CONSUMER -- which the scheduler is free to
            # sink past the barrier. Measured before the fix: ds_read_u16 x4 ->
            # s_barrier -> s_waitcnt lgkmcnt(0). sync_lds_only() emits the drain BEFORE
            # the barrier (CK Tile block_sync_lds); real LDS ops (mayLoad/mayStore)
            # cannot be scheduled across the side-effecting s_waitcnt, so the window is
            # structurally closed. vmcnt is deliberately NOT drained: it is already 0
            # (the tile-start s_waitcnt(vmcnt=0) drained this tile's DMA and NBUF=1
            # issues no new DMA until the next iteration), so a full sync() would add a
            # dead instruction. NOTE: the gfx950 sibling is NOT a precedent for a bare
            # barrier here. Its NBUF=2 separates K (read j%2, written (j+1)%2) but NOT
            # V: it reads vbuf_prev=(j+1)%2 and writes pbuf=(j+1)%2 -- the SAME buffer,
            # across a bare s_barrier_bare(). Correct there only by scheduler luck;
            # tracked separately, do not copy that idiom.
            b.sync_lds_only()
            b.scf_yield(m_new, l_new, *o_acc)

        # elide_trailing_barrier=False: the trailing barrier is NOT an optimizable
        # rendezvous -- it is the WAR guard on the SINGLE K/V LDS buffer. The elide
        # pass (lower_llvm._lower_unrolled_for) targets body_ops[-2], which is exactly
        # this barrier's slot, and only misses it today because the op-name match is
        # hardcoded to "tile.sync" and unroll defaults False. Pin it so a future
        # unroll=True (P3) or a sync_lds_only->sync swap cannot silently delete it.
        # Verified byte-identical codegen with and without the flag today.
        # KV-segment split: this CTA owns tiles [loop_lo, loop_hi) of [0, n_up),
        # one contiguous chunk of ceil(n_up / KVS) tiles. A segment past the end is
        # empty (zero-trip loops) and writes m = -inf, l = 0, O = 0.
        if KVS > 1:
            loop_lo, loop_hi = segment_tile_range(b, seg, b.const_i32(0), n_up, KVS)
        if USE_PIPELINE:
            # tile_body serves the shipped single-buffer configurations unchanged.
            # This BF16 CFVST pipeline independently schedules K[j+1] DMA and
            # V[j+1] register loads across softmax/PV[j]. KDB overlaps its next
            # DMA only with a distinct K slot; VPF overlaps V loads in registers.
            def publish_prefetched_v(values):
                # Prefetch carries only packed V data across softmax/PV, not the
                # transposition coordinates. Recompute coordinates at publication.
                payload = []
                for it, (x0, x1) in enumerate(values):
                    # blk=it*THREADS+tid, with THREADS=64*WAVES and
                    # V_DIM_PAIRS=64. Unroll-time offsets eliminate div/mod.
                    t0 = b.add(
                        b.mul(wave, b.const_i32(2)),
                        b.const_i32(it * 2 * WAVES),
                    )
                    d0 = b.mul(lane, b.const_i32(2))
                    payload.append((d0, t0, x0, x1))
                _cfvst_store_v(payload)

            def _vdb_store_steps(values):
                """qkla_vdb V(j+1) store: per tile 2 bases (rows d0, d0+1 with slot
                and swizzle folded in), then the item's token offset it*2*WAVES by
                XOR (disjoint from wave*2's bits, so (wave*2 + c) ^ s == c ^ (wave*2 ^ s))."""
                d0 = b.mul(lane, b.const_i32(2))
                w2 = b.mul(wave, b.const_i32(2))
                slot_e = b.mul(V_SLOT["w"], b.const_i32(D * V_LDROW))
                bases = []
                for r in range(2):
                    dr = d0 if r == 0 else b.add(d0, b.const_i32(1))
                    sw = b.shl(b.land(dr, b.const_i32(V_SWZ_MASK)), b.const_i32(2))
                    bases.append(b.add(
                        b.add(slot_e, b.mul(dr, b.const_i32(V_LDROW))), b.xor(w2, sw)
                    ))
                for it, (x0, x1) in enumerate(values):
                    rows = (
                        b.perm_b32(x0, x1, b.const_i32(0x01000504)),
                        b.perm_b32(x0, x1, b.const_i32(0x03020706)),
                    )
                    c = it * 2 * WAVES
                    for r in range(2):
                        e = bases[r] if c == 0 else b.xor(bases[r], b.const_i32(c))
                        b.smem_store_vN(
                            V_lds, [b.const_i32(0), b.const_i32(0), e],
                            b.bitcast(rows[r], VectorType(dtype, 2)), 2,
                        )
                    yield

            def publish_v_steps(values):
                """publish_prefetched_v one item per step (tilv_v_write_qk)."""
                if V_SLOT["w"] is not None and SWZ_V:
                    yield from _vdb_store_steps(values)
                    return
                for it, (x0, x1) in enumerate(values):
                    t0 = b.add(
                        b.mul(wave, b.const_i32(2)),
                        b.const_i32(it * 2 * WAVES),
                    )
                    d0 = b.mul(lane, b.const_i32(2))
                    _cfvst_store_v([(d0, t0, x0, x1)])
                    yield

            VWQ = spec.tilv_v_write_qk
            QKLA = spec.qk_lookahead

            # Prologue: prepare K[0] and publish V[0]. All waves rendezvous
            # after draining their VMEM and after V's LDS writes complete.
            zero = b.const_i32(0)
            if spec.q_batch_wait:
                b.s_waitcnt(vmcnt=0)  # Q batch lands before K/V are issued
            load_pipeline_k(zero, zero)
            if QKLA:
                # K[1] -> slot 1 (clamped); QK(0) runs here, S(0) is carried.
                la_last = b.sub(n_up, b.const_i32(1))
                one = b.const_i32(1)
                load_pipeline_k(one, b.smin(one, la_last))
            v0 = _cfvst_load_v_pipeline(zero)
            SPLITW = spec.qkla_split_wait
            if SPLITW:
                # Q (8 dwordx4) + K(0) (ROWS_PER_WAVE DMAs) were issued first; leave
                # K(1)'s DMAs and V(0)'s 2 * V_ITEMS loads in flight across QK(0).
                assert ROWS_PER_INSTR == 1
                b.s_waitcnt(vmcnt=ROWS_PER_WAVE + 2 * V_ITEMS)
            else:
                b.s_waitcnt(vmcnt=0)
            if spec.q_lds_stage:
                # V_lds is free until the barrier below (V[0] is published after
                # it). Per wave 32 rows x 32 dims per quarter, 8 B chunks XORed
                # with (row >> 1) & 7 so the MFMA-layout reads are conflict-free.
                q_base_e = b.mul(wave, b.const_i32(1024))
                if VDB:  # slot 0 takes V(0) below; stage Q in slot 1
                    q_base_e = b.add(q_base_e, b.const_i32(D * V_LDROW))
            if spec.q_lds_stage and spec.q_full_line:
                # Per wave 32 rows x 64 dims (one half of D) in both V slots
                # (wave * 2048 elements; V(0) is published after QK(0)). 16 B
                # chunks XORed with row & 7: a ds_write_b128 pass (8 lanes, one
                # row) and a ds_read_b128 pass (8 rows, one chunk) both hit 8
                # distinct 16 B slots. q_packs come out in the KB128 order.
                assert VDB and 2 * D * V_LDROW >= WAVES * 2048
                q_base_e = b.mul(wave, b.const_i32(2048))
                lane_q = b.optimization_barrier(lane)
                r_sw = b.land(b.mod(lane_q, b.const_i32(32)), b.const_i32(7))
                q_lrow = b.div(lane_q, b.const_i32(8))
                q_lc = b.mod(lane_q, b.const_i32(8))
                for h in range(2):
                    for i in range(4):
                        row = b.add(b.const_i32(i * 8), q_lrow)
                        w_sw = b.land(row, b.const_i32(7))
                        e = b.add(
                            b.add(q_base_e, b.mul(row, b.const_i32(64))),
                            b.mul(b.xor(q_lc, w_sw), b.const_i32(8)),
                        )
                        b.smem_store_vN(
                            V_lds, [b.const_i32(0), b.const_i32(0), e],
                            q_raw[h * 4 + i], 8,
                        )
                    for pq in range(2):
                        for tt in range(2):
                            ch = b.xor(
                                b.add(b.const_i32(pq * 4 + tt * 2), lane_h), r_sw
                            )
                            e = b.add(
                                b.add(q_base_e, b.mul(lane_m, b.const_i32(64))),
                                b.mul(ch, b.const_i32(8)),
                            )
                            v8 = b.smem_load_vN(
                                V_lds, b.const_i32(0), b.const_i32(0), e,
                                dtype=dtype, n=8,
                            )
                            for half in range(2):
                                q_packs.append(
                                    b.vec_pack(
                                        [b.vec_extract(v8, half * 4 + k)
                                         for k in range(4)],
                                        dtype,
                                    )
                                )
            elif spec.q_lds_stage and KB128:
                # Paired b128 layout: lane (row lane_m, half h) needs dims
                # 16t + 8h .. +7 for t = 0..7. Per quarter p (t = 2p, 2p+1) the
                # 64 B row holds four 16 B chunks, XORed with (row >> 1) & 3 so the
                # ds_read_b128 (8 lanes = 8 rows per pass) hits 8 distinct slots.
                q_lrow = b.div(lane, b.const_i32(4))
                q_lc = b.mod(lane, b.const_i32(4))
                r_sw = b.land(b.lshr(lane_m, b.const_i32(1)), b.const_i32(3))
                for p in range(4):
                    for i in range(2):
                        row = b.add(b.const_i32(i * 16), q_lrow)
                        w_sw = b.land(b.lshr(row, b.const_i32(1)), b.const_i32(3))
                        e = b.add(
                            b.add(q_base_e, b.mul(row, b.const_i32(32))),
                            b.mul(b.xor(q_lc, w_sw), b.const_i32(8)),
                        )
                        b.smem_store_vN(
                            V_lds, [b.const_i32(0), b.const_i32(0), e],
                            q_raw[p * 2 + i], 8,
                        )
                    for tt in range(2):
                        ch = b.xor(b.add(b.const_i32(tt * 2), lane_h), r_sw)
                        e = b.add(
                            b.add(q_base_e, b.mul(lane_m, b.const_i32(32))),
                            b.mul(ch, b.const_i32(8)),
                        )
                        v8 = b.smem_load_vN(
                            V_lds, b.const_i32(0), b.const_i32(0), e, dtype=dtype, n=8
                        )
                        for half in range(2):
                            q_packs.append(
                                b.vec_pack(
                                    [b.vec_extract(v8, half * 4 + k) for k in range(4)],
                                    dtype,
                                )
                            )
            elif spec.q_lds_stage:
                q_lrow = b.div(lane, b.const_i32(4))
                q_lch = b.mul(b.mod(lane, b.const_i32(4)), b.const_i32(2))
                r_sw = b.land(b.lshr(lane_m, b.const_i32(1)), b.const_i32(7))
                for p in range(4):
                    for i in range(2):
                        row = b.add(b.const_i32(i * 16), q_lrow)
                        w_sw = b.land(b.lshr(row, b.const_i32(1)), b.const_i32(7))
                        v8 = q_raw[p * 2 + i]
                        for hh in range(2):
                            ch = b.xor(b.add(q_lch, b.const_i32(hh)), w_sw)
                            e = b.add(
                                b.add(q_base_e, b.mul(row, b.const_i32(32))),
                                b.mul(ch, b.const_i32(4)),
                            )
                            b.smem_store_vN(
                                V_lds,
                                [b.const_i32(0), b.const_i32(0), e],
                                b.vec_pack(
                                    [b.vec_extract(v8, hh * 4 + t) for t in range(4)],
                                    dtype,
                                ),
                                4,
                            )
                    for kq in range(4):
                        ch = b.xor(b.add(b.const_i32(kq * 2), lane_h), r_sw)
                        e = b.add(
                            b.add(q_base_e, b.mul(lane_m, b.const_i32(32))),
                            b.mul(ch, b.const_i32(4)),
                        )
                        q_packs.append(
                            b.smem_load_vN(
                                V_lds, b.const_i32(0), b.const_i32(0), e,
                                dtype=dtype, n=4,
                            )
                        )
            if QKLA:
                if VDB and not SPLITW:
                    publish_prefetched_v(v0)  # V(0) -> slot 0
                b.sync_lds_only()
                s_acc0 = []
                for _ in qk_raw_steps(zero, s_acc0):
                    pass
                if SPLITW:
                    # K(1) + V(0) landed during QK(0); publish V(0) -> slot 0 and
                    # K(1) (other waves' DMAs) before tile 0 reads them. The
                    # sched_barrier keeps all QK(0) MFMAs ahead of the wait.
                    b.sched_barrier(0)
                    b.s_waitcnt(vmcnt=0)
                    publish_prefetched_v(v0)
                    b.sync_lds_only()
                if spec.qkla_chunked:
                    s_acc0 = [
                        b.vec_pack(
                            [b.fmul(b.vec_extract(a, i), qk_scale) for i in range(16)], F32
                        )
                        for a in s_acc0
                    ]
                iter_args = iter_args + ([] if VDB else [
                    (f"vp{k}", x) for k, x in enumerate(x for pr in v0 for x in pr)
                ]) + [(f"sa{nsub}", s_acc0[nsub]) for nsub in range(N_SUB)]
                if spec.qkla_max_ahead:
                    mx0 = neg_inf
                    for a_ in s_acc0:
                        for i_ in range(16):
                            mx0 = b.fmax(mx0, b.vec_extract(a_, i_))
                    ma0 = b.fmax(mx0, b.warp_shuffle_xor(mx0, 32))
                if spec.qkla_chunked and spec.qkla_chunk_rev == 3:
                    # rev 3: no previous tile -> zero pending PV (adds exactly 0)
                    iter_args = iter_args + [
                        (f"cv{k}", zero_vpack()) for k in range(QKLA_CARRY)
                    ] + [("cp", zero_vpack())]
                if spec.qkla_max_ahead:
                    iter_args = iter_args + [("ma", ma0)]  # last carried value
            elif VWQ:
                # V[0] stays in registers; tile 0 writes it during its QK nsub0.
                b.sync_lds_only()
                iter_args = iter_args + [
                    (f"vp{k}", x) for k, x in enumerate(x for pr in v0 for x in pr)
                ]
            else:
                publish_prefetched_v(v0)
                b.sync_lds_only()

            # Main loop deliberately excludes the final KV tile. n_up=1 makes
            # this a zero-trip loop; its prologue and epilogue still execute.
            last_j = b.sub(n_up, b.const_i32(1))

            def pipeline_step(j, carry, *, masked):
                """Compute tile j and publish tile j+1, retaining the LDS
                rendezvous and pipeline scheduling across split-loop boundaries."""
                m_i = carry[0]
                l_i = carry[1]
                o_acc = list(carry[2 : 2 + D_TILES])
                if spec.iglp:
                    b.iglp_opt(spec.iglp_mode)

                # The current K and V tiles are fully published by the prologue
                # or the preceding iteration. With one K slot, do not overwrite
                # it until *all* waves have completed current-tile LDS reads.
                kbuf = b.mod(j, b.const_i32(K_BUFS))
                if QKLA:
                    # S(j) arrives in the carry; QK(j+1) reads slot (j+1)%2 (its DMA
                    # landed before the previous end barrier). Slot j%2 was last read
                    # by QK(j) before the previous mid barrier, so K[j+2] (clamped:
                    # a harmless reload into the idle slot) lands there.
                    nv = 0 if VDB else 2 * V_ITEMS
                    pv_flat = carry[2 + D_TILES : 2 + D_TILES + nv]
                    v_cur = [(pv_flat[2 * i], pv_flat[2 * i + 1])
                             for i in range(0 if VDB else V_ITEMS)]
                    s_acc = list(carry[2 + D_TILES + nv : 2 + D_TILES + nv + N_SUB])
                    MAXA = spec.qkla_max_ahead
                    ma_in = carry[-1] if MAXA else None
                    pend_in = list(carry[2 + D_TILES + nv + N_SUB : len(carry) - (1 if MAXA else 0)])
                    ma_out = [] if MAXA else None
                    next_j = b.add(j, b.const_i32(1))
                    next_kbuf = b.mod(next_j, b.const_i32(K_BUFS))
                    s_next = []
                    k2 = b.smin(b.add(j, b.const_i32(2)), last_j)
                    pend_out = []
                    if spec.qkla_chunked:
                        rev3 = spec.qkla_chunk_rev == 3
                        if VDB:
                            # PV(j) reads slot j%2 (V(j) is already there); V(j+1)
                            # goes to slot (j+1)%2, whose last readers (PV(j-1))
                            # are all past the previous end barrier. The end
                            # barrier below publishes it for PV(j+1).
                            V_SLOT["r"] = b.mod(j, b.const_i32(2))
                            V_SLOT["w"] = b.mod(next_j, b.const_i32(2))
                            V_SLOT["rb"] = None
                        m_new, l_new, o_acc, v_next, s_next = tile_chunked(
                            j, m_i, l_i, o_acc, s_acc,
                            iter(()) if VDB else publish_v_steps(v_cur),
                            qk_raw_steps(next_kbuf, s_next, preyield=rev3), s_next, masked,
                            lambda vals: _cfvst_load_v_pipeline_steps(next_j, vals),
                            lambda: load_pipeline_k_steps(kbuf, k2),
                            pend=pend_in if rev3 else None,
                            pend_sink=pend_out if rev3 else None,
                            v_pub=publish_v_steps if VDB else None,
                            m_ahead=ma_in, ma_sink=ma_out,
                        )
                        V_SLOT["r"] = V_SLOT["w"] = V_SLOT["rb"] = None
                    else:
                        m_new, l_new, o_acc, v_next = tile_la(
                            j, m_i, l_i, o_acc, s_acc, publish_v_steps(v_cur),
                            qk_raw_steps(next_kbuf, s_next), masked,
                            lambda vals: _cfvst_load_v_pipeline_steps(next_j, vals),
                            lambda: load_pipeline_k_steps(kbuf, k2),
                        )
                    b.s_waitcnt(vmcnt=0)
                    b.sync_lds_only()
                    if spec.qkla_rhythm_pv:
                        # 1 MFMA then N VALU over the PV region (PV + QK tail MFMAs).
                        for _ in range(KK_STEPS * D_TILES + spec.qkla_qk_tail):
                            b.sched_group_barrier(0x008, 1, 0)
                            b.sched_group_barrier(0x002, spec.qkla_rhythm_pv, 0)
                    b.scf_yield(
                        m_new, l_new, *o_acc,
                        *([] if VDB else [x for pr in v_next for x in pr]), *s_next,
                        *pend_out,
                        *([(ma_out[0] if ma_out else ma_in)] if MAXA else []),
                    )
                    return
                if VWQ:
                    pv_flat = carry[2 + D_TILES:]
                    v_cur = [(pv_flat[2 * i], pv_flat[2 * i + 1]) for i in range(V_ITEMS)]
                if spec.tile_interleave and not masked:
                    next_j = b.add(j, b.const_i32(1))
                    next_kbuf = b.mod(next_j, b.const_i32(K_BUFS))
                    m_new, l_new, o_acc, v_next = tile_ilv(
                        kbuf, j, m_i, l_i, o_acc,
                        lambda vals: _cfvst_load_v_pipeline_steps(next_j, vals),
                        lambda: load_pipeline_k_steps(next_kbuf, next_j),
                        v_store_steps=publish_v_steps(v_cur) if VWQ else None,
                    )
                else:
                    if VWQ:
                        # Masked tile: publish V(j) up front (not interleaved).
                        publish_prefetched_v(v_cur)
                        b.sync_lds_only()
                    s = do_qk(kbuf)
                    next_j = b.add(j, b.const_i32(1))
                    next_kbuf = b.mod(next_j, b.const_i32(K_BUFS))
                    if USE_KDB:
                        # Distinct LDS slot: K[j+1] DMA overlaps current softmax/PV.
                        load_pipeline_k(next_kbuf, next_j)
                    if USE_VPF:
                        # Only V data (not addresses) stays live across softmax/PV.
                        v_next = _cfvst_load_v_pipeline(next_j)
                    # The pipeline cohort has no sliding window, so masked=True
                    # applies only the causal upper bound.
                    m_new, l_new, o_acc = softmax_pv(s, j, m_i, l_i, o_acc, masked)

                if VWQ:
                    # K(j+1) DMA and V(j+1) loads landed; every wave finished its
                    # PV(j) V reads -> next tile may read K(j+1) and overwrite V.
                    b.s_waitcnt(vmcnt=0)
                    b.sync_lds_only()
                    b.scf_yield(m_new, l_new, *o_acc, *[x for pr in v_next for x in pr])
                    return
                # WAR: finish all waves' K/V LDS reads before a single K slot
                # or the (always single-buffered) V slot can be overwritten.
                # LDS-only sync leaves early K DMA / V VMEM loads in flight.
                b.sync_lds_only()
                if not USE_KDB:
                    # Single K slot: only overwrite it after the cross-wave WAR.
                    load_pipeline_k(next_kbuf, next_j)
                if not USE_VPF:
                    # No V prefetch: issue V[j+1] loads only after current PV.
                    v_next = _cfvst_load_v_pipeline(next_j)
                # Drain next K async-DMA and next V VMEM loads before V LDS
                # publication; then rendezvous before any next-tile QK/PV.
                b.s_waitcnt(vmcnt=0)
                publish_prefetched_v(v_next)
                b.sync_lds_only()
                b.scf_yield(m_new, l_new, *o_acc)

            if causal and DIAG_SPLIT:
                # At kt < qb * (BLOCK_M/BN), every key precedes every query
                # row, so the causal upper mask is unnecessary. Split only the
                # processing loop (which excludes last_j). It continuously
                # publishes K/V tile j+1 across the loop transition, and the
                # final tile remains safely masked even when it is below the
                # diagonal (e.g. seqlen_kv < seqlen_q).
                diag_start = b.mul(qb, b.const_i32(n_per))
                body_upper = b.select(b.cmp_lt(diag_start, last_j), diag_start, last_j)
                body = b.scf_for_iter(
                    zero,
                    body_upper,
                    b.const_i32(1),
                    iter_args,
                    iv_name="kbp",
                    elide_trailing_barrier=False,
                )
                with body as (j, carry):
                    pipeline_step(j, carry, masked=False)
                tail_args = [
                    (name + "_t", val)
                    for (name, _), val in zip(iter_args, body.results)
                ]
                loop = b.scf_for_iter(
                    body_upper,
                    last_j,
                    b.const_i32(1),
                    tail_args,
                    iv_name="ktp",
                    elide_trailing_barrier=False,
                )
                with loop as (j, carry):
                    pipeline_step(j, carry, masked=True)
            else:
                # Preserve the non-split pipeline's original emitted operations.
                loop = b.scf_for_iter(
                    zero,
                    last_j,
                    b.const_i32(1),
                    iter_args,
                    iv_name="ktp",
                    elide_trailing_barrier=False,
                )
                with loop as (j, carry):
                    pipeline_step(j, carry, masked=True)

            # Epilogue: do not issue K/V loads beyond n_up-1. The last LDS
            # barrier also protects CTA-persistent storage reuse by the next item.
            loop_res = loop.results
            m_i = loop_res[0]
            l_i = loop_res[1]
            if spec.qkla_lhalf:
                # per-half partial row sums -> full row sum (once per work item)
                l_i = b.fadd(l_i, b.warp_shuffle_xor(l_i, 32))
            o_acc = list(loop_res[2 : 2 + D_TILES])
            last_kbuf = b.mod(last_j, b.const_i32(K_BUFS))
            if VWQ and not VDB:
                pv_flat = loop_res[2 + D_TILES : 2 + D_TILES + 2 * V_ITEMS]
                publish_prefetched_v(
                    [(pv_flat[2 * i], pv_flat[2 * i + 1]) for i in range(V_ITEMS)]
                )
                b.sync_lds_only()
            if QKLA and spec.qkla_chunked:
                # S was carried already scaled.
                sbase = 2 + D_TILES + (0 if VDB else 2 * V_ITEMS)
                s = [
                    [b.vec_extract(acc, i) for i in range(16)]
                    for acc in loop_res[sbase : sbase + N_SUB]
                ]
                if spec.qkla_chunk_rev == 3:
                    # rev 3: run the last tile's carried PV MFMAs before the epilogue
                    pend = list(loop_res[sbase + N_SUB :])
                    for i, dtc in enumerate(carried_dts()):
                        o_acc[dtc] = mfma_32x32x8_for_dtype(
                            b, dtype, pend[i], pend[-1], o_acc[dtc]
                        )
            elif QKLA:
                s = [
                    [b.fmul(b.vec_extract(acc, i), qk_scale) for i in range(16)]
                    for acc in loop_res[2 + D_TILES + 2 * V_ITEMS :]
                ]
            else:
                s = do_qk(last_kbuf)
            if VDB:
                V_SLOT["r"] = b.mod(last_j, b.const_i32(2))
                V_SLOT["rb"] = None
            m_i, l_i, o_acc = softmax_pv(s, last_j, m_i, l_i, o_acc, True)
            V_SLOT["r"] = V_SLOT["rb"] = None
            b.sync_lds_only()

        elif causal and not SW and DIAG_SPLIT:
            # Tiles below this query block's diagonal (kt < qb*n_per) hold only keys
            # <= every query row of the block, so the causal mask is a no-op there;
            # only [qb*n_per, n_up) is masked. The single K/V buffer spans both
            # loops, so both keep the trailing WAR barrier.
            if BR:
                # Tiles wholly at or below the block's first (shifted) query row.
                diag_start = b.div(
                    b.add(b.mul(qb, b.const_i32(BLOCK_M)), diag_off),
                    b.const_i32(BN),
                )
            else:
                diag_start = b.mul(qb, b.const_i32(n_per))
            if KVS > 1:
                body_lower = loop_lo
                body_upper = b.select(
                    b.cmp_gt(diag_start, loop_lo), diag_start, loop_lo
                )
                body_upper = b.select(
                    b.cmp_lt(body_upper, loop_hi), body_upper, loop_hi
                )
                tail_upper = loop_hi
            else:
                body_lower = None
                body_upper = b.select(b.cmp_lt(diag_start, n_up), diag_start, n_up)
                tail_upper = n_up
            body = b.scf_for_iter(
                body_lower if KVS > 1 else b.const_i32(0),
                body_upper,
                b.const_i32(1),
                iter_args,
                iv_name="kb",
                elide_trailing_barrier=False,
            )
            with body as (j, carry):
                tile_body(j, carry, masked=False)
            tail_args = [
                (name + "_t", val) for (name, _), val in zip(iter_args, body.results)
            ]
            loop = b.scf_for_iter(
                body_upper,
                tail_upper,
                b.const_i32(1),
                tail_args,
                iv_name="kt",
                elide_trailing_barrier=False,
            )
            with loop as (j, carry):
                tile_body(j, carry, masked=True)
        elif causal and SW and SWt >= n_per:
            # Sliding-window band loop split in three phases when the window spans
            # the query block (SWt >= n_per):
            #   L: [start_tile, mid_lo)  window-edge tiles: window bound only (every
            #                            key is below the block's first query)
            #   M: [mid_lo, mid_hi)      interior: mask-free
            #   R: [mid_hi, n_up)        diagonal tiles: causal bound only (every key
            #                            is at or past the block's first query, so
            #                            inside every row's window)
            # A narrower window keeps the single loop with both bounds: its M phase
            # is empty and both edges need both bounds, so a split only adds code.
            diag_start = b.mul(qb, b.const_i32(n_per))
            left_end = b.add(diag_start, b.const_i32(n_per - SWt))

            def _clamp(x, lo, hi):
                x = b.select(b.cmp_lt(x, lo), lo, x)  # max(x, lo)
                return b.select(b.cmp_lt(x, hi), x, hi)  # min(x, hi)

            mid_lo = _clamp(left_end, start_tile, n_up)
            mid_hi = _clamp(diag_start, mid_lo, n_up)
            phases = (
                ("kl", start_tile, mid_lo, True, False),
                ("km", mid_lo, mid_hi, False, False),
                ("kt", mid_hi, n_up, False, True),
            )
            args = iter_args
            for iv, lo, hi, lower, upper in phases:
                loop = b.scf_for_iter(
                    lo,
                    hi,
                    b.const_i32(1),
                    args,
                    iv_name=iv,
                    elide_trailing_barrier=False,
                )
                with loop as (j, carry):
                    tile_body(j, carry, masked=(lower, upper))
                args = [
                    (f"{name}_{iv}", val)
                    for (name, _), val in zip(iter_args, loop.results)
                ]
        else:
            loop = b.scf_for_iter(
                loop_lo if KVS > 1 else (start_tile if SW else b.const_i32(0)),
                loop_hi if KVS > 1 else n_up,
                b.const_i32(1),
                iter_args,
                iv_name="kt",
                elide_trailing_barrier=False,
            )
            with loop as (j, carry):
                tile_body(j, carry, masked=True)

        # Both branches leave the FINAL running max in m_i: the pipeline folds its
        # last tile in after the loop, so loop.results[0] is stale on that path.
        if not USE_PIPELINE:
            res = loop.results
            m_i = res[0]
            l_i = res[1]
            o_acc = list(res[2 : 2 + D_TILES])

        if KVS > 1:
            emit_segment_store(
                b,
                segment_params,
                token=b.add(
                    b.mul(bt, seqlen_q_p), b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
                ),
                head=hq,
                num_heads=hq_p,
                seg=seg,
                segments=KVS,
                head_size=D,
                o_acc=o_acc,
                m=m_i,
                denom=l_i,
                lane_h=lane_h,
            )
            return

        # Epilogue: O[query,dim] = (P@V)/l, from the transposed C[dim,query] accum.
        if spec.emit_lse:
            positive_l = b.fcmp("ogt", l_i, b.const_f32(0.0))
            rcp_l = b.rcp(b.select(positive_l, l_i, b.const_f32(1.0)))
        else:
            rcp_l = b.rcp(l_i)
        o_base = b.add(
            b.mul(
                b.mul(bt, seqlen_q_p if spec.runtime_shape else b.const_i32(Sq)),
                _stride_q(),
            ),
            b.mul(hq, b.const_i32(D)),
        )
        qtok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
        q_row_byte = b.add(o_base, b.mul(qtok, _stride_q()))
        d_half = b.mul(lane_h, b.const_i32(4))
        if spec.emit_lse:
            row_has_key = dense_row_has_key(
                b,
                qtok,
                seqlen_kv_p if spec.runtime_shape else b.const_i32(Skv),
                causal=causal,
                diagonal=0,
                window=SW,
            )
            lse_index = b.add(
                b.mul(
                    b.add(
                        b.mul(
                            bt,
                            hq_p if spec.runtime_shape else b.const_i32(Hq),
                        ),
                        hq,
                    ),
                    seqlen_q_p if spec.runtime_shape else b.const_i32(Sq),
                ),
                qtok,
            )
            # With sinks every row has mass: a row with no key keeps the sink seed
            # (m = sink * LOG2E, l = 1), so its LSE is the sink logit.
            row_has_mass = (
                b.cmp_eq(b.const_i32(1), b.const_i32(1))
                if spec.use_sinks
                else row_has_key
            )
            # Both half-waves hold the same query row; one of them writes its LSE.
            with b.scf_if(b.cmp_eq(lane_h, b.const_i32(0))):
                lse_store(b, lse, lse_index, m_i, l_i, row_has_mass=row_has_mass)
        if spec.o_lds_stage:
            # All waves past their last V/K reads and K DMAs, then O goes through
            # the K buffer: per wave 32 rows x 64 dims per half, 8 B chunks XORed
            # with row & 15; read back row-contiguous, 16 B per lane.
            b.s_waitcnt(vmcnt=0)
            b.sync_lds_only()
            o_base_e = b.mul(wave, b.const_i32(2048))
            # Opaque lane: the staging addresses are computed here, after the
            # loops, instead of being CSE'd with pre-loop values and kept live.
            lane_o = b.optimization_barrier(lane)
            rr = _mfma_32x32_c_col(b, lane_o, 0)
            lane_ho = b.div(lane_o, b.const_i32(32))
            w_sw = b.land(rr, b.const_i32(15))
            o_lrow = b.div(lane_o, b.const_i32(8))
            o_lk = b.mod(lane_o, b.const_i32(8))
            o_lk2 = b.mul(o_lk, b.const_i32(2))
            for hf in range(2):
                for dtl in range(2):
                    dt = hf * 2 + dtl
                    for g in range(4):
                        vals = [
                            b.cast_f32_to(
                                b.fmul(b.vec_extract(o_acc[dt], g * 4 + kk), rcp_l),
                                dtype,
                            )
                            for kk in range(4)
                        ]
                        ch = b.xor(b.add(b.const_i32(dtl * 8 + g * 2), lane_ho), w_sw)
                        e = b.add(
                            b.add(o_base_e, b.mul(rr, b.const_i32(64))),
                            b.mul(ch, b.const_i32(4)),
                        )
                        b.smem_store_vN(
                            K_lds,
                            [b.const_i32(0), b.const_i32(0), e],
                            b.vec_pack(vals, dtype),
                            4,
                        )
                for i in range(4):
                    row = b.add(b.const_i32(i * 8), o_lrow)
                    r_sw = b.land(row, b.const_i32(15))
                    halves = []
                    for hh in range(2):
                        ch = b.xor(b.add(o_lk2, b.const_i32(hh)), r_sw)
                        e = b.add(
                            b.add(o_base_e, b.mul(row, b.const_i32(64))),
                            b.mul(ch, b.const_i32(4)),
                        )
                        halves.append(
                            b.smem_load_vN(
                                K_lds, b.const_i32(0), b.const_i32(0), e,
                                dtype=dtype, n=4,
                            )
                        )
                    v8 = b.vec_pack(
                        [b.vec_extract(halves[t // 4], t % 4) for t in range(8)], dtype
                    )
                    addr = b.add(
                        b.add(o_base, b.mul(b.add(q_tok0, row), _stride_q())),
                        b.add(b.const_i32(hf * 64), b.mul(o_lk, b.const_i32(8))),
                    )
                    b.global_store_vN(o, addr, v8, 8, align=16)
            return
        _emit_o_store(
            b,
            o,
            q_row_byte,
            d_half,
            o_acc,
            rcp_l,
            dtype,
            D_TILES,
            spec.o_store_width,
            row_has_key=row_has_key if spec.emit_lse else None,
        )

    # ---- grid dispatch: default (one CTA per work item) vs persistent (P4) ----
    if spec.persistent:
        # 1-D grid of NP long-lived CTAs; each grid-strides over the flattened work
        # space W = NQB*Hq*B, amortizing per-CTA launch + scalar setup + the cold LDS
        # prime across many work items instead of once per query block. The inner
        # per-work-item compute is byte-identical to the default path (same
        # _run_work_item); only the outer loop + work decode + cross-item LDS
        # rendezvous are new. Ported from the gfx950 sibling.
        #
        # The shape is runtime here too: the work-item count and the fold constants
        # are derived from the params once per CTA, before the loop. The per-item
        # decode divides by runtime B / gqa / NQB / Hq, so each divide goes through
        # a host-computed magic multiplier and shift (spec.fastdiv_divisors) and
        # stays mul-hi + add + shift -- a plain b.div/b.mod on a runtime operand
        # would put the backend's software integer division in every iteration.
        if not spec.runtime_shape:
            raise ValueError(
                "gfx942 attention_dense: the persistent grid needs runtime_shape "
                "(ragged/varlen/paged are not supported)"
            )
        NP = spec.num_persistent
        # ragged rejected -> Sq % BLOCK_M == 0, exact. BLOCK_M is a compile-time
        # constant, so this per-CTA divide strength-reduces.
        nqb_p = b.div(seqlen_q_p, b.const_i32(BLOCK_M))
        w_p = b.mul(b.mul(nqb_p, hq_p), batch_p)
        work_shape = DecodeShape(
            b,
            spec,
            {**shape_values, "nqb": nqb_p, "w": w_p},
            fastdiv=make_fastdiv(b, rt),
        )
        cta_id = b.block_id_x()
        if spec.wg_timing:
            _rt = lambda: b.inline_asm(
                "s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", (), I64,
                result_name_hint="rt",
            )
            _ct = lambda: b.inline_asm(
                "s_memtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", (), I64,
                result_name_hint="ct",
            )
            wgt_t0 = _rt()
            wgt_c0 = _ct()
            wgt_hw = b.inline_asm("s_getreg_b32 $0, hwreg(HW_REG_HW_ID)", "=s", (), I32)
            wgt_xcc = b.inline_asm("s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)", "=s", (), I32)
        outer = b.scf_for(cta_id, w_p, b.const_i32(NP), iv_name="wi")
        with outer as wi:
            # Cross-work-item LDS reuse guard. The previous item's epilogue touches
            # only registers + the O buffer, and its last KV tile's trailing
            # sync_lds_only was the final LDS-read barrier -- so lgkm is already
            # drained. Re-sync all waves (and drain the epilogue O store via vmcnt=0)
            # before this item reissues loads into the shared K/V buffers; a bare
            # barrier suffices because there is no pending lgkm to sink here.
            b.s_waitcnt(vmcnt=0)
            b.s_barrier_bare()
            if spec.xcdlb:
                _run_work_item(*_xcdlb_decode(work_shape, wi))
            else:
                d = PERSIST_DECODES[spec.resolved_persist_decode].emit_decode(
                    work_shape, wi
                )
                # The kv head from the decode where it is a digit, else hq // gqa
                # through the gqa fastdiv pair: no runtime division in the loop.
                hkv_v = d.hkv if d.hkv is not None else work_shape.div(d.hq, "gqa")
                _run_work_item(d.qb, d.hq, d.bt, hkv=hkv_v)
        if spec.wg_timing:
            # After this wave's last O stores have drained; lane 0 of each wave
            # writes its 8-word record at ts_ptr[(cta*8 + wave) * 8].
            b.s_waitcnt(vmcnt=0)
            wgt_t1 = _rt()
            wgt_c1 = _ct()
            c32 = b.const_i64(32)
            lo = lambda t: b.trunc(t, I32)
            hi = lambda t: b.trunc(b.lshr(t, c32), I32)
            rec = b.vec_pack(
                [lo(wgt_t0), hi(wgt_t0), lo(wgt_t1), hi(wgt_t1), wgt_hw, wgt_xcc,
                 cta_id, wave],
                I32,
            )
            z = b.const_i32(0)
            rec2 = b.vec_pack([lo(wgt_c0), hi(wgt_c0), lo(wgt_c1), hi(wgt_c1), z, z, z, z], I32)
            entry = b.add(b.mul(cta_id, b.const_i32(8)), wave)
            with b.scf_if(b.cmp_eq(lane, b.const_i32(0))):
                b.global_store_vN(ts_p, b.mul(entry, b.const_i32(16)), rec, 8, align=16)
                b.global_store_vN(
                    ts_p, b.add(b.mul(entry, b.const_i32(16)), b.const_i32(8)), rec2, 8,
                    align=16,
                )
    elif KVS > 1:
        # The segment index is the fastest-varying part of grid x, so the decode
        # sees the unsplit grid unchanged.
        raw_x = b.block_id_x()
        seg_id = b.mod(raw_x, b.const_i32(KVS))
        d = NONPERSIST_DECODES[spec.resolved_nonpersist_decode].emit_decode(
            DecodeShape(b, spec, shape_values),
            b.div(raw_x, b.const_i32(KVS)),
            b.block_id_y(),
            b.block_id_z(),
        )
        _run_work_item(d.qb, d.hq, d.bt, seg=seg_id)
    else:
        d = NONPERSIST_DECODES[spec.resolved_nonpersist_decode].emit_decode(
            DecodeShape(b, spec, shape_values),
            b.block_id_x(),
            b.block_id_y(),
            b.block_id_z(),
        )
        _run_work_item(d.qb, d.hq, d.bt)
    b.ret()
    return b.kernel


# --- public geometry / ABI surface (arch-neutral; mirrors the gfx950 helpers) ---


def attention_dense_grid(spec: AttentionDenseSpec) -> tuple[int, int, int]:
    """Launch grid: persistent = 1-D grid of ``num_persistent`` CTAs; default =
    one CTA per (query-block, query-head, batch), laid out by the nonpersist
    decode.

    Sized from ``spec.block_m`` (``_BLOCK_M`` at the default) so the grid and the
    body's query tiling cannot disagree -- a mismatch writes some rows twice and
    others never. A plain :class:`AttentionDenseSpec` is promoted to the gfx942
    subclass first, so a caller that only knows the shared spec gets the shipped
    geometry."""
    if spec.persistent:
        return (spec.num_persistent, 1, 1)
    spec = _as_gfx942_spec(spec)
    gx, gy, gz = NONPERSIST_DECODES[spec.resolved_nonpersist_decode].grid(spec)
    return (gx * spec.kv_segments, gy, gz)


def attention_dense_block(spec: AttentionDenseSpec) -> tuple[int, int, int]:
    """CTA block dims: ``spec.block_m // 32`` wave64s.

    The shared spec derives ``num_waves`` from its explicit ``block_m`` field,
    so a block_m sweep point launches the thread count the body actually emits."""
    return (_as_gfx942_spec(spec).num_waves * 64, 1, 1)


def attention_dense_signature(spec: AttentionDenseSpec):
    """ABI signature: q/k/v/o pointers + f32 scale, plus one i32 per
    ``spec.runtime_kernarg_fields`` entry: the shape (batch, seqlen_q, seqlen_kv,
    num_query_heads, num_kv_heads), then on the persistent grid a magic/shift pair
    per work-decode divisor. Then a ``sink_ptr`` ([Hq], q dtype) when
    ``spec.use_sinks`` and an f32 ``lse_ptr`` when ``spec.emit_lse``, in that order.

    THE single definition of this kernel's ABI -- the builder and the benchmark both
    call it rather than re-deriving the parameter list, so a reordering cannot drift
    between the emitted kernel and the launcher packing arguments for it.

    No ``cu_seqlens`` pair: varlen is rejected by :func:`supports_attention_dense`, so
    there is no varlen kernel to describe. Those pointers land when varlen does.
    """
    from rocke.helpers.spec import SignatureBuilder

    sig = (
        SignatureBuilder()
        .ptr("q_ptr", spec.dtype)
        .ptr("k_ptr", spec.dtype)
        .ptr("v_ptr", spec.dtype)
        .ptr("o_ptr", spec.dtype)
        .scalar("scale", "f32")
    )
    # The same tuple, in the same order, that build_attention_dense declares as
    # params right after scale -- cross-checked against KernelDef.params by
    # test_signature_matches_the_built_kernels_params in
    # library/tests/test_attention_builds.py.
    for field in _as_gfx942_spec(spec).runtime_kernarg_fields:
        sig = sig.scalar(field, "i32")
    if spec.use_sinks:
        # Mirrors the sink_ptr param declared after the runtime kernarg fields.
        sig = sig.ptr("sink_ptr", spec.dtype)
    if _as_gfx942_spec(spec).emit_lse:
        sig = sig.ptr("lse_ptr", "f32")
    if _as_gfx942_spec(spec).kv_segments > 1:
        for name in SEGMENT_PARAMS:
            sig = sig.ptr(name, "f32")
    if _as_gfx942_spec(spec).wg_timing:
        sig = sig.ptr("ts_ptr", "i32")
    return sig.build()


def attention_dense_runtime_args(spec: AttentionDenseSpec) -> dict[str, int]:
    """Kernarg values for every ``spec.runtime_kernarg_fields`` entry: the shape
    params read off the spec, then the persistent decode's fast-division pairs
    (:func:`~kernels.common.attention_dense_decode.fastdiv_kernargs`). Empty on the
    baked paths. Every launcher merges this into its pointer/scale args rather than
    listing the fields itself."""
    spec = _as_gfx942_spec(spec)
    args = {f: int(getattr(spec, f)) for f in spec.runtime_param_fields}
    args.update(fastdiv_kernargs(spec, "gfx942"))
    return args


_DENSE_LAUNCHER_CACHE: dict = {}


def run_attention_dense_torch(
    *,
    spec: AttentionDenseSpec,
    q,
    k,
    v,
    out,
    scale: float,
    stream: int = 0,
    arch: str = "gfx942",
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    sinks=None,
    lse=None,
):
    """High-level framework entry: compile (cached) + launch the gfx942 dense prefill
    kernel on torch tensors. ``q``/``out`` are ``[B, S, Hq, D]`` and ``k``/``v`` are
    ``[B, Skv, Hkv, D]``, dense contiguous; ``scale`` is the softmax scale (1/sqrt(D)).
    ``sinks`` is the per-query-head sink logit ``[Hq]`` in q's dtype, required iff
    ``spec.use_sinks``.
    Returns ``out``. torch is imported lazily by the launcher, so this module stays
    torch-free at import time. Serves both the default and the P4 persistent grid
    (``spec.persistent``) -- ``attention_dense_grid`` picks the right launch shape.

    Mirrors ``kernels.gfx950.attention_dense.run_attention_dense_torch`` and keys
    the launcher cache by ``attention_dense_cache_key``. On the runtime-shape path
    (``spec.runtime_shape``: both grids, sliding-window included) batch, seqlen_q,
    seqlen_kv and both head counts are kernel params and drop out of that key, so
    one compiled binary serves every shape; every other IR-live field still
    participates without relying on manual name tokens. On the persistent grid the
    key carries the resolved decode order, so one binary serves each order.

    varlen / ragged are rejected by :func:`supports_attention_dense` on gfx942, so the
    ABI has no ``cu_seqlens`` args (:func:`attention_dense_signature`); passing
    ``cu_seqlens_*`` is a caller error rather than a silently-ignored argument.

    ``spec.emit_lse=True`` requires a caller-owned contiguous FP32 ``lse`` tensor
    [B, Hq, Sq, 1] on the same GPU, separate from every input and output. The kernel
    writes the natural-log softmax normalizer while still returning ``out``. Rows
    with no allowed key (a sliding window past the end of the keys) write O=0 and
    LSE=-inf; with an attention sink, those rows instead write the sink logit as
    LSE. Supplying ``lse`` with ``emit_lse=False`` is an error."""
    spec = _as_gfx942_spec(spec)
    ok, why = supports_attention_dense(spec, arch=arch)
    if not ok:
        raise NotImplementedError(f"attention_dense unsupported for spec: {why}")
    if cu_seqlens_q is not None or cu_seqlens_kv is not None:
        raise ValueError(
            "cu_seqlens_* provided but gfx942 attention_dense is dense-only (varlen "
            "is rejected by supports_attention_dense); the ABI has no cu_seqlens args"
        )
    if not spec.use_sinks and sinks is not None:
        raise ValueError("sinks provided but spec.use_sinks is False")
    if spec.use_sinks:
        if sinks is None:
            raise ValueError("spec.use_sinks=True requires sinks that are not None")
        if sinks.shape != (spec.num_query_heads,):
            raise ValueError(
                f"sinks must have shape ({spec.num_query_heads},), got {tuple(sinks.shape)}"
            )
        if sinks.dtype != q.dtype:
            raise ValueError(f"sinks dtype {sinks.dtype} must match q dtype {q.dtype}")
        if not sinks.is_contiguous():
            raise ValueError("sinks must be contiguous")
        if not sinks.is_cuda:
            raise ValueError("sinks must be a CUDA tensor")
    validate_dense_lse(spec, q, k, v, out, lse, sinks=sinks)
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig

    key = attention_dense_cache_key(spec, arch=arch)
    launcher = _DENSE_LAUNCHER_CACHE.get(key)
    if launcher is None:
        art = compile_kernel(
            build_attention_dense(spec, arch=arch),
            arch=arch,
            backend="python",
            capture_ir_text=False,
        )
        assert art.kernel_name == spec.kernel_name(), (
            art.kernel_name,
            spec.kernel_name(),
        )
        launcher = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=attention_dense_signature(spec),
        )
        _DENSE_LAUNCHER_CACHE[key] = launcher
    vals = {"q_ptr": q, "k_ptr": k, "v_ptr": v, "o_ptr": out, "scale": float(scale)}
    vals.update(attention_dense_runtime_args(spec))
    if spec.use_sinks:
        vals["sink_ptr"] = sinks
    if spec.emit_lse:
        vals["lse_ptr"] = lse
    if spec.kv_segments > 1:
        workspace = allocate_segment_workspace(spec, q.device)
        vals.update(workspace)
    launcher(
        vals,
        config=LaunchConfig(
            grid=attention_dense_grid(spec),
            block=attention_dense_block(spec),
            stream=int(stream),
        ),
    )
    if spec.kv_segments > 1:
        launch_segment_combine(spec, arch, out, workspace, stream)
    return out
