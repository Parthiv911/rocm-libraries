#!/usr/bin/env python3
"""
Local Step-0 lever sweep for gfx942 attention_dense.

Target use:
  gfx942 / attention_dense.py / BF16 D128, on the CFVST-only commit,
  BEFORE K-LDS double buffering + V-register prefetch are added.

Core lever space:
  block_m        = 32,64,128,256,512
  block_n        = 32,64,128
  waves_per_eu   = 1..8
  lds_row_pad    = 0,4,8,...,32
  use_cfvst      = off,on
  use_v_swizzle  = off,on       (only meaningful with CFVST on)
  use_exp2_fast  = off,on
  iglp           = off,on       (only meaningful with CFVST on)

The script:
  1) builds the production-resolved baseline spec;
  2) enumerates the requested lever product;
  3) applies applicability pruning (V swizzle / IGLP only on CFVST path);
  4) lets dataclass validation + supports_attention_dense() prune illegal specs;
  5) compiles every legal spec with the real gfx942 attention_dense builder;
  6) checks every compiled candidate against one shared fp32 SDPA reference;
  7) screens correct candidates with HIP-event timing;
  8) re-times the top candidates against the production baseline in alternating
     same-session A/B order and ranks by median(base_ms / candidate_ms).

Two strategies:
  --strategy full
      Exact Cartesian sweep over all eight lever axes (after applicability +
      legality pruning). This is the strictest / largest search.

  --strategy funnel
      Phase A: geometry/resource axes
        block_m x block_n x waves_per_eu x lds_row_pad
      using the shipped body-policy values.
      Phase B: cross the top geometry configs with
        CFVST x V-swizzle x exp2_fast x IGLP.
      This is much cheaper and mirrors how targeted upstream sweeps are commonly
      structured.

This file is intended to remain local/uncommitted.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo imports.
#
# Put this file at:
#   dnn-providers/hip-kernel-provider/rocke/library/builders/gfx942/attention/
#   prefill/sweep_attention_dense_levers.py
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_ROCKE = _HERE.parents[4]  # .../dnn-providers/hip-kernel-provider/rocke
sys.path.insert(0, str(_ROCKE / "platform" / "python"))
sys.path.insert(0, str(_ROCKE / "library"))

import torch  # noqa: E402

from builders.gfx942.attention.prefill.attention_dense_prefill import (  # noqa: E402
    dense_request,
    resolve_dense_spec,
)
from kernels.gfx942.attention_dense import (  # noqa: E402
    Gfx942AttentionDenseSpec,
    attention_dense_block,
    attention_dense_grid,
    attention_dense_signature,
    build_attention_dense,
    gfx942_kernel_name,
    supports_attention_dense,
)
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.runtime import KernelLauncher, LaunchConfig  # noqa: E402


ARCH = "gfx942"

DEFAULT_BLOCK_M = [32, 64, 128, 256, 512]
DEFAULT_BLOCK_N = [32, 64, 128]
DEFAULT_WPE = list(range(1, 9))
DEFAULT_LDS_ROW_PAD = list(range(0, 33, 4))
BOOL = [False, True]


def int_csv(s: str) -> list[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return vals


def bool01(v: bool) -> int:
    return 1 if v else 0


def spec_fields(spec: Gfx942AttentionDenseSpec) -> dict[str, Any]:
    return {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec)}


def config_fields(spec: Gfx942AttentionDenseSpec) -> dict[str, Any]:
    return {
        "block_m": spec.block_m,
        "block_n": spec.block_n,
        "waves_per_eu": spec.resolved_waves_per_eu(),
        "lds_row_pad": spec.lds_row_pad,
        "use_cfvst": spec.resolved_use_cfvst(),
        "use_v_swizzle": spec.resolved_use_v_swizzle(),
        "use_exp2_fast": spec.resolved_use_exp2_fast(),
        "iglp": spec.iglp,
        "persistent": spec.persistent,
        "num_persistent": spec.num_persistent,
        "persist_decode": spec.resolved_persist_decode,
        "interleave": spec.interleave,
    }


def cfg_string(spec: Gfx942AttentionDenseSpec) -> str:
    c = config_fields(spec)
    return (
        f"bm={c['block_m']} bn={c['block_n']} wpe={c['waves_per_eu']} "
        f"krowpad={c['lds_row_pad']} cfvst={bool01(c['use_cfvst'])} "
        f"vswz={bool01(c['use_v_swizzle'])} "
        f"exp2={bool01(c['use_exp2_fast'])} iglp={bool01(c['iglp'])}"
    )


def make_request(args):
    # No explicit tuning overrides here: first get exactly what dispatch ships.
    request_args = argparse.Namespace(
        persistent=None,
        num_persistent=None,
        persist_decode=None,
        block_n=None,
        waves_per_eu=None,
        interleave=None,
        lds_k_group_pad=None,
        sliding_window=None,
    )
    return dense_request(
        request_args,
        batch=args.batch,
        seqlen_q=args.s,
        seqlen_kv=args.s,
        num_query_heads=args.hq,
        num_kv_heads=args.hkv,
        head_size=args.d,
        causal=True,
        dtype=args.dtype,
    )


class Workload:
    """One shared input set + fp32 reference for every candidate."""

    def __init__(self, spec: Gfx942AttentionDenseSpec, seed: int):
        dev = "cuda"
        dt = torch.bfloat16 if spec.dtype == "bf16" else torch.float16
        B, Sq, Skv = spec.batch, spec.seqlen_q, spec.seqlen_kv
        Hq, Hkv, D = spec.num_query_heads, spec.num_kv_heads, spec.head_size

        torch.manual_seed(seed)
        self.q = (
            torch.randn(B, Sq, Hq, D, device=dev, dtype=dt) * 0.2
        ).contiguous()
        self.k = (
            torch.randn(B, Skv, Hkv, D, device=dev, dtype=dt) * 0.2
        ).contiguous()
        self.v = (
            torch.randn(B, Skv, Hkv, D, device=dev, dtype=dt) * 0.2
        ).contiguous()
        self.out = torch.empty_like(self.q)
        self.scale = 1.0 / math.sqrt(D)

        # One shared fp32 SDPA oracle for the fixed shape.
        qh = self.q.transpose(1, 2).float()
        rep = Hq // Hkv
        kh = self.k.transpose(1, 2).repeat_interleave(rep, 1).float()
        vh = self.v.transpose(1, 2).repeat_interleave(rep, 1).float()
        self.ref = torch.nn.functional.scaled_dot_product_attention(
            qh, kh, vh, is_causal=spec.causal
        ).transpose(1, 2)

        del qh, kh, vh
        torch.cuda.synchronize()


def compile_launcher(spec: Gfx942AttentionDenseSpec):
    art = compile_kernel(
        build_attention_dense(spec, arch=ARCH),
        arch=ARCH,
        backend="python",
        capture_ir_text=False,
    )
    return KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=attention_dense_signature(spec),
    )


def run_candidate(
    spec: Gfx942AttentionDenseSpec,
    wl: Workload,
    *,
    warmup: int,
    iters: int,
    check: bool,
):
    launcher = compile_launcher(spec)
    stream = torch.cuda.current_stream().cuda_stream
    cfg = LaunchConfig(
        grid=attention_dense_grid(spec),
        block=attention_dense_block(spec),
        stream=stream,
    )
    vals = {
        "q_ptr": wl.q,
        "k_ptr": wl.k,
        "v_ptr": wl.v,
        "o_ptr": wl.out,
        "scale": wl.scale,
    }

    def call():
        launcher(vals, config=cfg)

    # First launch also catches launch/codegen mismatches.
    call()
    torch.cuda.synchronize()

    err = float("nan")
    if check:
        err = float((wl.out.float() - wl.ref).abs().max().item())

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    end.record()
    end.synchronize()
    ms = float(start.elapsed_time(end) / iters)
    return ms, err


def construct_spec(req, overrides):
    """
    Use the real dispatch-resolved spec as the base, then explicitly override
    only the lever values under test.
    """
    return resolve_dense_spec(req, overrides)


def applicable_body_combos():
    """
    Body-policy cross product with obvious applicability pruning.

    V swizzle only exists on the CFVST transposed-V path.
    IGLP's documented precondition in this kernel is the in-loop ds_write traffic
    introduced by CFVST; on the naive-V path it is a priori neutral.
    """
    for cfvst in BOOL:
        v_swizzles = BOOL if cfvst else [False]
        iglps = BOOL if cfvst else [False]
        for vswz, exp2, iglp in itertools.product(v_swizzles, BOOL, iglps):
            yield {
                "use_cfvst": cfvst,
                "use_v_swizzle": vswz,
                "use_exp2_fast": exp2,
                "iglp": iglp,
            }


def geometry_combos(args):
    for bm, bn, wpe, pad in itertools.product(
        args.block_m,
        args.block_n,
        args.wpe,
        args.lds_row_pad,
    ):
        yield {
            "block_m": bm,
            "block_n": bn,
            "waves_per_eu": wpe,
            "lds_row_pad": pad,
        }


def enumerate_full(args):
    for geom, body in itertools.product(
        list(geometry_combos(args)),
        list(applicable_body_combos()),
    ):
        yield {**geom, **body}


def legalize(req, overrides):
    """
    Returns (spec, None) when legal, else (None, reason).

    Constructor/__post_init__ failures are expected pruning, not fatal errors.
    supports_attention_dense() is the kernel's authoritative legality gate.
    """
    try:
        spec = construct_spec(req, overrides)
    except Exception as exc:
        return None, f"construct: {exc}"

    try:
        ok, why = supports_attention_dense(spec, arch=ARCH)
    except Exception as exc:
        return None, f"supports exception: {exc}"

    if not ok:
        return None, f"unsupported: {why}"
    return spec, None


def unique_specs(req, override_iter):
    seen = set()
    legal = []
    rejected = []

    for overrides in override_iter:
        spec, reason = legalize(req, overrides)
        if spec is None:
            rejected.append({"overrides": overrides, "reason": reason})
            continue

        key = tuple((f.name, getattr(spec, f.name)) for f in dataclasses.fields(spec))
        if key in seen:
            continue
        seen.add(key)
        legal.append((spec, overrides))

    return legal, rejected


SCREEN_COLUMNS = [
    "rank",
    "kernel",
    "ms",
    "max_abs",
    "correct",
    "baseline_ratio",
    "block_m",
    "block_n",
    "waves_per_eu",
    "lds_row_pad",
    "use_cfvst",
    "use_v_swizzle",
    "use_exp2_fast",
    "iglp",
    "persistent",
    "num_persistent",
    "persist_decode",
    "interleave",
    "error",
]


def row_for(spec, *, ms=None, err=None, baseline_ms=None, error=""):
    c = config_fields(spec)
    correct = None if err is None else bool(math.isfinite(err))
    return {
        "rank": "",
        "kernel": gfx942_kernel_name(spec),
        "ms": ms,
        "max_abs": err,
        "correct": correct,
        "baseline_ratio": (
            baseline_ms / ms
            if baseline_ms is not None and ms is not None and ms > 0
            else None
        ),
        **c,
        "error": error,
    }


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def screen_specs(
    specs,
    wl,
    *,
    baseline_ms,
    tol,
    warmup,
    iters,
    progress_every,
):
    rows = []
    survivors = []
    n = len(specs)
    best_ms = float("inf")

    for i, (spec, overrides) in enumerate(specs, 1):
        try:
            ms, err = run_candidate(
                spec,
                wl,
                warmup=warmup,
                iters=iters,
                check=True,
            )
            correct = math.isfinite(err) and err < tol
            row = row_for(
                spec,
                ms=ms,
                err=err,
                baseline_ms=baseline_ms,
            )
            row["correct"] = correct
            rows.append(row)

            if correct:
                survivors.append((ms, spec, overrides))
                if ms < best_ms:
                    best_ms = ms
                    print(
                        f"  NEW BEST {i}/{n}: {ms:.6f} ms "
                        f"screen={baseline_ms/ms:.4f}x  {cfg_string(spec)}"
                    )
        except Exception as exc:
            row = row_for(spec, error=repr(exc))
            row["correct"] = False
            rows.append(row)

        if i % progress_every == 0 or i == n:
            best_txt = f"{best_ms:.6f} ms" if math.isfinite(best_ms) else "none"
            print(
                f"  progress {i}/{n}: correct={len(survivors)} "
                f"best={best_txt}"
            )

    survivors.sort(key=lambda x: x[0])
    rank_by_kernel = {
        gfx942_kernel_name(spec): rank
        for rank, (_, spec, _) in enumerate(survivors, 1)
    }
    for row in rows:
        row["rank"] = rank_by_kernel.get(row["kernel"], "")

    return survivors, rows


def paired_ab(
    baseline,
    candidate,
    wl,
    *,
    rounds,
    warmup,
    iters,
):
    ratios = []
    base_ms = []
    cand_ms = []

    for r in range(rounds):
        if r % 2 == 0:
            b, _ = run_candidate(
                baseline, wl, warmup=warmup, iters=iters, check=False
            )
            c, _ = run_candidate(
                candidate, wl, warmup=warmup, iters=iters, check=False
            )
        else:
            c, _ = run_candidate(
                candidate, wl, warmup=warmup, iters=iters, check=False
            )
            b, _ = run_candidate(
                baseline, wl, warmup=warmup, iters=iters, check=False
            )

        base_ms.append(b)
        cand_ms.append(c)
        ratios.append(b / c)

    return {
        "median_ratio": statistics.median(ratios),
        "median_baseline_ms": statistics.median(base_ms),
        "median_candidate_ms": statistics.median(cand_ms),
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
        "base_ms": base_ms,
        "candidate_ms": cand_ms,
        "ratios": ratios,
    }


def retime_top(
    baseline,
    survivors,
    wl,
    *,
    top,
    rounds,
    warmup,
    iters,
):
    rows = []
    details = []

    for i, (_, spec, overrides) in enumerate(survivors[:top], 1):
        print(f"\n  A/B candidate {i}/{min(top, len(survivors))}: {cfg_string(spec)}")
        st = paired_ab(
            baseline,
            spec,
            wl,
            rounds=rounds,
            warmup=warmup,
            iters=iters,
        )
        c = config_fields(spec)
        rows.append(
            {
                "kernel": gfx942_kernel_name(spec),
                **c,
                "median_ratio": st["median_ratio"],
                "median_baseline_ms": st["median_baseline_ms"],
                "median_candidate_ms": st["median_candidate_ms"],
                "min_ratio": st["min_ratio"],
                "max_ratio": st["max_ratio"],
                "rounds": rounds,
            }
        )
        details.append(
            {
                "kernel": gfx942_kernel_name(spec),
                "spec": spec_fields(spec),
                "overrides": overrides,
                **st,
            }
        )
        print(
            f"    median base/candidate={st['median_ratio']:.4f}x "
            f"candidate={st['median_candidate_ms']:.6f} ms "
            f"baseline={st['median_baseline_ms']:.6f} ms"
        )

    rows.sort(key=lambda r: r["median_ratio"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows, details


FINAL_COLUMNS = [
    "rank",
    "kernel",
    "median_ratio",
    "median_candidate_ms",
    "median_baseline_ms",
    "min_ratio",
    "max_ratio",
    "rounds",
    "block_m",
    "block_n",
    "waves_per_eu",
    "lds_row_pad",
    "use_cfvst",
    "use_v_swizzle",
    "use_exp2_fast",
    "iglp",
    "persistent",
    "num_persistent",
    "persist_decode",
    "interleave",
]


def production_body_overrides(shipped):
    """
    Explicitly pin the body-policy values the production spec resolves to.
    Useful for funnel phase A: only geometry/resource levers move.
    """
    return {
        "use_cfvst": shipped.resolved_use_cfvst(),
        "use_v_swizzle": shipped.resolved_use_v_swizzle(),
        "use_exp2_fast": shipped.resolved_use_exp2_fast(),
        "iglp": shipped.iglp,
    }


def run_full(args, req, shipped, wl, baseline_ms):
    print("\n=== FULL CARTESIAN SEARCH ===")
    raw_geometry = (
        len(args.block_m)
        * len(args.block_n)
        * len(args.wpe)
        * len(args.lds_row_pad)
    )
    body_count = len(list(applicable_body_combos()))
    print(f"raw geometry combinations: {raw_geometry}")
    print(f"applicable body combinations per geometry: {body_count}")
    print(f"raw applicability-pruned product: {raw_geometry * body_count}")

    legal, rejected = unique_specs(req, enumerate_full(args))
    print(f"legal unique specs after __post_init__/supports pruning: {len(legal)}")
    print(f"statically rejected: {len(rejected)}")

    (args.out_dir / "rejected_full.json").write_text(
        json.dumps(rejected, indent=2, default=str)
    )

    if args.dry_run:
        return [], []

    survivors, rows = screen_specs(
        legal,
        wl,
        baseline_ms=baseline_ms,
        tol=args.tol,
        warmup=args.screen_warmup,
        iters=args.screen_iters,
        progress_every=args.progress_every,
    )
    write_csv(args.out_dir / "screen_full.csv", rows, SCREEN_COLUMNS)
    return survivors, rows


def run_funnel(args, req, shipped, wl, baseline_ms):
    print("\n=== FUNNEL SEARCH ===")
    shipped_body = production_body_overrides(shipped)

    # Phase A: geometry/resource axes only.
    phase_a_overrides = (
        {**geom, **shipped_body} for geom in geometry_combos(args)
    )
    legal_a, rejected_a = unique_specs(req, phase_a_overrides)

    print(
        f"Phase A: geometry legal={len(legal_a)} "
        f"rejected={len(rejected_a)}"
    )
    (args.out_dir / "rejected_funnel_A.json").write_text(
        json.dumps(rejected_a, indent=2, default=str)
    )

    if args.dry_run:
        return [], []

    surv_a, rows_a = screen_specs(
        legal_a,
        wl,
        baseline_ms=baseline_ms,
        tol=args.tol,
        warmup=args.screen_warmup,
        iters=args.screen_iters,
        progress_every=args.progress_every,
    )
    write_csv(args.out_dir / "screen_funnel_A.csv", rows_a, SCREEN_COLUMNS)

    if not surv_a:
        return [], rows_a

    top_geometry = surv_a[: args.funnel_geometry_top]
    print(
        f"\nPhase B: crossing top {len(top_geometry)} geometry configs "
        "with body-policy levers"
    )

    phase_b_overrides = []
    for _, geom_spec, geom_overrides in top_geometry:
        geom = {
            "block_m": geom_spec.block_m,
            "block_n": geom_spec.block_n,
            "waves_per_eu": geom_spec.resolved_waves_per_eu(),
            "lds_row_pad": geom_spec.lds_row_pad,
        }
        for body in applicable_body_combos():
            phase_b_overrides.append({**geom, **body})

    legal_b, rejected_b = unique_specs(req, phase_b_overrides)
    print(
        f"Phase B: legal={len(legal_b)} "
        f"rejected={len(rejected_b)}"
    )
    (args.out_dir / "rejected_funnel_B.json").write_text(
        json.dumps(rejected_b, indent=2, default=str)
    )

    surv_b, rows_b = screen_specs(
        legal_b,
        wl,
        baseline_ms=baseline_ms,
        tol=args.tol,
        warmup=args.screen_warmup,
        iters=args.screen_iters,
        progress_every=args.progress_every,
    )
    write_csv(args.out_dir / "screen_funnel_B.csv", rows_b, SCREEN_COLUMNS)
    return surv_b, rows_a + rows_b


def main() -> int:
    ap = argparse.ArgumentParser(
        description="gfx942 attention_dense existing-lever Step-0 sweep"
    )

    # Fixed workload dimensions.
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--s", type=int, default=4096)
    ap.add_argument("--hq", type=int, default=32)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")

    # Strategy.
    ap.add_argument(
        "--strategy",
        choices=("full", "funnel"),
        default="full",
        help="full = exact eight-lever Cartesian; funnel = cheaper two-stage search",
    )

    # Lever domains.
    ap.add_argument(
        "--block-m",
        type=int_csv,
        default=DEFAULT_BLOCK_M,
        help="default: 32,64,128,256,512",
    )
    ap.add_argument(
        "--block-n",
        type=int_csv,
        default=DEFAULT_BLOCK_N,
        help="default: 32,64,128",
    )
    ap.add_argument(
        "--wpe",
        type=int_csv,
        default=DEFAULT_WPE,
        help="default: 1,2,3,4,5,6,7,8",
    )
    ap.add_argument(
        "--lds-row-pad",
        type=int_csv,
        default=DEFAULT_LDS_ROW_PAD,
        help="default: 0,4,8,12,16,20,24,28,32",
    )

    # Measurement.
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--screen-warmup", type=int, default=2)
    ap.add_argument("--screen-iters", type=int, default=10)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--round-warmup", type=int, default=10)
    ap.add_argument("--round-iters", type=int, default=50)
    ap.add_argument(
        "--funnel-geometry-top",
        type=int,
        default=20,
        help="number of Phase-A geometry winners crossed with body levers",
    )
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="enumerate/prune only; do not compile or benchmark",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/gfx942_attention_dense_lever_sweep"),
    )

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.d != 128:
        print(
            "WARNING: this default lever set is specifically chosen for D128. "
            "D64 uses lds_k_group_pad rather than lds_row_pad as its K-padding lever.",
            file=sys.stderr,
        )

    req = make_request(args)
    shipped = resolve_dense_spec(req, {})

    print("============================================================")
    print("gfx942 attention_dense Step-0 existing-lever sweep")
    print("============================================================")
    print(
        f"shape: B={args.batch} S={args.s} HQ={args.hq} HKV={args.hkv} "
        f"D={args.d} dtype={args.dtype} causal=1"
    )
    print(f"strategy: {args.strategy}")
    print(f"production baseline: {gfx942_kernel_name(shipped)}")
    print(f"                     {cfg_string(shipped)}")

    # Record the experiment definition before running.
    manifest = {
        "shape": {
            "batch": args.batch,
            "s": args.s,
            "hq": args.hq,
            "hkv": args.hkv,
            "d": args.d,
            "dtype": args.dtype,
            "causal": True,
        },
        "strategy": args.strategy,
        "axes": {
            "block_m": args.block_m,
            "block_n": args.block_n,
            "waves_per_eu": args.wpe,
            "lds_row_pad": args.lds_row_pad,
            "use_cfvst": [False, True],
            "use_v_swizzle": "False only if CFVST off; False/True if on",
            "use_exp2_fast": [False, True],
            "iglp": "False only if CFVST off; False/True if on",
        },
        "baseline_spec": spec_fields(shipped),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )

    if args.dry_run:
        # No GPU tensors required to enumerate legality.
        if args.strategy == "full":
            run_full(args, req, shipped, None, float("nan"))
        else:
            # Funnel dry-run only has Phase A because Phase B needs Phase-A timing
            # to choose the top geometry candidates.
            shipped_body = production_body_overrides(shipped)
            legal, rejected = unique_specs(
                req,
                ({**geom, **shipped_body} for geom in geometry_combos(args)),
            )
            print(
                f"\nFunnel Phase A dry-run: legal={len(legal)} "
                f"rejected={len(rejected)}"
            )
            (args.out_dir / "rejected_funnel_A.json").write_text(
                json.dumps(rejected, indent=2, default=str)
            )
        print(f"\nOutputs: {args.out_dir}")
        return 0

    print("\nCreating one shared input set + fp32 SDPA reference ...")
    wl = Workload(shipped, args.seed)

    print("\nBenchmarking production baseline ...")
    baseline_ms, baseline_err = run_candidate(
        shipped,
        wl,
        warmup=args.round_warmup,
        iters=args.round_iters,
        check=True,
    )
    print(
        f"baseline: {baseline_ms:.6f} ms  "
        f"max_abs={baseline_err:.6e}  "
        f"{'PASS' if baseline_err < args.tol else 'FAIL'}"
    )
    if not math.isfinite(baseline_err) or baseline_err >= args.tol:
        print("FATAL: production baseline failed numeric validation.", file=sys.stderr)
        return 2

    start = time.time()

    if args.strategy == "full":
        survivors, _ = run_full(args, req, shipped, wl, baseline_ms)
    else:
        survivors, _ = run_funnel(args, req, shipped, wl, baseline_ms)

    if not survivors:
        print("No correct candidate survived.", file=sys.stderr)
        return 3

    print("\n=== SCREENING WINNERS ===")
    for rank, (ms, spec, _) in enumerate(survivors[: args.top], 1):
        print(
            f"{rank:2d}. {baseline_ms/ms:.4f}x  {ms:.6f} ms  "
            f"{cfg_string(spec)}"
        )

    print(
        f"\n=== SAME-SESSION A/B RETEST: top "
        f"{min(args.top, len(survivors))} ==="
    )
    final_rows, final_details = retime_top(
        shipped,
        survivors,
        wl,
        top=args.top,
        rounds=args.rounds,
        warmup=args.round_warmup,
        iters=args.round_iters,
    )

    write_csv(args.out_dir / "final_ab.csv", final_rows, FINAL_COLUMNS)
    (args.out_dir / "final_ab_details.json").write_text(
        json.dumps(
            {
                "baseline": {
                    "kernel": gfx942_kernel_name(shipped),
                    "spec": spec_fields(shipped),
                    "initial_ms": baseline_ms,
                    "max_abs": baseline_err,
                },
                "results": final_details,
            },
            indent=2,
            default=str,
        )
    )

    print("\n============================================================")
    print("FINAL A/B RANKING")
    print("============================================================")
    for r in final_rows:
        print(
            f"{r['rank']:2d}. {r['median_ratio']:.4f}x  "
            f"cand={r['median_candidate_ms']:.6f} ms  "
            f"base={r['median_baseline_ms']:.6f} ms  "
            f"bm={r['block_m']} bn={r['block_n']} "
            f"wpe={r['waves_per_eu']} pad={r['lds_row_pad']} "
            f"cfvst={bool01(r['use_cfvst'])} "
            f"vswz={bool01(r['use_v_swizzle'])} "
            f"exp2={bool01(r['use_exp2_fast'])} "
            f"iglp={bool01(r['iglp'])}"
        )

    print(f"\nElapsed: {(time.time() - start) / 60.0:.1f} min")
    print(f"Results: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())