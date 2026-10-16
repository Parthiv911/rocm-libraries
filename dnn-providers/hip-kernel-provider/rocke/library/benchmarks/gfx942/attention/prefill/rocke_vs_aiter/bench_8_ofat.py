#!/usr/bin/env python3
"""One-factor-at-a-time rocKE BF16 D128 benchmark using A/B winning levers.

Place beside bench_8.py (the rocKE-only eight-shape benchmark) and run:
    python3 bench_8_ofat.py --list-only
    python3 bench_8_ofat.py --warmup 10 --repeat 50
    python3 bench_8_ofat.py --shape B1_S4096_Hq32_Hkv8_D128 \
                           --shape B16_S4096_Hq32_Hkv8_D128

Each candidate starts from that SHAPE's dispatch-resolved default; knobs are
NEVER accumulated across candidates. The baseline is always first.

IGLP requires two cooperating fields: enable IGLP and select mode 1. It is
included as one logical scheduling lever, not a strictly one-field edit.
An additional IGLP-on/mode-0 probe separates enabling IGLP from selecting mode 1.
"""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import torch

from bench_8 import (
    SHAPES, D, DTYPE, baseline_spec, kernel_name,
    check_supported, benchmark_rocke,
)

# ONLY the parameters that differ from dispatch in config A or config B.
# Each dictionary is applied to the original baseline, not another candidate.
OFAT = [
    ("waves_per_eu_3",        {"waves_per_eu": 3},                 "A"),
    ("waves_per_eu_4",        {"waves_per_eu": 4},                 "B"),
    ("lds_row_pad_4",        {"lds_row_pad": 4},                 "A+B"),
    ("causal_diag_split",    {"causal_diag_split": True},        "A+B"),
    ("pv_priority_2",        {"pv_priority": 2},                 "A"),
    ("iglp_on_mode_0",       {"iglp": True},                      "A probe"),
    ("iglp_on_mode_1",       {"iglp": True, "iglp_mode": 1},      "A"),
    ("pv_loop_order_k_major",{"pv_loop_order": "k_major"},       "B"),
    ("pv_fence_mask_0",      {"pv_sched_fence_mask": 0},         "B"),
    ("persist_decode_hkv",  {"persist_decode": "hkv_major"},   "B"),
]


def shape_key(B, S, Hq, Hkv):
    return f"B{B}_S{S}_Hq{Hq}_Hkv{Hkv}_D{D}"


def variants_for(base):
    """Default first, followed by distinct, supported one-lever overrides."""
    variants = [("default", base, {}, "baseline")]
    seen = {kernel_name(base)}
    for label, knobs, origin in OFAT:
        candidate = replace(base, **knobs)
        check_supported(candidate)
        actual = kernel_name(candidate)
        # Some levers are already dispatch defaults on some shapes.
        if actual in seen:
            continue
        seen.add(actual)
        variants.append((label, candidate, knobs, origin))
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--shape", action="append", default=[], metavar="SHAPE",
                        help="Only run this exact shape ID; repeat as needed")
    parser.add_argument("--list-only", action="store_true",
                        help="Print generated kernel names without compiling or timing")
    parser.add_argument("--out", type=Path,
                        default=Path("bench_8_shapes_rocke/rocke_ofat.csv"))
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be >=0 and --repeat >=1")
    known = {shape_key(*s): s for s in SHAPES}
    unknown = set(args.shape) - set(known)
    if unknown:
        parser.error(f"Unknown shapes {sorted(unknown)}. Available: {sorted(known)}")
    selected = [s for s in SHAPES if not args.shape or shape_key(*s) in args.shape]

    rows = []
    for idx, (B, S, Hq, Hkv) in enumerate(selected, 1):
        sid = shape_key(B, S, Hq, Hkv)
        base = baseline_spec(B, S, Hq, Hkv)
        variants = variants_for(base)
        print(f"\n[{idx}/{len(selected)}] {sid} — {len(variants)} configs", flush=True)
        if not args.list_only:
            torch.manual_seed(0)
            q = (torch.randn(B, S, Hq, D, dtype=torch.bfloat16, device="cuda") * 0.2).contiguous()
            k = (torch.randn(B, S, Hkv, D, dtype=torch.bfloat16, device="cuda") * 0.2).contiguous()
            v = (torch.randn(B, S, Hkv, D, dtype=torch.bfloat16, device="cuda") * 0.2).contiguous()
            out = torch.empty_like(q)
        baseline_ms = None
        for label, spec, knobs, origin in variants:
            actual = kernel_name(spec)
            print(f"  {label:23} [{origin}] {knobs}", flush=True)
            print(f"    {actual}", flush=True)
            if args.list_only:
                continue
            try:
                ms = benchmark_rocke(spec, q, k, v, out, args.warmup, args.repeat)
                if label == "default":
                    baseline_ms = ms
                speedup = baseline_ms / ms if baseline_ms else None
                reduction = (1 - ms / baseline_ms) * 100 if baseline_ms else None
                print(f"    {ms:.6f} ms  {speedup:.4f}x  "
                      f"latency_reduction={reduction:+.2f}%", flush=True)
                status = "OK"
            except Exception as exc:
                status = f"FAILED: {exc}"
                ms = speedup = reduction = None
                print(f"    {status}", flush=True)
            rows.append({
                "shape": sid, "B": B, "S": S, "Hq": Hq, "Hkv": Hkv,
                "D": D, "dtype": DTYPE, "variant": label, "origin": origin,
                "overrides": repr(knobs), "kernel": actual,
                "latency_ms": f"{ms:.9f}" if ms is not None else "",
                "speedup_vs_default": f"{speedup:.6f}" if speedup is not None else "",
                "latency_reduction_pct": f"{reduction:.4f}" if reduction is not None else "",
                "status": status,
            })
        if not args.list_only:
            del q, k, v, out
            torch.cuda.empty_cache()

    if args.list_only:
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {args.out}", flush=True)
    return 1 if any(r["status"] != "OK" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())