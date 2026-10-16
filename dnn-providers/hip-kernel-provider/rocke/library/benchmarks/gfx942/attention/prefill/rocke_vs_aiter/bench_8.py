#!/usr/bin/env python3
"""Compare dispatch-default rocKE against explicitly named rocKE kernels.

From rocke/library/builders/gfx942/attention/prefill/:
  python3 rocke_multi_kernel_bench.py
  python3 rocke_multi_kernel_bench.py --kernel 'rocke_attention_dense_...'
  python3 rocke_multi_kernel_bench.py --kernel NAME1 --kernel NAME2
  python3 rocke_multi_kernel_bench.py --sweep-dir ./sweep_results_compact

For every selected shape, benchmark the dispatch default FIRST. Named kernels are
reconstructed from their gfx942 name tags and accepted only when the generated
kernel name matches the requested name EXACTLY. A sweep directory supplies the
full spec knobs directly, without reverse-parsing its recorded kernel name.
"""

import argparse
import csv
import math
import re
import sys
from dataclasses import replace
from pathlib import Path
import json

import torch

# Make imports work when invoked from the prefill directory, without depending
# on a particular PYTHONPATH in the virtual environment.
for location in (Path(__file__).resolve().parent, Path.cwd()):
    for parent in (location, *location.parents):
        if (parent / "library").is_dir() and (parent / "platform/python").is_dir():
            sys.path[:0] = [str(parent / "platform/python"), str(parent / "library")]
            break
    else:
        continue
    break

from builders.gfx942.attention.prefill.attention_dense_prefill import (
    dense_request,
    resolve_dense_spec,
    _make_launcher,
    _launch_config,
)
from kernels.gfx942 import attention_dense as dense
from rocke.runtime import time_launches

D = 128
DTYPE = "bf16"
SHAPES = [
    (1, 4096, 32, 8),
    (1, 4096, 32, 16),
    (1, 8192, 32, 8),
    (1, 8192, 32, 16),
    (1, 16384, 32, 8),
    (16, 4096, 32, 8),
    (16, 8192, 32, 8),
    (16, 4096, 32, 16),
]


def kernel_name(spec):
    return dense.gfx942_kernel_name(spec)


def baseline_spec(B, S, Hq, Hkv):
    ns = argparse.Namespace(use_sinks=False)
    req = dense_request(
        ns,
        batch=B,
        seqlen_q=S,
        seqlen_kv=S,
        num_query_heads=Hq,
        num_kv_heads=Hkv,
        head_size=D,
        causal=True,
        dtype=DTYPE,
    )
    spec = resolve_dense_spec(req, {})
    check_supported(spec)
    return spec


def check_supported(spec):
    ok, reason = dense.supports_attention_dense(spec, arch="gfx942")
    if not ok:
        raise ValueError(f"unsupported rocKE spec: {reason}")


def runtime_args(spec):
    """Use the in-tree ABI helper, with a fallback for older ROCm branches."""
    helper = getattr(dense, "attention_dense_runtime_args", None)
    if callable(helper):
        return dict(helper(spec))

    names = tuple(getattr(spec, "runtime_kernarg_fields", ()))
    if not names:
        if getattr(spec, "runtime_shape", False):
            raise RuntimeError("runtime-shape kernel has no runtime ABI helper/metadata")
        return {}

    values = {n: int(getattr(spec, n)) for n in names if hasattr(spec, n)}
    if set(names) - set(values):
        fastdiv = getattr(dense, "_fastdiv_kernargs", None)
        if callable(fastdiv):
            values.update(fastdiv(spec))
        elif getattr(spec, "fastdiv_divisors", ()):
            from rocke.helpers.transforms import calculate_magic_numbers

            nqb = spec.seqlen_q // spec.block_m
            divisors = {
                "batch": spec.batch,
                "num_query_heads": spec.num_query_heads,
                "gqa": spec.num_queries_per_kv,
                "nqb": nqb,
            }
            for name in spec.fastdiv_divisors:
                magic, shift = calculate_magic_numbers(divisors[name])
                values[name + "_magic"] = magic - (1 << 32) if magic >= (1 << 31) else magic
                values[name + "_shift"] = shift

    missing = set(names) - set(values)
    if missing:
        raise RuntimeError(f"missing runtime kernargs: {sorted(missing)}")
    return {n: values[n] for n in names}


def name_number(name, tag, base=10):
    match = re.search(r"(?:^|_)" + re.escape(tag) + r"([0-9a-f]+)(?=_|$)", name)
    return int(match.group(1), base) if match else None


def named_spec(base, requested):
    """Reverse known gfx942 name tags; never silently benchmark a different name.

    Name tags are an *identity*, not a generic spec deserializer: unknown tags
    cannot be interpreted reliably. Exact round-trip checking is mandatory.
    """
    if not requested.startswith("rocke_attention_dense_"):
        return None

    bm = name_number(requested, "bm")
    bn = name_number(requested, "bn")
    wpe = name_number(requested, "wpe")
    persistent_count = name_number(requested, "persist")
    if bn is None or wpe is None:
        return None

    pad = name_number(requested, "krowpad")
    exp = name_number(requested, "e2f")
    prio = name_number(requested, "prio")
    iglp_mode = name_number(requested, "iglpm")
    store_width = name_number(requested, "osw")
    fence = name_number(requested, "fence", 16)  # kernel-name masks are hex
    cfvst = name_number(requested, "cfvst")

    overrides = {
        "block_m": bm if bm is not None else 256,
        "block_n": bn,
        "waves_per_eu": wpe,
        "lds_row_pad": pad if pad is not None else 8,
        "pv_loop_order": "k_major" if "_pvkmaj" in requested else "d_major",
        "pv_priority": prio if prio is not None else 0,
        "pv_sched_fence_mask": fence,
        "iglp": "_iglp1" in requested,
        "iglp_mode": iglp_mode if iglp_mode is not None else 0,
        "causal_diag_split": "_dsplit" in requested,
        "o_store_width": store_width if store_width is not None else 4,
        "use_exp2_fast": bool(exp) if exp is not None else None,
        "persistent": persistent_count is not None,
    }
    if persistent_count is not None:
        overrides["num_persistent"] = persistent_count
        overrides["persist_decode"] = (
            "hkv_major" if "_hkvmaj" in requested else "qb_major"
        )
    # Only override the CFVST policy if the name explicitly forces a setting.
    if cfvst is not None:
        overrides["use_cfvst"] = bool(cfvst)
    # PR branches may encode the KDB+VPF opt-in this way; source policy still
    # decides whether it is actually supported. Never infer that a tag alone
    # enables an optimization if the rebuilt symbol does not match.
    if "_kdbvpf1" in requested:
        overrides["lds_num_buffers"] = 2
        overrides["use_cfvst"] = True

    try:
        spec = replace(base, **overrides)
        if kernel_name(spec) != requested:
            return None
        check_supported(spec)
        return spec
    except (ValueError, TypeError, AttributeError):
        return None


def read_sweep_winner(sweep_dir, B, S, Hq, Hkv):
    if sweep_dir is None:
        return None
    path = sweep_dir / f"B{B}_S{S}_Hq{Hq}_Hkv{Hkv}_D{D}" / "final_validation.json"
    if not path.is_file():
        return None
    best = json.loads(path.read_text())["summary"].get("best_nonbaseline")
    if best is None:
        return None
    return best["knobs"], best["kernel"]


def benchmark_rocke(spec, q, k, v, out, warmup, repeat):
    launcher = _make_launcher(spec)
    stream = int(torch.cuda.current_stream().cuda_stream)
    cfg = _launch_config(spec, stream)
    args = {
        "q_ptr": q,
        "k_ptr": k,
        "v_ptr": v,
        "o_ptr": out,
        "scale": 1.0 / math.sqrt(D),
    }
    args.update(runtime_args(spec))

    def call():
        launcher(args, config=cfg)

    # Check the actual launch path before timing; time_launches handles warmup.
    call()
    torch.cuda.synchronize()
    ms = float(time_launches(call, warmup=warmup, iters=repeat, stream=stream))
    torch.cuda.synchronize()
    if not math.isfinite(ms) or ms <= 0:
        raise ValueError(f"invalid rocKE timing: {ms}")
    return ms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernel", action="append", default=[], metavar="FULL_NAME",
                    help="Full rocKE kernel symbol; repeat for additional kernels")
    ap.add_argument("--sweep-dir", type=Path,
                    help="Optional sweep_results_compact folder; use per-shape winning knobs")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("bench_8_shapes_rocke/rocke_kernels.csv"))
    args = ap.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        ap.error("--warmup must be >=0 and --repeat must be >=1")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("rocKE-only gfx942 attention benchmark (BF16 D128 causal)", flush=True)
    print(f"warmup={args.warmup}, repeat={args.repeat}; default always first", flush=True)

    rows = []
    matched_names = set()
    for idx, (B, S, Hq, Hkv) in enumerate(SHAPES, 1):
        print(f"\n[{idx}/{len(SHAPES)}] B={B} S={S} Hq={Hq} Hkv={Hkv}", flush=True)
        base = baseline_spec(B, S, Hq, Hkv)
        # Order matters: default baseline FIRST for each shape.
        variants = [("default", base)]
        seen = {kernel_name(base)}
        for requested in args.kernel:
            spec = named_spec(base, requested)
            if spec is None:
                continue
            matched_names.add(requested)
            if kernel_name(spec) not in seen:
                variants.append(("named", spec))
                seen.add(kernel_name(spec))

        winner = read_sweep_winner(args.sweep_dir, B, S, Hq, Hkv)
        if winner is not None:
            knobs, winner_name = winner
            spec = replace(base, **knobs)
            check_supported(spec)
            actual = kernel_name(spec)
            if actual != winner_name:
                raise RuntimeError(
                    f"Sweep kernel mismatch for B{B} S{S}: expected {winner_name}, "
                    f"rebuilt {actual}; check checkout/revision before benchmarking"
                )
            if actual not in seen:
                variants.append(("sweep_best", spec))
                seen.add(actual)

        torch.manual_seed(0)
        q = (torch.randn(B, S, Hq, D, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        k = (torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        v = (torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        out = torch.empty_like(q)

        baseline_ms = None
        for label, spec in variants:
            name = kernel_name(spec)
            try:
                ms = benchmark_rocke(spec, q, k, v, out, args.warmup, args.repeat)
                if label == "default":
                    baseline_ms = ms
                speedup = baseline_ms / ms if baseline_ms else None
                pct = (1 - ms / baseline_ms) * 100 if baseline_ms else None
                print(f"  {label:10} {ms:10.6f} ms   speedup={speedup:.4f}x  "
                      f"latency_reduction={pct:+.2f}%", flush=True)
                print(f"    {name}", flush=True)
                status = "OK"
            except Exception as exc:
                ms, speedup, pct = None, None, None
                status = f"FAILED: {exc}"
                print(f"  {label:10} {status}", flush=True)
            rows.append({
                "shape_idx": idx, "B": B, "S": S, "Hq": Hq, "Hkv": Hkv,
                "D": D, "dtype": DTYPE, "causal": True,
                "variant": label, "rocke_kernel": name,
                "rocke_ms": f"{ms:.9f}" if ms is not None else "",
                "speedup_vs_default": f"{speedup:.6f}" if speedup is not None else "",
                "latency_reduction_pct": f"{pct:.4f}" if pct is not None else "",
                "status": status,
            })

        del q, k, v, out
        torch.cuda.empty_cache()

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV: {args.out}", flush=True)
    unmatched = set(args.kernel) - matched_names
    if unmatched:
        print("\nRequested names not produced for any of the eight shapes:", flush=True)
        for name in sorted(unmatched):
            print(f"  {name}", flush=True)
        print("Kernel names must match the rebuilt spec on this source revision exactly.", flush=True)
    return 1 if unmatched or any(r["status"] != "OK" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())