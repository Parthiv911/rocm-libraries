#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

from builders.gfx942.attention.prefill.attention_dense_prefill import (
    dense_request,
    resolve_dense_spec,
    describe_dense_spec,
    _make_launcher,
    _launch_config,
)
from kernels.gfx942.attention_dense import supports_attention_dense
from rocke.runtime import time_launches


ROOT = Path("/root/gpu-bench")
AITER = ROOT / "aiter"
AIT_ENV = ROOT / "env" / "aiter_env.sh"

D = 128
DTYPE = "bf16"

# Exact 8-shape cohort: (B, S, Hq, Hkv)
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


def make_rocke_spec(B, S, Hq, Hkv):
    ns = argparse.Namespace(
        persistent=None,
        num_persistent=None,
        persist_decode=None,
        block_n=None,
        waves_per_eu=None,
        interleave=None,
        lds_k_group_pad=None,
        sliding_window=None,
    )

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

    ok, why = supports_attention_dense(spec, arch="gfx942")
    if not ok:
        raise RuntimeError(f"unsupported ROCKE spec: {why}")

    name = describe_dense_spec(spec)

    if spec.block_n != 64:
        raise RuntimeError(
            f"expected BN64, got block_n={spec.block_n}: {name}"
        )

    if "_kdbvpf1" not in name:
        raise RuntimeError(
            f"expected kdbvpf1 kernel, got: {name}"
        )

    return spec, name


def benchmark_rocke(B, S, Hq, Hkv, warmup, repeat):
    spec, name = make_rocke_spec(B, S, Hq, Hkv)

    torch.manual_seed(0)
    dt = torch.bfloat16

    q = (torch.randn(B, S, Hq, D, dtype=dt, device="cuda") * 0.2).contiguous()
    k = (torch.randn(B, S, Hkv, D, dtype=dt, device="cuda") * 0.2).contiguous()
    v = (torch.randn(B, S, Hkv, D, dtype=dt, device="cuda") * 0.2).contiguous()
    out = torch.zeros(B, S, Hq, D, dtype=dt, device="cuda")

    scale = 1.0 / math.sqrt(D)

    launcher = _make_launcher(spec)
    stream = int(torch.cuda.current_stream().cuda_stream)
    cfg = _launch_config(spec, stream)

    vals = {
        "q_ptr": q,
        "k_ptr": k,
        "v_ptr": v,
        "o_ptr": out,
        "scale": scale,
    }

    def call():
        launcher(vals, config=cfg)

    ms = time_launches(
        call,
        warmup=warmup,
        iters=repeat,
        stream=stream,
    )
    torch.cuda.synchronize()

    del q, k, v, out, launcher
    torch.cuda.empty_cache()

    return ms, name, spec


def benchmark_aiter(B, S, Hq, Hkv, warmup, repeat):
    cmd = f"""
source '{AIT_ENV}'
cd '{AITER / "op_tests/cpp/mha"}'

./fwd.exe \
  -prec=bf16 \
  -b={B} \
  -h={Hq} \
  -h_k={Hkv} \
  -d=128 \
  -d_v=128 \
  -s={S} \
  -s_k={S} \
  -iperm=0 \
  -operm=0 \
  -mask=1 \
  -lse=0 \
  -fwd_v3=1 \
  -v3_bf16_cvt=2 \
  -mode=0 \
  -timer=gpu \
  -kname=1 \
  -v=0 \
  -warmup={warmup} \
  -repeat={repeat}
"""

    p = subprocess.run(
        ["bash", "-lc", cmd],
        text=True,
        capture_output=True,
    )

    output = p.stdout + p.stderr

    if p.returncode != 0:
        raise RuntimeError(
            f"AITER failed with rc={p.returncode}\n{output}"
        )

    # AITER prints lines like:
    # ", 0.282 ms, 487.05 TFlops, ..."
    matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s+ms", output)
    if not matches:
        raise RuntimeError(
            f"could not parse AITER latency\n{output}"
        )

    ms = float(matches[-1])
    return ms, output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument(
        "--out",
        default="bench_8_shapes_kdbvpf/rocke_vs_aiter_8_shapes.csv",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_path.parent / "aiter_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("8-shape ROCKE vs AITER benchmark")
    print("BF16, D=128, causal, Sq=Sk=S")
    print(f"warmup={args.warmup} repeat={args.repeat}")
    print("=" * 80)

    rows = []

    for idx, (B, S, Hq, Hkv) in enumerate(SHAPES, 1):
        print()
        print(
            f"[{idx}/8] B={B} S={S} Hq={Hq} Hkv={Hkv}"
        )

        rocke_ms = None
        aiter_ms = None
        rocke_name = ""
        rocke_status = "OK"
        aiter_status = "OK"

        try:
            rocke_ms, rocke_name, spec = benchmark_rocke(
                B, S, Hq, Hkv, args.warmup, args.repeat
            )
            print(
                f"  ROCKE: {rocke_ms:.6f} ms "
                f"(BN={spec.block_n}, persistent={spec.persistent})"
            )
            print(f"         {rocke_name}")
        except Exception as exc:
            rocke_status = f"FAILED: {exc}"
            print(f"  ROCKE: {rocke_status}")

        try:
            aiter_ms, aiter_output = benchmark_aiter(
                B, S, Hq, Hkv, args.warmup, args.repeat
            )
            (log_dir / f"shape_{idx}.log").write_text(aiter_output)
            print(f"  AITER: {aiter_ms:.6f} ms")
        except Exception as exc:
            aiter_status = f"FAILED: {exc}"
            print(f"  AITER: {aiter_status}")

        ratio = None
        improvement_vs_064 = None

        if rocke_ms is not None and aiter_ms is not None and aiter_ms > 0:
            ratio = rocke_ms / aiter_ms
            print(f"  AITER speedup vs ROCKE: {ratio:.3f}x")

        # Useful for the original ~0.64 ms ROCKE reference only on B1/S4096/Hq32/Hkv8.
        if (B, S, Hq, Hkv) == (1, 4096, 32, 8) and rocke_ms is not None:
            improvement_vs_064 = (0.64 - rocke_ms) / 0.64 * 100.0
            print(
                f"  ROCKE improvement vs 0.640 ms baseline: "
                f"{improvement_vs_064:.2f}%"
            )

        rows.append(
            {
                "idx": idx,
                "B": B,
                "S": S,
                "Hq": Hq,
                "Hkv": Hkv,
                "D": D,
                "dtype": DTYPE,
                "causal": True,
                "rocke_ms": "" if rocke_ms is None else f"{rocke_ms:.9f}",
                "aiter_ms": "" if aiter_ms is None else f"{aiter_ms:.9f}",
                "aiter_speedup_vs_rocke": ""
                if ratio is None
                else f"{ratio:.6f}",
                "rocke_improvement_vs_0p64_pct": ""
                if improvement_vs_064 is None
                else f"{improvement_vs_064:.4f}",
                "rocke_status": rocke_status,
                "aiter_status": aiter_status,
                "rocke_kernel": rocke_name,
            }
        )

    fields = [
        "idx",
        "B",
        "S",
        "Hq",
        "Hkv",
        "D",
        "dtype",
        "causal",
        "rocke_ms",
        "aiter_ms",
        "aiter_speedup_vs_rocke",
        "rocke_improvement_vs_0p64_pct",
        "rocke_status",
        "aiter_status",
        "rocke_kernel",
    ]

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(
        f"{'#':>2} {'B':>3} {'S':>6} {'Hq':>4} {'Hkv':>4} "
        f"{'ROCKE ms':>11} {'AITER ms':>11} {'AITER x':>9}"
    )

    for r in rows:
        print(
            f"{r['idx']:>2} "
            f"{r['B']:>3} "
            f"{r['S']:>6} "
            f"{r['Hq']:>4} "
            f"{r['Hkv']:>4} "
            f"{r['rocke_ms']:>11} "
            f"{r['aiter_ms']:>11} "
            f"{r['aiter_speedup_vs_rocke']:>9}"
        )

    print()
    print(f"CSV: {out_path}")


if __name__ == "__main__":
    main()