#!/usr/bin/env python3
"""
Simple MI300X/gfx942 attention benchmark.

Compares:
  - ROCKE dense attention
  - CK Tile FMHA
  - AITER FMHA-v3 ASM

Assumes the existing setup script has already created:
  ../aiter
  ../env/rocke_env.sh
  ../env/aiter_env.sh
  ../env/ck_env.sh

Run this script from the ROCKE Python environment.
"""

import argparse
import csv
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


WARMUP = int(os.environ.get("WARMUP", "10"))
REPEAT = int(os.environ.get("REPEAT", "50"))
D = 128

CONFIGS = [
    (1, 1, 4096, 32, 8),
    (2, 1, 4096, 32, 16),
    (3, 1, 8192, 32, 8),
    (4, 1, 8192, 32, 16),
    (5, 1, 16384, 32, 8),
    (6, 16, 4096, 32, 8),
    (7, 16, 8192, 32, 8),
    (8, 16, 4096, 32, 16),
    (9, 64, 4096, 32, 8),
    (10, 64, 8192, 32, 8),
]

SCRIPT_DIR = Path(__file__).resolve().parent
ROCM_LIBS = Path(
    subprocess.check_output(
        ["git", "-C", str(SCRIPT_DIR), "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
)
ROOT = ROCM_LIBS.parent
HIP_KERNEL_PROVIDER = ROCM_LIBS / "dnn-providers/hip-kernel-provider"
AITER = ROOT / "aiter"
CK = AITER / "3rdparty/composable_kernel"

ROCKE_ENV = ROOT / "env/rocke_env.sh"
AITER_ENV = ROOT / "env/aiter_env.sh"
CK_ENV = ROOT / "env/ck_env.sh"

AITER_EXE = AITER / "op_tests/cpp/mha/fwd.exe"
CK_EXE = CK / "build/bin/tile_example_fmha_fwd"

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(HIP_KERNEL_PROVIDER))

from builders.gfx942.attention.prefill.attention_dense_prefill import (  # noqa: E402
    dense_request,
    resolve_dense_spec,
    run,
)


ROCKE_ARGS = argparse.Namespace(
    persistent=None,
    num_persistent=None,
    persist_decode=None,
    block_n=None,
    waves_per_eu=None,
    interleave=None,
    lds_k_group_pad=None,
    sliding_window=None,
)

PERF_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*ms")


def run_rocke(B, S, Hq, Hkv):
    """Call the ROCKE benchmark API once for this shape."""
    req = dense_request(
        ROCKE_ARGS,
        batch=B,
        seqlen_q=S,
        seqlen_kv=S,
        num_query_heads=Hq,
        num_kv_heads=Hkv,
        head_size=D,
        causal=True,
        dtype="bf16",
    )
    spec = resolve_dense_spec(req, {})
    ms, _, _ = run(
        spec,
        warmup=WARMUP,
        iters=REPEAT,
        check=False,
        overrides={},
    )
    return float(ms)


def run_external(env_file, cwd, command):
    """Run one benchmark executable and return its printed latency."""
    shell_cmd = (
        f"source {shlex.quote(str(env_file))} && "
        f"cd {shlex.quote(str(cwd))} && "
        f"{shlex.join([str(x) for x in command])}"
    )

    output = subprocess.check_output(
        ["bash", "-lc", shell_cmd],
        text=True,
        stderr=subprocess.STDOUT,
    )

    matches = PERF_RE.findall(output)
    if not matches:
        raise RuntimeError(f"No timing found in output:\n{output}")

    return float(matches[-1])


def run_aiter(B, S, Hq, Hkv):
    """Run AITER FMHA-v3 ASM once for this shape."""
    cmd = [
        AITER_EXE,
        "-prec=bf16",
        f"-b={B}",
        f"-h={Hq}",
        f"-h_k={Hkv}",
        f"-d={D}",
        f"-d_v={D}",
        f"-s={S}",
        f"-s_k={S}",
        "-iperm=0",
        "-operm=0",
        "-mask=1",
        "-lse=0",
        "-fwd_v3=1",
        "-v3_bf16_cvt=2",
        "-mode=0",
        "-timer=gpu",
        "-v=0",
        f"-warmup={WARMUP}",
        f"-repeat={REPEAT}",
    ]
    return run_external(AITER_ENV, AITER / "op_tests/cpp/mha", cmd)


def run_ck(B, S, Hq, Hkv):
    """Run CK Tile FMHA once for this shape."""
    cmd = [
        CK_EXE,
        "-v=0",
        "-mode=0",
        f"-b={B}",
        f"-h={Hq}",
        f"-h_k={Hkv}",
        f"-s={S}",
        f"-s_k={S}",
        f"-d={D}",
        f"-d_v={D}",
        "-scale_s=0",
        "-iperm=0",
        "-operm=0",
        "-bias=n",
        "-prec=bf16",
        "-mask=1",
        "-vlayout=r",
        "-lse=0",
        "-num_splits=1",
        f"-warmup={WARMUP}",
        f"-repeat={REPEAT}",
    ]
    return run_external(CK_ENV, CK, cmd)


def try_run(name, fn, B, S, Hq, Hkv):
    """
    Minimal failure handling:
    print one line, record blank, continue to the next implementation/config.
    """
    try:
        return fn(B, S, Hq, Hkv)
    except Exception as exc:
        print(f"  {name}: FAILED ({exc})")
        return None


def fmt(x, digits=4):
    return "—" if x is None else f"{x:.{digits}f}"


def speedup(base, other):
    if base is None or other is None or other == 0:
        return None
    return base / other


def git_head(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main():
    results = []

    for cid, B, S, Hq, Hkv in CONFIGS:
        print(
            f"\n[{cid}/{len(CONFIGS)}] "
            f"B={B} S={S} Hq={Hq} Hkv={Hkv} GQA={Hq // Hkv}:1"
        )

        rocke_ms = try_run("ROCKE", run_rocke, B, S, Hq, Hkv)
        print(f"  ROCKE: {fmt(rocke_ms)} ms")

        ck_ms = try_run("CK", run_ck, B, S, Hq, Hkv)
        print(f"  CK:    {fmt(ck_ms)} ms")

        aiter_ms = try_run("AITER", run_aiter, B, S, Hq, Hkv)
        print(f"  AITER: {fmt(aiter_ms)} ms")

        results.append(
            {
                "id": cid,
                "B": B,
                "S": S,
                "Hq": Hq,
                "Hkv": Hkv,
                "GQA": f"{Hq // Hkv}:1",
                "ROCKE_ms": rocke_ms,
                "CK_ms": ck_ms,
                "AITER_ms": aiter_ms,
                "CK_vs_ROCKE": speedup(rocke_ms, ck_ms),
                "AITER_vs_ROCKE": speedup(rocke_ms, aiter_ms),
                "AITER_vs_CK": speedup(ck_ms, aiter_ms),
            }
        )

    csv_path = RESULTS_DIR / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    md_path = RESULTS_DIR / "RESULTS.md"

    rocm_version = "unknown"
    rocm_version_file = Path("/opt/rocm/.info/version")
    if rocm_version_file.exists():
        rocm_version = rocm_version_file.read_text().strip()

    lines = [
        "# ROCKE vs CK vs AITER Attention Benchmark",
        "",
        f"- GPU target: `gfx942` / MI300X",
        f"- ROCm: `{rocm_version}`",
        f"- BF16, causal, BSHD, D={D}",
        f"- Warmup: {WARMUP}",
        f"- Measured launches: {REPEAT}",
        f"- rocm-libraries commit: `{git_head(ROCM_LIBS)}`",
        f"- AITER commit: `{git_head(AITER)}`",
        f"- CK commit: `{git_head(CK)}`",
        "",
        "| # | B | S | Hq | Hkv | GQA | ROCKE ms | CK ms | AITER ms | CK vs ROCKE | AITER vs ROCKE | AITER vs CK |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in results:
        lines.append(
            f"| {r['id']} | {r['B']} | {r['S']} | {r['Hq']} | {r['Hkv']} | {r['GQA']} | "
            f"{fmt(r['ROCKE_ms'])} | {fmt(r['CK_ms'])} | {fmt(r['AITER_ms'])} | "
            f"{fmt(r['CK_vs_ROCKE'], 2)}× | "
            f"{fmt(r['AITER_vs_ROCKE'], 2)}× | "
            f"{fmt(r['AITER_vs_CK'], 2)}× |"
        )

    md_path.write_text("\n".join(lines) + "\n")

    print(f"\nCSV:      {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
