#!/usr/bin/env python3
import os
from pathlib import Path

ROOT = Path(os.environ.get("GPU_BENCH_ROOT", "/root/gpu-bench"))
EXE = ROOT / "aiter" / "op_tests" / "cpp" / "mha" / "fwd.exe"

if not EXE.is_file():
    raise SystemExit(f"missing AITER executable: {EXE}")

warmup = os.environ.get("AITER_WARMUP", "0")
repeat = os.environ.get("AITER_REPEAT", "1")

argv = [
    str(EXE),
    "-prec=bf16",
    "-b=1", "-h=32", "-h_k=8",
    "-d=128", "-d_v=128",
    "-s=4096", "-s_k=4096",
    "-iperm=0", "-operm=0",
    "-mask=1", "-lse=0",
    "-fwd_v3=1",
    "-v3_bf16_cvt=0",   # RTNE
    "-mode=0",
    "-timer=gpu",
    "-kname=1",
    "-v=0",
    f"-warmup={warmup}",
    f"-repeat={repeat}",
]

os.chdir(EXE.parent)
os.execv(str(EXE), argv)
