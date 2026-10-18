#!/usr/bin/env bash
set -Eeuo pipefail

# Replays the captured Python invocation in the captured environment.
# The pinned source files and wheel environment must still exist.
export ROCKE_ROOT=/root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke
export ROCKE_BACKEND=python
export ROCKE_LLVM_FLAVOR=llvm23
export ROCKE_COMGR_LIB=/root/gpu-bench/venvs/rocke/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamd_comgr.so.3
export ROCKE_HIP_LIB=/root/gpu-bench/venvs/rocke/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7
export PYTHONPATH=/root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library:/root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/platform/Python:/root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke
export LD_LIBRARY_PATH=/root/gpu-bench/venvs/rocke/lib/python3.12/site-packages/_rocm_sdk_core/lib

exec /root/gpu-bench/venvs/rocke/bin/python /root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/bench_rocke_5way_specs.py --rocke-root /root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke --out-dir /root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/results/5way_specs
