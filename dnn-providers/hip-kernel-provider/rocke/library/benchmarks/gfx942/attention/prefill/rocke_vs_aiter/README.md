# ROCKE vs CK vs AITER Attention Benchmark — MI300X

This benchmark compares forward attention performance across:

* **ROCKE `attention_dense`**
* **Composable Kernel (CK) Tile FMHA**
* **AITER FMHA v3 ASM**

## Benchmark Environment

The reported results were collected on an AMD Instinct MI300X `gfx942` system using the following software stack:

```text
GPU:                AMD Instinct MI300X
GPU architecture:   gfx942
ROCm:               10.0.0
ROCm wheel version: 10.0.0
PyTorch:            2.13.0+rocm10.0.0
HIP:                7.15.26333
Triton:             3.8.0+git4cff872c.rocm10.0.0
Python:             3.12
```

## Kernels

The benchmark compares the following implementations:

### ROCKE

ROCKE dense prefill attention using:

```text
builders.gfx942.attention.prefill.attention_dense_prefill
```

The benchmark constructs the request using `dense_request`, resolves the kernel specification using `resolve_dense_spec`, and benchmarks it through the ROCKE `run()` API.

### Composable Kernel

CK Tile FMHA using the following selected kernel:

```text
fmha_fwd_d128_bf16_batch_b128x128x32x128x32x128_r4x1x1_r4x1x1_w32x32x16_w32x32x16_qr_async_vr_psddv_nlogits_nbias_mask_nlse_ndropout_nskip_nqscale_ntrload_nsink
```

Only this CK kernel is built for the benchmark.

### AITER

AITER FMHA v3 ASM forward attention using:

```text
fwd_v3=1
v3_bf16_cvt=2
```

The AITER benchmark executable is built from:

```text
op_tests/cpp/mha/benchmark_mha_fwd.cpp
```

## Setup

Create a workspace:

```bash
mkdir -p /root/gpu-bench
cd /root/gpu-bench
```

Clone this ROCm Libraries fork:

```bash
git clone \
  --depth 1 \
  --single-branch \
  --filter=blob:none \
  --sparse \
  --branch develop \
  https://github.com/Parthiv911/rocm-libraries.git

cd rocm-libraries

git sparse-checkout set dnn-providers/hip-kernel-provider
```

Move to the benchmark directory:

```bash
cd /root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter
```

Run the setup and benchmark scripts:

```bash
chmod +x setup run_benchmark.sh

./setup
./run_benchmark.sh
```

The setup script clones and configures AITER and its pinned Composable Kernel submodule, creates the required Python environments, installs the ROCm 10.0.0 wheel stack, and builds the AITER and CK benchmark executables.

The resulting workspace is approximately:

```text
/root/gpu-bench/
├── rocm-libraries/
├── aiter/
│   └── 3rdparty/
│       └── composable_kernel/
├── env/
├── venvs/
│   ├── rocke/
│   └── aiter/
└── rocm-libraries/
    └── dnn-providers/
        └── hip-kernel-provider/
            └── rocke/
                └── library/
                    └── benchmarks/
                        └── gfx942/
                            └── attention/
                                └── prefill/
                                    └── rocke_vs_aiter/
                                        └── results/
```

## Benchmark Methodology

All three implementations use the same attention configurations.

```text
Datatype:            BF16
Attention:           Causal
Head dimension:      128
Query sequence:      Sq = S
Key/value sequence:  Sk = S
Warmup iterations:   10
Measured iterations: 50
```

The benchmark evaluates both 4:1 and 2:1 grouped-query attention configurations using:

```text
Hq = 32

Hkv = 8   -> 4:1 GQA
Hkv = 16  -> 2:1 GQA
```

Each implementation is invoked once per benchmark configuration, with its internal benchmark loop performing 10 warmup iterations followed by 50 measured iterations.

The benchmark records latency in milliseconds and computes relative speedups from the measured latency.

## Source Revisions

The benchmark results were generated using:

```text
rocm-libraries:
60c7600d905cd8322166575d998460747fbf881d

AITER:
cdf6ee88a128c2c160b0512fe37cfb67be161b1d

Composable Kernel:
af9e1d1f1ae347c22feeb08fd2d42645075e0c5d
```

## Results

|  # |  B |     S | Hq | Hkv | GQA | ROCKE ms |   CK ms | AITER ms | CK vs ROCKE | AITER vs ROCKE | AITER vs CK |
| -: | -: | ----: | -: | --: | --: | -------: | ------: | -------: | ----------: | -------------: | ----------: |
|  1 |  1 |  4096 | 32 |   8 | 4:1 |   0.6439 |  0.3460 |   0.2770 |       1.86× |          2.32× |       1.25× |
|  2 |  1 |  4096 | 32 |  16 | 2:1 |   0.6434 |  0.3560 |   0.2820 |       1.81× |          2.28× |       1.26× |
|  3 |  1 |  8192 | 32 |   8 | 4:1 |   1.9939 |  1.1750 |   0.9870 |       1.70× |          2.02× |       1.19× |
|  4 |  1 |  8192 | 32 |  16 | 2:1 |   2.0063 |  1.2540 |   0.9780 |       1.60× |          2.05× |       1.28× |
|  5 |  1 | 16384 | 32 |   8 | 4:1 |   7.5652 |  4.5650 |   3.9260 |       1.66× |          1.93× |       1.16× |
|  6 | 16 |  4096 | 32 |   8 | 4:1 |   7.2277 |  5.1220 |   4.1100 |       1.41× |          1.76× |       1.25× |
|  7 | 16 |  8192 | 32 |   8 | 4:1 |  27.8976 | 18.9250 |  15.3450 |       1.47× |          1.82× |       1.23× |
|  8 | 16 |  4096 | 32 |  16 | 2:1 |   7.6793 |  5.4950 |   4.1310 |       1.40× |          1.86× |       1.33× |
|  9 | 64 |  4096 | 32 |   8 | 4:1 |  30.1792 | 20.6850 |  16.6010 |       1.46× |          1.82× |       1.25× |
| 10 | 64 |  8192 | 32 |   8 | 4:1 |        — | 76.4110 |  62.0690 |           — |              — |       1.23× |

ROCKE did not produce a result for configuration 10, so ROCKE-relative speedups are not reported for that configuration.
