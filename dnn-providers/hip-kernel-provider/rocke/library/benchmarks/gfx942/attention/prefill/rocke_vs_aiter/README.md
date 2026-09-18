# ROCKE vs CK vs AITER Attention Benchmark — MI300X

This benchmark compares attention performance:

* **ROCKE attention_dense**
* **Composable Kernel (CK) Tile FMHA**
* **AITER FMHA v3 ASM**

## Benchmark Environment

The reported results were collected on an AMD Instinct MI300X-class `gfx942` system using the following software stack:

```text
GPU:                AMD Instinct MI300X
GPU architecture:   gfx942
ROCm wheel version: 10.0.0
PyTorch:            2.13.0+rocm10.0.0
HIP:                7.15.26333
Triton:             3.8.0+git4cff872c.rocm10.0.0
Python:             3.12
```

## Setup

Create a workspace at root:

```bash
mkdir gpu-bench
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
cd gpu-bench/ocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter
``

Run the setup and benchmark scripts:

```bash
chmod +x setup.sh
chmod +x run_benchmark.sh
./setup
./run_benchmark.sh
```
The resulting workspace is approximately:

```text
~/gpu-bench/
├── rocm-libraries/
├── aiter/
│   └── 3rdparty/
│       └── composable_kernel/
├── env/
└── results/
```

## Benchmark Methodology

```text
Warmup iterations: 10
Measured iterations: 50
```


## Results

| # | B | S | Hq | Hkv | GQA | ROCKE ms | CK ms | AITER ms | CK vs ROCKE | AITER vs ROCKE | AITER vs CK |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 4096 | 32 | 8 | 4:1 | 0.6439 | 0.3460 | 0.2770 | 1.86× | 2.32× | 1.25× |
| 2 | 1 | 4096 | 32 | 16 | 2:1 | 0.6434 | 0.3560 | 0.2820 | 1.81× | 2.28× | 1.26× |
| 3 | 1 | 8192 | 32 | 8 | 4:1 | 1.9939 | 1.1750 | 0.9870 | 1.70× | 2.02× | 1.19× |
| 4 | 1 | 8192 | 32 | 16 | 2:1 | 2.0063 | 1.2540 | 0.9780 | 1.60× | 2.05× | 1.28× |
| 5 | 1 | 16384 | 32 | 8 | 4:1 | 7.5652 | 4.5650 | 3.9260 | 1.66× | 1.93× | 1.16× |
| 6 | 16 | 4096 | 32 | 8 | 4:1 | 7.2277 | 5.1220 | 4.1100 | 1.41× | 1.76× | 1.25× |
| 7 | 16 | 8192 | 32 | 8 | 4:1 | 27.8976 | 18.9250 | 15.3450 | 1.47× | 1.82× | 1.23× |
| 8 | 16 | 4096 | 32 | 16 | 2:1 | 7.6793 | 5.4950 | 4.1310 | 1.40× | 1.86× | 1.33× |
| 9 | 64 | 4096 | 32 | 8 | 4:1 | 30.1792 | 20.6850 | 16.6010 | 1.46× | 1.82× | 1.25× |
| 10 | 64 | 8192 | 32 | 8 | 4:1 | — | 76.4110 | 62.0690 | —× | —× | 1.23× |
