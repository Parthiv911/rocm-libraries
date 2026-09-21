# ROCKE CFVST A/B Benchmark

This directory contains an A/B benchmark for the **conflict-free V store transpose (CFVST)** path in the ROCKE `attention_dense` prefill kernel on `gfx942`.

The experiment compares the same BF16 attention workload with:

* **A — CFVST disabled**
* **B — CFVST enabled**

The benchmark modifies only the CFVST policy in:

```text
dnn-providers/hip-kernel-provider/rocke/library/kernels/gfx942/attention_dense.py
```

The corresponding source change is also recorded in:

```text
cfvst.patch
```

## Benchmark Shape

```text
B=1
S=4096
Hq=32
Hkv=8
D=128
dtype=BF16
causal=true
layout=BSHD
```


## Directory

The commands below should be run from:

```bash
cd /root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_cfvst_ab
```

The benchmark scripts resolve the repository location automatically.

## Environment Setup

If the ROCKE environment has not already been created, use the setup script in the neighboring `rocke_vs_aiter` benchmark directory:

```bash
../rocke_vs_aiter/setup
```

This creates the ROCKE environment under:

```text
/root/gpu-bench/venvs/rocke
```


## Run the A/B Benchmark

From the `rocke_cfvst_ab` directory:

```bash
chmod +x run_ab_cfvst.sh
./run_ab_cfvst.sh
```

By default, the script performs **5 A/B rounds**.

Each individual measurement uses:

```text
Warmup iterations: 10
Timed iterations: 50
```

You can specify a different number of A/B rounds:

```bash
./run_ab_cfvst.sh 10
```

for 10 rounds.

## Results

| Round | Variant          | CFVST | Latency (ms) |
|------:|------------------|:-----:|-------------:|
| 1 | A_without_cfvst | False | 0.640708 |
| 1 | B_with_cfvst    | True  | 0.583333 |
| 2 | A_without_cfvst | False | 0.642399 |
| 2 | B_with_cfvst    | True  | 0.580008 |
| 3 | A_without_cfvst | False | 0.645111 |
| 3 | B_with_cfvst    | True  | 0.575292 |
| 4 | A_without_cfvst | False | 0.644582 |
| 4 | B_with_cfvst    | True  | 0.582563 |
| 5 | A_without_cfvst | False | 0.645442 |
| 5 | B_with_cfvst    | True  | 0.580152 |

```text
Without CFVST: 0.644582 ms
With CFVST: 0.580152 ms

10% improvement or 1.11x speed up
```
## What the A/B Script Changes

### A — Without CFVST

The benchmark temporarily uses:

```python
return _rows_per_instr(head_size) == 1 and dtype == "fp16"
```

For the BF16 D128 benchmark shape, this resolves to:

```text
cfvst: False
```

### B — With CFVST

The benchmark changes the policy to:

```python
return _rows_per_instr(head_size) == 1
```

For the same BF16 D128 shape, this resolves to:

```text
cfvst: True
```

The script verifies the resolved CFVST state before accepting each timing result.

After the benchmark finishes, the source is left with **CFVST enabled**.

## Benchmark Methodology

`bench_rocke_once.py`:

1. Resolves the ROCKE dense-attention specification.
2. Compiles the `attention_dense` kernel for `gfx942`.
3. Allocates BF16 Q, K, V, and output tensors.
4. Executes 10 warmup iterations.
5. Executes 50 timed kernel launches.
6. Reports the average kernel latency as:

```text
RESULT_MS=<latency>
```

`run_ab_cfvst.sh` alternates between the CFVST-disabled and CFVST-enabled variants and records every measurement.

## Results

Results are written to:

```text
results/
```

The main CSV is:

```text
results/results.csv
```

Format:

```text
round,variant,cfvst,ms
```

Example:

```text
1,A_without_cfvst,False,0.640708
1,B_with_cfvst,True,0.583333
2,A_without_cfvst,False,0.642399
2,B_with_cfvst,True,0.580008
```

The script also prints:

* individual A/B measurements
* median latency without CFVST
* median latency with CFVST
* latency reduction
* speedup

Individual run logs are stored under:

```text
results/
```

and the combined log is:

```text
results/all_runs.log
```

## Files

```text
rocke_cfvst_ab/
├── README.md
├── bench_rocke_once.py
├── cfvst.patch
├── run_ab_cfvst.sh
└── results/
    ├── results.csv
    ├── all_runs.log
    └── individual A/B run logs
```

## Quick Reproduction

From a fresh checkout on a `gfx942` system:

```bash
cd /root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_cfvst_ab

../rocke_vs_aiter/setup

chmod +x run_ab_cfvst.sh
./run_ab_cfvst.sh
```

If the ROCKE environment has already been created, only:

```bash
cd /root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_cfvst_ab

./run_ab_cfvst.sh
```

is required.
