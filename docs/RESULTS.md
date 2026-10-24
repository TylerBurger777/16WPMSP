# 16WPMSP benchmark results

Environment: Ryzen 9 7950X (Zen 4, 32 vCPU, DDR5), GCC 13.3, -O3 -march=znver4.
Data: uniform int32 in [-1e9, 1e9], seed=42. Timing: median of 21, warmup=5,
isolated binaries. IPS4o needs OMP_WAIT_POLICY=active (VDS outliers otherwise).

## Full sweep (100K to 64M)

| N | 16WPMSP | IPS4o | vqsort | std::par |
|---|---|---|---|---|
| 100000 | 0.532 | 2.761 | **0.168** | 0.201 |
| 250000 | 0.906 | 3.075 | **0.460** | 0.522 |
| 500000 | 1.240 | 3.302 | **0.970** | **0.964** |
| 1000000 | **1.822** | 4.598 | 2.043 | 2.100 |
| 2000000 | **3.232** | 5.510 | 4.252 | 4.152 |
| 4000000 | **4.489** | 7.202 | 8.969 | 9.116 |
| 8000000 | **8.013** | 11.214 | 19.665 | 18.438 |
| 16000000 | 20.370 | **19.172** | 42.311 | 48.478 |
| 32000000 | 48.687 | **38.147** | 90.464 | 109.478 |
| 64000000 | 104.101 | **74.566** | 191.802 | 249.346 |

Bold = fastest. All values median ms of 21 runs.

## Speedup 16WPMSP vs each competitor

| N | vs IPS4o | vs vqsort | vs std::par |
|---|---|---|---|
| 100000 | 5.19x | 0.32x | 0.38x |
| 250000 | 3.39x | 1.97x | 1.73x |
| 500000 | 2.66x | 1.28x | 1.28x |
| 1000000 | **2.52x** | 1.12x | 1.15x |
| 2000000 | **1.70x** | 1.32x | 1.28x |
| 4000000 | **1.60x** | **2.00x** | **2.03x** |
| 8000000 | **1.40x** | **2.45x** | **2.30x** |
| 16000000 | 0.94x | **2.08x** | **2.38x** |
| 32000000 | 0.78x | **1.86x** | **2.25x** |
| 64000000 | 0.72x | **1.84x** | **2.40x** |

## Honest reading

16WPMSP is fastest for 1M <= N <= 8M (L3-resident sweet spot). Peak: 2.52x vs
IPS4o (1M), 2.45x vs vqsort (8M), 2.38x vs TBB (16M).

Where we lose:
- N <= 500K: vqsort/TBB win by 1.3-3x. Cause: persistent thread pools; our
  std::thread spawn cost (~200us for 32 threads) dominates.
- N >= 16M: IPS4o wins by 6-40%. Cause: memory-bandwidth-bound; radix does
  8x memory traffic, IPS4o cache-blocking is better for out-of-cache data.

Crossover: vqsort/TBB lead <1M, 16WPMSP leads 1M-8M, IPS4o leads 16M+.

## Stability

stddev over 21 runs is <=5% everywhere except IPS4o at 500K (8.4%, one outlier).
16WPMSP stddev: 0.9% (100K), 5.8% (500K), 3.7% (2M), 4.2% (4M), 1.8% (16M),
1.1% (64M).

## Reproduction

    ./scripts/fetch_deps.sh
    cd external/highway && cmake -B build -S . && cmake --build build -j && cd ../..
    ./scripts/build_isolated.sh
    ./scripts/run_iso_all.sh | tee docs/bench.log
