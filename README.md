# 16WPMSP — 16-Way Partitioned Merge Sort Parallel

Parallel integer sort on **AVX-512 + std::thread**, drop-in replacement for
`std::sort` / `pdqsort` / `IPS4o` on `int32_t` arrays of size **>= 50 000**.

* **Adaptive K + TH**
  * `K_eff = pick_K_eff(N, K_max)` — thread count follows a step function
    (1 -> 10 -> 12 -> 16 -> 24 -> K_max), calibrated on Zen 4.
  * `TH = clamp((N / K_eff) / 8, 2000, 50 000)`.
* **Auto core detection**: sched_getaffinity -> sysconf -> hardware_concurrency -> 1.
  Respects `taskset` and cgroups.
* **Any K_max >= 1**, including 6 / 10 / 12 / 24.
* **`NUM_THREADS` is a cap** (`-DNUM_THREADS=0` = auto, default).

## Architecture

Phase 1 (parallel chunks): Radix LSD + 16-way merge tree with register-fused
16->256 bitonic networks and SIMD min via `_mm512_reduce_min_epi32`.

Phase 2a (slab-splitter): sample 1024, 15 splitters, 16 slabs, `lower_bound`
on both ends.

Phase 2b (tree merge): merge-path intersect + `std::merge`, log2(G) levels.

Safety: sequential SIMD scan for INT_MAX/INT_MIN, fallback to std::sort.

## Build

    ./scripts/build.sh             # 16WPMSP only
    ./scripts/fetch_deps.sh        # optional competitors
    ./scripts/build.sh
    ./bench_16wpmsp | tee docs/bench.log

Or: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`
Or: `make -j`

Flags: `-O3 -march=znver4 -std=c++17 -pthread`.

## Benchmark

Isolated binaries, warmup=5 measure=21, is_sorted verified. See docs/RESULTS.md for full table.

| N | 16WPMSP | IPS4o | vqsort | std::par |
|---|---|---|---|---|
| 1M | **1.822** | 4.598 | 2.043 | 2.100 |
| 4M | **4.489** | 7.202 | 8.969 | 9.116 |
| 8M | **8.013** | 11.214 | 19.665 | 18.438 |
| 16M | 20.370 | **19.172** | 42.311 | 48.478 |

16WPMSP wins 1M-8M; IPS4o wins 16M+ (memory-bound); vqsort/TBB win <=500K (persistent thread pool).

## API

    #include "16wpmsp/16wpmsp.hpp"
    std::vector<int> a(N), scratch(N);
    double p1, p2;
    parallel_16wpmsp_sort(a.data(), scratch.data(), N, 0, p1, p2);

`num_threads = 0` -> auto. `> 0` -> upper cap, clamped to real HW.

## Correctness

* int32 only.
* No int64 packing in hot path.
* FIXED_BUF_SIZE = 256 tied to 16-way fusion.
* Sentinel contract: no INT_MAX/INT_MIN in data (scan enforces).
* `sort_local_16wpmsp` can leave result in `a` or `scratch` (merge-level parity);
  K=2 branch reads `a`, writes `scratch`, memcpy back.

## History

Formerly PTMS / PTMS-Hybrid. 4.538 ms on 4M int32 (42x pdqsort). 16WPMSP adds
adaptive K + TH, auto core detection, generalisation to any K_max, benchmark suite.

## License

MIT.
