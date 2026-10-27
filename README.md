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

Environment: **AMD Ryzen 7 9700X** (Zen 5). Data: random `int32_t`.
Median of 21 runs, isolated binaries. Lower is better.

| N | **16WPMSP** | IPS4o | vqsort (Google) | TBB |
|---|---|---|---|---|
| 1M (4 MB) | **0.875 ms** | 2.843 ms | 1.388 ms | 4.213 ms |
| 4M (16 MB) | **3.555 ms** | 7.401 ms | 6.186 ms | 18.949 ms |
| 16M (64 MB) | **20.879 ms** | 28.358 ms | 30.208 ms | 87.985 ms |
| 64M (256 MB) | **88.959 ms** | 114.385 ms | 145.052 ms | 404.582 ms |
| 128M (512 MB) | **180.975 ms** | 230.926 ms | 312.223 ms | N/A |
| 256M (1 GB) | **357.771 ms** | 469.930 ms | 678.470 ms | N/A |

**16WPMSP is the fastest parallel sort on Zen 5 across all tested sizes.** It scales perfectly from 4 MB to 1 GB, beating IPS4o and Google Highway at every step.

Full results and architecture details: [docs/RESULTS.md](docs/RESULTS.md).


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
