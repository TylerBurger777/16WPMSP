# 16WPMSP benchmark results

Environment: **AMD Ryzen 7 9700X** (Zen 5, 8C/16T, 60GB DDR5), GCC 13.3, `-march=native`.
Data: uniform random `int32_t` in `[-1e9, 1e9]`, seed = 42.
Timing: median of 21 runs, warmup = 5, isolated binaries. `std::is_sorted` verified.

## Complete Sweep: 1M to 256M (Zen 5)

| N | 16WPMSP | IPS4o | vqsort | TBB |
|---|---|---|---|---|
| 1 000 000 (4 MB) | **0.875** | 2.843 | 1.388 | 4.213 |
| 4 000 000 (16 MB) | **3.555** | 7.401 | 6.186 | 18.949 |
| 16 000 000 (64 MB) | **20.879** | 28.358 | 30.208 | 87.985 |
| 64 000 000 (256 MB) | **88.959** | 114.385 | 145.052 | 404.582 |
| 128 000 000 (512 MB) | **180.975** | 230.926 | 312.223 | N/A |
| 256 000 000 (1 GB) | **357.771** | 469.930 | 678.470 | N/A |

*All values in milliseconds, lower is better.*

## Observations

1. **In-Cache & Mid-Range (1M – 16M):** 16WPMSP dominates. At 1M, it is **3.3x faster** than IPS4o.
2. **Out-of-Cache (64M – 256M):** Through adaptive cache blocking (stepped TH formula), 16WPMSP avoids DDR5 bandwidth saturation. We achieve a **21% speedup** at 128M and 256M compared to naive TH, and beat IPS4o by a wide margin.
3. **Stability:** At 128M and 256M, standard deviation is < 0.2%, showing extremely stable performance.

## Architecture summary

*   **Local sort:** `qi::radix_11` (AVX-512) for chunks <= 100k elements.
*   **Merge:** Custom 16-way merge tree (zero `malloc` in hot path).
*   **ThreadPool:** Persistent thread pool for Phase 1 and Phase 2a. Phase 2b uses fresh `std::thread` for clean L1/L2 caches.
*   **Adaptive Threshold (TH):** Stepped function based on empirical cache sizes (100k / 96k / 24k / 48k / 96k).
