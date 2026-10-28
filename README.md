# 16WPMSP: 16-Way Partitioned Merge Sort Parallel

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![AVX-512](https://img.shields.io/badge/AVX--512-required-orange)
![Platform](https://img.shields.io/badge/platform-Linux%20x86--64-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

**16WPMSP** is a high-performance parallel integer sort for `int32_t` arrays, designed for modern CPU architectures. It combines AVX-512 radix sort with a custom 16-way merge tree to achieve exceptional throughput, outperforming established libraries like **IPS4o**, **Google Highway (vqsort)**, and **TBB** on monolithic architectures.

## Key Features

*   **Hybrid Local Sorting:** Uses `qi::radix_11` (AVX-512) for small chunks and a highly optimized scalar MSD radix for larger chunks to avoid AVX-512 frequency throttling.
*   **Custom 16-Way Merge:** A custom merge tree that uses zero `malloc` in the hot path, avoiding thread contention (unlike standard library approaches).
*   **Adaptive Cache Blocking (TH):** A stepped empirical formula (`TH = per_thread / 166` for large arrays) perfectly tuned for L2/L3 cache sizes.
*   **Persistent ThreadPool V2:** A thread pool where the main thread coordinates and sleeps, eliminating `std::thread` spawn overhead for small-medium sizes.
*   **Extreme Stability:** Achieves standard deviation of `< 0.2%` on arrays up to 1 GB.

---

## Benchmark Results

Environment: AMD Ryzen 7 9700X (Zen 5, 8C/16T, DDR5), GCC 13.3, `-march=native`. 
Data: uniform random `int32_t`, seed=42. Timing: median of 21 runs, isolated binaries.

### AMD Zen 5 (Undisputed Leader)

| N | **16WPMSP** | IPS4o | vqsort (Google) | TBB (std::par) |
|---|---|---|---|---|
| 1M (4 MB) | **0.875 ms** | 2.843 ms | 1.388 ms | 4.213 ms |
| 4M (16 MB) | **3.555 ms** | 7.401 ms | 6.186 ms | 18.949 ms |
| 16M (64 MB) | **20.879 ms** | 28.358 ms | 30.208 ms | 87.985 ms |
| 64M (256 MB) | **88.959 ms** | 114.385 ms | 145.052 ms | 404.582 ms |
| 128M (512 MB) | **180.975 ms** | 230.926 ms | 312.223 ms | N/A |
| 256M (1 GB) | **357.771 ms** | 469.930 ms | 678.470 ms | N/A |

**16WPMSP is 3.3x faster than IPS4o at 1M and scales perfectly up to 1 GB**, beating all competitors by a wide margin.

### Limitations: NUMA & Legacy Architectures (Intel Xeon)

Testing on dual-socket Intel Xeon Gold 6140 (Skylake, 72 threads, DDR4) reveals architectural limits:

| N | 16WPMSP | IPS4o | vqsort | TBB |
|---|---|---|---|---|
| 1M | 5.832 ms | 10.254 ms | 4.871 ms | 4.271 ms |
| 4M | 46.719 ms | **13.818 ms** | 22.705 ms | 19.177 ms |
| 16M | 68.984 ms | **30.168 ms** | 69.153 ms | 68.536 ms |
| 64M | 188.082 ms | **84.494 ms** | 321.639 ms | 270.866 ms |

**Why does it lose on older Intel?**
1.  **AVX-512 Throttling:** Intel Skylake reduces CPU frequency when executing heavy 512-bit instructions. IPS4o uses standard comparisons and runs at full clock speed.
2.  **NUMA Effects:** Dual-socket systems suffer from cross-socket memory latency. Radix Sort makes 8 passes over memory, making it highly sensitive to NUMA. IPS4o uses NUMA-aware allocation via OpenMP.
3.  **Memory Bandwidth:** DDR4 limits throughput compared to modern DDR5.

*Conclusion: 16WPMSP is optimized for modern monolithic architectures (like AMD Zen 5). On legacy multi-socket NUMA systems, in-place algorithms (IPS4o) remain superior.*

---

## Build & Run

```bash
# Clone the repository
git clone https://github.com/TylerBurger777/16WPMSP.git
cd 16WPMSP

# Build the benchmark (Zen 5 / Modern Intel)
g++ -O3 -march=native -std=c++20 -pthread -Iinclude \
    -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vbmi -mavx512vbmi2 \
    -o bench_16wpmsp bench/bench_algo.cpp -DALGO_16WPMSP -ltbb -latomic -fopenmp

# Run on 1 million elements
./bench_16wpmsp --pin 1000000
