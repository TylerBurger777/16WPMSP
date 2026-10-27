// quill_core.hpp — shared sorting core for the Quill companion accelerators.
// ---------------------------------------------------------------------------
// Header-only C++17. Portable across macOS (arm64/x86_64) and Linux
// (x86_64/aarch64). No third-party deps — only <thread> from the stdlib for the
// parallel paths. Every kernel sorts a *value-only* contiguous buffer ascending
// and is a drop-in for numpy's np.sort on NaN-free numeric data.
//
// The Quill dispatcher (quill/_backends.py) guarantees, before it ever calls a
// kernel here:
//   * the buffer is C-contiguous and writeable,
//   * NaN has already been stripped (float kernels never see NaN),
//   * the dtype matches the entry point invoked.
// So the kernels below assume those invariants and do not re-validate them.
//
// Design
// ======
//   * Integers/floats are mapped to an order-preserving unsigned key (Herf's
//     float trick; sign-bit flip for signed ints) so a single unsigned LSD
//     radix sorts every dtype correctly, including negatives, ±inf and ±0.
//   * Serial path: 8-bit LSD radix (radix_sort_u*).
//   * Parallel path: one generic partition-and-conquer framework
//     (parallel_partition_sort) shared by BOTH the radix backend (bucket = top
//     byte of the encoded key) and the samplesort backend (bucket = splitter
//     rank). Buckets are globally value-ordered, so each is sorted
//     independently in parallel with no merge step.
#pragma once

#include <cstdint>
#include <cstring>
#include <cstddef>
#include <vector>
#include <thread>
#include <algorithm>
#include <atomic>
#include <type_traits>

namespace quillcore {

// ── unsigned-key encode / decode (order preserving) ────────────────────────
// enc(x) is monotonic non-decreasing in x, so sorting enc(x) as an unsigned
// integer sorts x ascending. dec(enc(x)) == x for every finite value.

inline uint32_t enc(uint32_t x) { return x; }
inline uint64_t enc(uint64_t x) { return x; }
inline uint32_t dec_u32(uint32_t u) { return u; }
inline uint64_t dec_u64(uint64_t u) { return u; }

inline uint32_t enc(int32_t x) { return (uint32_t)x ^ 0x80000000u; }
inline int32_t  dec_i32(uint32_t u) { return (int32_t)(u ^ 0x80000000u); }
inline uint64_t enc(int64_t x) { return (uint64_t)x ^ 0x8000000000000000ull; }
inline int64_t  dec_i64(uint64_t u) { return (int64_t)(u ^ 0x8000000000000000ull); }

inline uint32_t enc(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    uint32_t mask = (uint32_t)(-(int32_t)(u >> 31)) | 0x80000000u;
    return u ^ mask;
}
inline float dec_f32(uint32_t u) {
    uint32_t mask = ((u >> 31) - 1u) | 0x80000000u;
    uint32_t r = u ^ mask; float f; std::memcpy(&f, &r, 4); return f;
}
inline uint64_t enc(double f) {
    uint64_t u; std::memcpy(&u, &f, 8);
    uint64_t mask = (uint64_t)(-(int64_t)(u >> 63)) | 0x8000000000000000ull;
    return u ^ mask;
}
inline double dec_f64(uint64_t u) {
    uint64_t mask = ((u >> 63) - 1ull) | 0x8000000000000000ull;
    uint64_t r = u ^ mask; double f; std::memcpy(&f, &r, 8); return f;
}

// Map a value type T to the matching unsigned key type.
template <class T> struct key_of;
template <> struct key_of<int32_t>  { using U = uint32_t; };
template <> struct key_of<uint32_t> { using U = uint32_t; };
template <> struct key_of<float>    { using U = uint32_t; };
template <> struct key_of<int64_t>  { using U = uint64_t; };
template <> struct key_of<uint64_t> { using U = uint64_t; };
template <> struct key_of<double>   { using U = uint64_t; };

template <class T> inline T dec(typename key_of<T>::U u);
template <> inline int32_t  dec<int32_t>(uint32_t u)  { return dec_i32(u); }
template <> inline uint32_t dec<uint32_t>(uint32_t u) { return dec_u32(u); }
template <> inline float    dec<float>(uint32_t u)    { return dec_f32(u); }
template <> inline int64_t  dec<int64_t>(uint64_t u)  { return dec_i64(u); }
template <> inline uint64_t dec<uint64_t>(uint64_t u) { return dec_u64(u); }
template <> inline double   dec<double>(uint64_t u)   { return dec_f64(u); }

// ── serial 8-bit LSD radix over an unsigned array ──────────────────────────
// Sorts a[0..n) ascending. `tmp` must have room for n elements. On return the
// sorted data is in `a` (sizeof(UINT) passes is even → ends in the source).
template <class UINT>
void radix_sort_u(UINT* a, size_t n, UINT* tmp) {
    if (n < 2) return;
    constexpr int PASSES = sizeof(UINT);
    UINT* src = a; UINT* dst = tmp;
    for (int p = 0; p < PASSES; ++p) {
        const int shift = p * 8;
        size_t count[256] = {0};
        for (size_t i = 0; i < n; ++i) count[(src[i] >> shift) & 0xFF]++;
        // Skip a pass whose digit is identical for every element (common for
        // the high bytes of small-magnitude data) — keeps data in `src`.
        size_t nonzero = 0, seen = 0;
        for (int b = 0; b < 256; ++b) if (count[b]) { nonzero = b; seen++; }
        if (seen <= 1) { (void)nonzero; continue; }
        size_t off[256]; size_t sum = 0;
        for (int b = 0; b < 256; ++b) { off[b] = sum; sum += count[b]; }
        for (size_t i = 0; i < n; ++i) dst[off[(src[i] >> shift) & 0xFF]++] = src[i];
        std::swap(src, dst);
    }
    if (src != a) std::memcpy(a, src, n * sizeof(UINT));
}

// ── serial value sort for any supported dtype ──────────────────────────────
template <class T>
void serial_sort(T* a, size_t n) {
    using U = typename key_of<T>::U;
    if (n < 2) return;
    // Small inputs: std::sort beats radix setup cost.
    if (n < 256) { std::sort(a, a + n); return; }
    std::vector<U> keys(n), tmp(n);
    for (size_t i = 0; i < n; ++i) keys[i] = enc(a[i]);
    radix_sort_u<U>(keys.data(), n, tmp.data());
    for (size_t i = 0; i < n; ++i) a[i] = dec<T>(keys[i]);
}

// ── thread-count helper ────────────────────────────────────────────────────
inline int default_threads() {
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 4;
    if (hc > 64) hc = 64;
    return (int)hc;
}

// ── generic parallel partition-and-conquer ─────────────────────────────────
// Splits [0,n) into K globally value-ordered buckets using `bucket_of`, scatters
// in parallel, then sorts each bucket independently with `leaf_sort`. No merge:
// bucket b's every element compares <= bucket b+1's, so concatenation is sorted.
//
//   bucket_of(const T&)  -> int in [0,K)   (monotone in value across buckets)
//   leaf_sort(T* lo, T* hi)                (sorts a sub-range in place)
template <class T, class BucketFn, class LeafFn>
void parallel_partition_sort(T* a, size_t n, int K,
                             BucketFn bucket_of, LeafFn leaf_sort,
                             int nthreads) {
    if (nthreads < 1) nthreads = 1;
    if (n < (size_t)(1 << 15) || nthreads == 1 || K < 2) {
        leaf_sort(a, a + n);
        return;
    }
    int P = nthreads;
    if ((size_t)P > n) P = (int)n;

    // chunk boundaries
    std::vector<size_t> cstart(P + 1);
    for (int p = 0; p <= P; ++p) cstart[p] = (size_t)((__int128)n * p / P);

    // per-thread histograms: hist[p*K + b]
    std::vector<size_t> hist((size_t)P * K, 0);
    {
        std::vector<std::thread> th;
        for (int p = 0; p < P; ++p) {
            th.emplace_back([&, p]() {
                size_t* h = &hist[(size_t)p * K];
                for (size_t i = cstart[p]; i < cstart[p + 1]; ++i)
                    h[bucket_of(a[i])]++;
            });
        }
        for (auto& t : th) t.join();
    }

    // global bucket base offsets + per-(thread,bucket) starting cursors
    std::vector<size_t> base(K + 1, 0);
    for (int b = 0; b < K; ++b) {
        size_t tot = 0;
        for (int p = 0; p < P; ++p) tot += hist[(size_t)p * K + b];
        base[b + 1] = base[b] + tot;
    }
    std::vector<size_t> cursor((size_t)P * K);
    for (int b = 0; b < K; ++b) {
        size_t run = base[b];
        for (int p = 0; p < P; ++p) {
            cursor[(size_t)p * K + b] = run;
            run += hist[(size_t)p * K + b];
        }
    }

    // parallel scatter into tmp
    std::vector<T> tmp(n);
    {
        std::vector<std::thread> th;
        for (int p = 0; p < P; ++p) {
            th.emplace_back([&, p]() {
                size_t* cur = &cursor[(size_t)p * K];
                for (size_t i = cstart[p]; i < cstart[p + 1]; ++i) {
                    const T v = a[i];
                    tmp[cur[bucket_of(v)]++] = v;
                }
            });
        }
        for (auto& t : th) t.join();
    }

    // parallel leaf-sort of buckets (atomic work-stealing over bucket ids)
    {
        std::atomic<int> next{0};
        std::vector<std::thread> th;
        int workers = std::min(P, K);
        for (int w = 0; w < workers; ++w) {
            th.emplace_back([&]() {
                for (;;) {
                    int b = next.fetch_add(1);
                    if (b >= K) break;
                    size_t lo = base[b], hi = base[b + 1];
                    if (hi - lo > 1) leaf_sort(&tmp[lo], &tmp[hi]);
                }
            });
        }
        for (auto& t : th) t.join();
    }

    std::memcpy(a, tmp.data(), n * sizeof(T));
}

// ── parallel radix backend (bucket = top byte of the encoded key) ──────────
template <class T>
void parallel_radix(T* a, size_t n, int nthreads = 0) {
    using U = typename key_of<T>::U;
    if (nthreads <= 0) nthreads = default_threads();
    constexpr int TOPSHIFT = (sizeof(U) - 1) * 8;
    auto bucket_of = [](const T& v) -> int {
        return (int)((enc(v) >> TOPSHIFT) & 0xFF);
    };
    auto leaf = [](T* lo, T* hi) { serial_sort<T>(lo, (size_t)(hi - lo)); };
    parallel_partition_sort<T>(a, n, 256, bucket_of, leaf, nthreads);
}

// ── parallel samplesort backend (comparison based) ─────────────────────────
template <class T>
void parallel_samplesort(T* a, size_t n, int nthreads = 0) {
    if (nthreads <= 0) nthreads = default_threads();
    if (n < (size_t)(1 << 15) || nthreads == 1) { serial_sort<T>(a, n); return; }

    const int K = std::min<int>(256, std::max<int>(2, nthreads * 8));
    // Oversample, sort the sample, pick K-1 splitters.
    const int oversample = 8;
    size_t S = (size_t)(K - 1) * oversample;
    if (S >= n) { serial_sort<T>(a, n); return; }
    std::vector<T> sample(S);
    // deterministic stride sampling (no RNG → reproducible, resume-safe)
    for (size_t i = 0; i < S; ++i) sample[i] = a[(size_t)((__int128)n * (i + 1) / (S + 1))];
    std::sort(sample.begin(), sample.end());
    std::vector<T> split(K - 1);
    for (int i = 0; i < K - 1; ++i) split[i] = sample[(size_t)(i + 1) * oversample - 1];
    // bucket = number of splitters strictly less than v (upper_bound rank)
    auto bucket_of = [&](const T& v) -> int {
        return (int)(std::upper_bound(split.begin(), split.end(), v) - split.begin());
    };
    auto leaf = [](T* lo, T* hi) { std::sort(lo, hi); };
    parallel_partition_sort<T>(a, n, K, bucket_of, leaf, nthreads);
}

} // namespace quillcore
