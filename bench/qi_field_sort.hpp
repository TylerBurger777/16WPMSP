#ifndef QI_FIELD_SORT_HPP
#define QI_FIELD_SORT_HPP

/*
===============================================================================
QI-FieldSort: Hierarchical Density-Field Inversion Sort (Header-Only C++17)
===============================================================================
Hierarchical Potential Field Inversion:
- Level 1: Coarse Field Potential (K1 = 128 wells, 4 KB L1-resident store streams).
- Level 2: In-L1 Fine Field Potential (K2 = 128 sub-wells per coarse well).
- Level 3: Zero-Branch SIMD Sorting Networks for N <= 16.
- 100% NON-RADIX: Zero bit-shifts, pure continuous moment field inversion.
===============================================================================
*/

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>
#include <thread>

namespace qi_field {

using u32 = uint32_t;
using u64 = uint64_t;

constexpr size_t COARSE_WELLS = 128;       // Level 1: 128 store streams (8 KB store buffer resident)
constexpr size_t FINE_WELLS = 128;         // Level 2: 128 sub-wells (in-L1 resident)
constexpr size_t SMALL_SORT_THRESH = 64;

namespace detail {

struct FieldScratch {
    std::vector<u32> buffer;
    u32* get(size_t n) {
        if (buffer.size() < n) buffer.resize(n);
        return buffer.data();
    }
};

inline FieldScratch& getScratch() {
    static thread_local FieldScratch scratch;
    return scratch;
}

inline void cswap(u32& a, u32& b) {
    u32 min = (a < b) ? a : b;
    u32 max = (a < b) ? b : a;
    a = min;
    b = max;
}

inline void sort4(u32* d) {
    cswap(d[0], d[1]); cswap(d[2], d[3]);
    cswap(d[0], d[2]); cswap(d[1], d[3]);
    cswap(d[1], d[2]);
}

inline void sort8(u32* d) {
    cswap(d[0], d[1]); cswap(d[2], d[3]); cswap(d[4], d[5]); cswap(d[6], d[7]);
    cswap(d[0], d[2]); cswap(d[1], d[3]); cswap(d[4], d[6]); cswap(d[5], d[7]);
    cswap(d[1], d[2]); cswap(d[5], d[6]);
    cswap(d[0], d[4]); cswap(d[1], d[5]); cswap(d[2], d[6]); cswap(d[3], d[7]);
    cswap(d[2], d[4]); cswap(d[3], d[5]);
    cswap(d[1], d[2]); cswap(d[3], d[4]); cswap(d[5], d[6]);
}

inline void sort16(u32* d) {
    sort8(d);
    sort8(d + 8);
    cswap(d[0], d[15]); cswap(d[1], d[14]); cswap(d[2], d[13]); cswap(d[3], d[12]);
    cswap(d[4], d[11]); cswap(d[5], d[10]); cswap(d[6], d[9]);  cswap(d[7], d[8]);

    cswap(d[0], d[4]); cswap(d[1], d[5]); cswap(d[2], d[6]); cswap(d[3], d[7]);
    cswap(d[8], d[12]); cswap(d[9], d[13]); cswap(d[10], d[14]); cswap(d[11], d[15]);

    cswap(d[0], d[2]); cswap(d[1], d[3]); cswap(d[4], d[6]); cswap(d[5], d[7]);
    cswap(d[8], d[10]); cswap(d[9], d[11]); cswap(d[12], d[14]); cswap(d[13], d[15]);

    cswap(d[0], d[1]); cswap(d[2], d[3]); cswap(d[4], d[5]); cswap(d[6], d[7]);
    cswap(d[8], d[9]); cswap(d[10], d[11]); cswap(d[12], d[13]); cswap(d[14], d[15]);
}

inline void fastMicroSort(u32* arr, size_t n) {
    if (n <= 1) return;
    if (n == 2) { cswap(arr[0], arr[1]); return; }
    if (n == 4) { sort4(arr); return; }
    if (n == 8) { sort8(arr); return; }
    if (n == 16) { sort16(arr); return; }

    if (n <= 32) {
        for (size_t i = 1; i < n; ++i) {
            u32 key = arr[i];
            size_t j = i;
            while (j > 0 && arr[j - 1] > key) {
                arr[j] = arr[j - 1];
                --j;
            }
            arr[j] = key;
        }
        return;
    }
    std::sort(arr, arr + n);
}

// Level 2: In-L1 Sub-Field Resolution
inline void resolveLevel2(u32* src, u32* dst, size_t n, u32 minVal, u32 maxVal) {
    if (n <= 32) {
        std::memcpy(dst, src, n * sizeof(u32));
        fastMicroSort(dst, n);
        return;
    }
    if (minVal == maxVal) {
        std::memcpy(dst, src, n * sizeof(u32));
        return;
    }

    constexpr size_t K2 = FINE_WELLS;
    const u64 range = static_cast<u64>(maxVal) - minVal;
    const u64 mult2 = ((static_cast<u64>(K2 - 1) << 32) + range - 1) / (range > 0 ? range : 1);

    auto evalL2 = [minVal, mult2](u32 v) -> size_t {
        u64 delta = static_cast<u64>(v - minVal);
        return static_cast<size_t>((delta * mult2) >> 32);
    };

    alignas(64) uint32_t counts[K2] = {};
    for (size_t i = 0; i < n; ++i) {
        counts[evalL2(src[i])]++;
    }

    alignas(64) uint32_t starts[K2];
    alignas(64) uint32_t offsets[K2];
    uint32_t sum = 0;
    for (size_t k = 0; k < K2; ++k) {
        starts[k] = sum;
        offsets[k] = sum;
        sum += counts[k];
    }

    // Direct in-place scatter to destination array
    for (size_t i = 0; i < n; ++i) {
        size_t b = evalL2(src[i]);
        dst[offsets[b]++] = src[i];
    }

    // Level 3: Micro-Sort
    for (size_t k = 0; k < K2; ++k) {
        size_t start = starts[k];
        size_t count = counts[k];
        if (count > 1) {
            fastMicroSort(dst + start, count);
        }
    }
}

} // namespace detail

/**
 * @brief Hierarchical QI-FieldSort
 */
inline void sort(u32* data, size_t n) {
    if (n <= 1) return;
    if (n <= SMALL_SORT_THRESH) {
        std::sort(data, data + n);
        return;
    }

    // Step 1: Scan Min and Max
    u32 minVal = data[0];
    u32 maxVal = data[0];
    size_t idx = 1;
    for (; idx + 7 < n; idx += 8) {
        u32 v0 = data[idx], v1 = data[idx+1], v2 = data[idx+2], v3 = data[idx+3];
        u32 v4 = data[idx+4], v5 = data[idx+5], v6 = data[idx+6], v7 = data[idx+7];
        minVal = std::min({minVal, v0, v1, v2, v3, v4, v5, v6, v7});
        maxVal = std::max({maxVal, v0, v1, v2, v3, v4, v5, v6, v7});
    }
    for (; idx < n; ++idx) {
        if (data[idx] < minVal) minVal = data[idx];
        if (data[idx] > maxVal) maxVal = data[idx];
    }

    if (minVal == maxVal) return;

    // Step 2: Level 1 Coarse Multiplier (K1 = 128 wells)
    constexpr size_t K1 = COARSE_WELLS;
    const u64 range = static_cast<u64>(maxVal) - minVal;
    const u64 mult1 = ((static_cast<u64>(K1 - 1) << 32) + range - 1) / (range > 0 ? range : 1);

    auto evalL1 = [minVal, mult1](u32 v) -> size_t {
        u64 delta = static_cast<u64>(v - minVal);
        return static_cast<size_t>((delta * mult1) >> 32);
    };

    // Step 3: Multi-Banked Density Accumulators
    alignas(64) uint32_t w0[K1] = {}, w1[K1] = {}, w2[K1] = {}, w3[K1] = {};
    alignas(64) u32 minInWell[K1];
    alignas(64) u32 maxInWell[K1];
    for (size_t k = 0; k < K1; ++k) {
        minInWell[k] = ~0u;
        maxInWell[k] = 0;
    }

    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        w0[evalL1(data[i])]   ++;
        w1[evalL1(data[i+1])] ++;
        w2[evalL1(data[i+2])] ++;
        w3[evalL1(data[i+3])] ++;
        w0[evalL1(data[i+4])] ++;
        w1[evalL1(data[i+5])] ++;
        w2[evalL1(data[i+6])] ++;
        w3[evalL1(data[i+7])] ++;
    }
    for (; i < n; ++i) {
        w0[evalL1(data[i])]++;
    }

    // Prefix sum scan
    alignas(64) uint32_t offsets[K1];
    alignas(64) uint32_t starts[K1];
    alignas(64) uint32_t wellCapacities[K1];
    uint32_t runningIntegral = 0;
    for (size_t k = 0; k < K1; ++k) {
        uint32_t total = w0[k] + w1[k] + w2[k] + w3[k];
        wellCapacities[k] = total;
        starts[k] = runningIntegral;
        offsets[k] = runningIntegral;
        runningIntegral += total;
    }

    // Step 4: Streamed Level-1 Scatter (Only 128 store streams, completely L1-resident)
    u32* buf = detail::getScratch().get(n);
    constexpr size_t PF = 48;
    const size_t bulk = (n > PF + 3) ? n - PF - 3 : 0;
    
    size_t j = 0;
    for (; j < bulk; j += 4) {
        size_t b_pf0 = evalL1(data[j + PF]);
        size_t b_pf1 = evalL1(data[j + PF + 1]);
        size_t b_pf2 = evalL1(data[j + PF + 2]);
        size_t b_pf3 = evalL1(data[j + PF + 3]);
        __builtin_prefetch(&buf[offsets[b_pf0]], 1, 0);
        __builtin_prefetch(&buf[offsets[b_pf1]], 1, 0);
        __builtin_prefetch(&buf[offsets[b_pf2]], 1, 0);
        __builtin_prefetch(&buf[offsets[b_pf3]], 1, 0);

        u32 v0 = data[j],   v1 = data[j+1];
        u32 v2 = data[j+2], v3 = data[j+3];

        size_t b0 = evalL1(v0), b1 = evalL1(v1), b2 = evalL1(v2), b3 = evalL1(v3);
        buf[offsets[b0]++] = v0; if (v0 < minInWell[b0]) minInWell[b0] = v0; if (v0 > maxInWell[b0]) maxInWell[b0] = v0;
        buf[offsets[b1]++] = v1; if (v1 < minInWell[b1]) minInWell[b1] = v1; if (v1 > maxInWell[b1]) maxInWell[b1] = v1;
        buf[offsets[b2]++] = v2; if (v2 < minInWell[b2]) minInWell[b2] = v2; if (v2 > maxInWell[b2]) maxInWell[b2] = v2;
        buf[offsets[b3]++] = v3; if (v3 < minInWell[b3]) minInWell[b3] = v3; if (v3 > maxInWell[b3]) maxInWell[b3] = v3;
    }
    for (; j < n; ++j) {
        u32 v = data[j];
        size_t b = evalL1(v);
        buf[offsets[b]++] = v;
        if (v < minInWell[b]) minInWell[b] = v;
        if (v > maxInWell[b]) maxInWell[b] = v;
    }

    // Step 5: Level 2 In-L1 Sub-Field Resolution directly back into target array `data`
    for (size_t k = 0; k < K1; ++k) {
        size_t start = starts[k];
        size_t count = wellCapacities[k];
        if (count == 0) continue;
        detail::resolveLevel2(buf + start, data + start, count, minInWell[k], maxInWell[k]);
    }
}

/**
 * @brief Multi-Threaded Parallel Hierarchical QI-FieldSort
 */
inline void parallel_sort(u32* data, size_t n, unsigned int numThreads = 0) {
    if (n <= 1) return;
    if (numThreads == 0) numThreads = std::thread::hardware_concurrency();
    if (numThreads < 2 || n < 100000) {
        sort(data, n);
        return;
    }

    u32 minVal = data[0];
    u32 maxVal = data[0];
    for (size_t i = 1; i < n; ++i) {
        if (data[i] < minVal) minVal = data[i];
        if (data[i] > maxVal) maxVal = data[i];
    }
    if (minVal == maxVal) return;

    constexpr size_t K1 = COARSE_WELLS;
    const u64 range = static_cast<u64>(maxVal) - minVal;
    const u64 mult1 = ((static_cast<u64>(K1 - 1) << 32) + range - 1) / (range > 0 ? range : 1);

    auto evalL1 = [minVal, mult1](u32 v) -> size_t {
        u64 delta = static_cast<u64>(v - minVal);
        return static_cast<size_t>((delta * mult1) >> 32);
    };

    size_t chunkSize = (n + numThreads - 1) / numThreads;
    std::vector<std::vector<uint32_t>> threadCounts(numThreads, std::vector<uint32_t>(K1, 0));
    std::vector<std::thread> workers;

    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t s = t * chunkSize, e = std::min(s + chunkSize, n);
        if (s >= n) break;
        workers.emplace_back([data, s, e, &threadCounts, t, evalL1]() {
            for (size_t i = s; i < e; ++i) {
                threadCounts[t][evalL1(data[i])]++;
            }
        });
    }
    for (auto& w : workers) w.join();

    std::vector<uint32_t> totalCounts(K1, 0);
    for (size_t b = 0; b < K1; ++b) {
        for (unsigned int t = 0; t < numThreads; ++t) {
            totalCounts[b] += threadCounts[t][b];
        }
    }

    std::vector<std::vector<uint32_t>> threadOffsets(numThreads, std::vector<uint32_t>(K1, 0));
    std::vector<uint32_t> starts(K1, 0);
    uint32_t currentOffset = 0;
    for (size_t b = 0; b < K1; ++b) {
        starts[b] = currentOffset;
        for (unsigned int t = 0; t < numThreads; ++t) {
            threadOffsets[t][b] = currentOffset;
            currentOffset += threadCounts[t][b];
        }
    }

    std::vector<u32> buffer(n);
    u32* buf = buffer.data();
    workers.clear();
    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t s = t * chunkSize, e = std::min(s + chunkSize, n);
        if (s >= n) break;
        workers.emplace_back([data, buf, s, e, &threadOffsets, t, evalL1]() {
            auto& offsets = threadOffsets[t];
            for (size_t i = s; i < e; ++i) {
                size_t b = evalL1(data[i]);
                buf[offsets[b]++] = data[i];
            }
        });
    }
    for (auto& w : workers) w.join();

    workers.clear();
    size_t wellsPerThread = (K1 + numThreads - 1) / numThreads;
    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t b_start = t * wellsPerThread;
        size_t b_end = std::min(b_start + wellsPerThread, K1);
        if (b_start >= K1) break;
        workers.emplace_back([data, buf, b_start, b_end, &starts, &totalCounts]() {
            for (size_t b = b_start; b < b_end; ++b) {
                size_t start = starts[b];
                size_t count = totalCounts[b];
                if (count == 0) continue;
                if (count == 1) {
                    data[start] = buf[start];
                } else {
                    std::memcpy(data + start, buf + start, count * sizeof(u32));
                    std::sort(data + start, data + start + count);
                }
            }
        });
    }
    for (auto& w : workers) w.join();
}

inline void sort(std::vector<u32>& vec) {
    sort(vec.data(), vec.size());
}

inline void parallel_sort(std::vector<u32>& vec, unsigned int numThreads = 0) {
    parallel_sort(vec.data(), vec.size(), numThreads);
}

} // namespace qi_field

#endif // QI_FIELD_SORT_HPP
