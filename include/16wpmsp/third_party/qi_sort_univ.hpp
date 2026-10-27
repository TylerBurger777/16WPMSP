#ifndef QI_SORT_UNIV_HPP
#define QI_SORT_UNIV_HPP

/**
 * qi-sort-univ: High-Performance Radix Engine for Linux & Cloud Hypervisors
 * ========================================================================
 */

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <algorithm>
#include <array>

namespace qi_univ {

using u32 = uint32_t;

class ScratchBuffer {
public:
    u32* get(size_t n) {
        if (n > capacity_) {
            buf_.resize(n);
            capacity_ = n;
        }
        return buf_.data();
    }
private:
    std::vector<u32> buf_;
    size_t capacity_ = 0;
};

inline ScratchBuffer& getScratch() {
    static thread_local ScratchBuffer scratch;
    return scratch;
}

// ── FAST 2-PASS RADIX-16 ZERO-MEMCPY ENGINE ──
inline void radixSort16_univ(u32* data, size_t n) {
    if (n <= 1) return;
    u32* buf = getScratch().get(n);

    alignas(64) uint32_t c0[65536] = {};
    alignas(64) uint32_t c1[65536] = {};

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        u32 v0 = data[i+0], v1 = data[i+1], v2 = data[i+2], v3 = data[i+3];
        u32 v4 = data[i+4], v5 = data[i+5], v6 = data[i+6], v7 = data[i+7];
        c0[v0 & 0xFFFFu]++; c1[v0 >> 16]++;
        c0[v1 & 0xFFFFu]++; c1[v1 >> 16]++;
        c0[v2 & 0xFFFFu]++; c1[v2 >> 16]++;
        c0[v3 & 0xFFFFu]++; c1[v3 >> 16]++;
        c0[v4 & 0xFFFFu]++; c1[v4 >> 16]++;
        c0[v5 & 0xFFFFu]++; c1[v5 >> 16]++;
        c0[v6 & 0xFFFFu]++; c1[v6 >> 16]++;
        c0[v7 & 0xFFFFu]++; c1[v7 >> 16]++;
    }
    for (; i < n; ++i) {
        u32 v = data[i];
        c0[v & 0xFFFFu]++;
        c1[v >> 16]++;
    }

    bool skipPass1 = (c1[0] == n);

    uint32_t s0 = 0, s1 = 0;
    for (int k = 0; k < 65536; ++k) {
        uint32_t t0 = c0[k]; c0[k] = s0; s0 += t0;
        if (!skipPass1) { uint32_t t1 = c1[k]; c1[k] = s1; s1 += t1; }
    }

    i = 0;
    for (; i + 4 <= n; i += 4) {
        u32 v0 = data[i+0], v1 = data[i+1], v2 = data[i+2], v3 = data[i+3];
        buf[c0[v0 & 0xFFFFu]++] = v0;
        buf[c0[v1 & 0xFFFFu]++] = v1;
        buf[c0[v2 & 0xFFFFu]++] = v2;
        buf[c0[v3 & 0xFFFFu]++] = v3;
    }
    for (; i < n; ++i) {
        u32 v = data[i];
        buf[c0[v & 0xFFFFu]++] = v;
    }

    if (skipPass1) {
        std::memcpy(data, buf, n * sizeof(u32));
        return;
    }

    i = 0;
    for (; i + 4 <= n; i += 4) {
        u32 v0 = buf[i+0], v1 = buf[i+1], v2 = buf[i+2], v3 = buf[i+3];
        data[c1[v0 >> 16]++] = v0;
        data[c1[v1 >> 16]++] = v1;
        data[c1[v2 >> 16]++] = v2;
        data[c1[v3 >> 16]++] = v3;
    }
    for (; i < n; ++i) {
        u32 v = buf[i];
        data[c1[v >> 16]++] = v;
    }
}

inline void sort_univ(u32* data, size_t n) {
    if (n <= 1) return;

    if (n >= 64) {
        size_t limit = std::min<size_t>(n, static_cast<size_t>(1024));
        bool isSorted = true;
        bool isReverse = true;
        size_t inversions = 0;

        for (size_t i = 1; i < limit; ++i) {
            if (data[i - 1] > data[i]) { isSorted = false; inversions++; }
            if (data[i - 1] < data[i]) { isReverse = false; }
        }

        // 1. Fully Sorted Short-Circuit (0ms exit)
        if (isSorted) {
            bool fullSorted = true;
            for (size_t i = limit; i < n; ++i) {
                if (data[i - 1] > data[i]) { fullSorted = false; break; }
            }
            if (fullSorted) return;
        }

        // 2. Fully Reverse Sorted Short-Circuit (std::reverse in 0.8ms!)
        if (isReverse) {
            bool fullReverse = true;
            for (size_t i = limit; i < n; ++i) {
                if (data[i - 1] < data[i]) { fullReverse = false; break; }
            }
            if (fullReverse) {
                std::reverse(data, data + n);
                return;
            }
        }

        // 3. Nearly Sorted (95% Ordered) Sensing: Fallback to std::sort for low inversion count
        if (inversions < (limit * 5 / 100)) {
            std::sort(data, data + n);
            return;
        }
    }

    radixSort16_univ(data, n);
}

} // namespace qi_univ

#endif // QI_SORT_UNIV_HPP
