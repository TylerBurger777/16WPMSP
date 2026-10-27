#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <immintrin.h>

#if __has_include("qi_radix.hpp")
#  include "qi_radix.hpp"
#  define HAVE_QI 1
#endif
#if __has_include("quill_core.hpp")
#  include "quill_core.hpp"
#  define HAVE_QUILL 1
#endif

using Clock = std::chrono::high_resolution_clock;
using std::chrono::duration;

void msd_radix_scalar(int* a, int* tmp, int left, int right, int shift) {
    if (right - left < 64) { std::sort(a + left, a + right + 1); return; }
    if (shift < 0) return;
    int count[16] = {0};
    for (int i = left; i <= right; ++i) count[(a[i] >> shift) & 0xF]++;
    int offset[16]; int sum = 0;
    for (int i = 0; i < 16; ++i) { offset[i] = left + sum; sum += count[i]; }
    for (int i = left; i <= right; ++i) { int v = a[i]; tmp[offset[(v >> shift) & 0xF]++] = v; }
    std::memcpy(a + left, tmp + left, (right - left + 1) * sizeof(int));
    sum = 0;
    for (int i = 0; i < 16; ++i) {
        int nl = left + sum, nr = nl + count[i] - 1;
        if (count[i] > 1) msd_radix_scalar(a, tmp, nl, nr, shift - 4);
        sum += count[i];
    }
}

int main() {
    std::vector<int> sizes = {10000, 50000, 100000, 500000, 1000000, 4000000};
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1000000000, 1000000000);

    std::printf("%-8s | %-12s | %-12s | %-12s | %-12s\n", 
                "N", "Scalar(ms)", "qi-sort(ms)", "quill(ms)", "Speed(qi)");
    std::printf("----------------------------------------------------------------------\n");

    for (int n : sizes) {
        std::vector<int> orig(n + 64), a(n + 64), tmp(n + 64);
        for (int i = 0; i < n; ++i) orig[i] = dist(rng);

        double min_scalar = 1e9;
        for (int r = 0; r < 5; ++r) {
            std::copy(orig.begin(), orig.end(), a.begin());
            auto t0 = Clock::now();
            msd_radix_scalar(a.data(), tmp.data(), 0, n - 1, 28);
            auto t1 = Clock::now();
            min_scalar = std::min(min_scalar, duration<double, std::milli>(t1 - t0).count());
        }

        double min_qi = 1e9;
#ifdef HAVE_QI
        for (int r = 0; r < 5; ++r) {
            std::copy(orig.begin(), orig.end(), a.begin());
            auto t0 = Clock::now();
            qi::sort(a.data(), n);
            auto t1 = Clock::now();
            min_qi = std::min(min_qi, duration<double, std::milli>(t1 - t0).count());
        }
#endif

        double min_quill = 1e9;
#ifdef HAVE_QUILL
        for (int r = 0; r < 5; ++r) {
            std::copy(orig.begin(), orig.end(), a.begin());
            auto t0 = Clock::now();
            quillcore::serial_sort(a.data(), n);
            auto t1 = Clock::now();
            min_quill = std::min(min_quill, duration<double, std::milli>(t1 - t0).count());
        }
#endif

        if (min_qi > 1e8) min_qi = -1;
        if (min_quill > 1e8) min_quill = -1;

        std::printf("%-8d | %-12.4f | ", n, min_scalar);
        if (min_qi > 0) std::printf("%-12.4f", min_qi); else std::printf("%-12s", "N/A");
        std::printf(" | ");
        if (min_quill > 0) std::printf("%-12.4f", min_quill); else std::printf("%-12s", "N/A");
        std::printf(" | ");
        if (min_qi > 0) std::printf("%-12.2fx\n", min_scalar / min_qi); else std::printf("%-12s\n", "N/A");
    }
    return 0;
}
