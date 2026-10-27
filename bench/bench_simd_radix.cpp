#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstring>
#include "radixSort.hpp"

using Clock = std::chrono::high_resolution_clock;
using std::chrono::duration;

// Наш скалярный 4-битный MSD Radix Sort
void msd_radix_scalar(int* a, int* tmp, int left, int right, int shift) {
    if (right - left < 64) {
        std::sort(a + left, a + right + 1);
        return;
    }
    if (shift < 0) return;

    int count[16] = {0};
    for (int i = left; i <= right; ++i) count[(a[i] >> shift) & 0xF]++;

    int offset[16];
    int sum = 0;
    for (int i = 0; i < 16; ++i) { offset[i] = left + sum; sum += count[i]; }

    for (int i = left; i <= right; ++i) {
        int val = a[i];
        tmp[offset[(val >> shift) & 0xF]++] = val;
    }
    std::memcpy(a + left, tmp + left, (right - left + 1) * sizeof(int));

    sum = 0;
    for (int i = 0; i < 16; ++i) {
        int new_left = left + sum;
        int new_right = new_left + count[i] - 1;
        if (count[i] > 1) msd_radix_scalar(a, tmp, new_left, new_right, shift - 4);
        sum += count[i];
    }
}

int main() {
    // Ищем точку окупаемости с шагом 5000
    std::vector<int> sizes = {10000, 15000, 20000, 25000, 30000, 
                              35000, 40000, 45000, 50000, 60000, 80000, 100000};
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1000000000, 1000000000);

    std::printf("%-8s | %-12s | %-12s | %-10s\n", "N", "Scalar (ms)", "SIMD (ms)", "Speedup");
    std::printf("-----------------------------------------------------\n");

    for (int n : sizes) {
        std::vector<int> orig(n + 64), a(n + 64), tmp(n + 64);
        for (int i = 0; i < n; ++i) orig[i] = dist(rng);

        // --- Скалярный ---
        // Увеличиваем warmup до 10, measure до 15
        for (int r = 0; r < 10; ++r) {
            std::copy(orig.begin(), orig.end(), a.begin());
            msd_radix_scalar(a.data(), tmp.data(), 0, n - 1, 28);
        }
        double min_scalar = 1e9;
        for (int r = 0; r < 15; ++r) {
            std::copy(orig.begin(), orig.end(), a.begin());
            auto t0 = Clock::now();
            msd_radix_scalar(a.data(), tmp.data(), 0, n - 1, 28);
            auto t1 = Clock::now();
            double ms = duration<double, std::milli>(t1 - t0).count();
            if (ms < min_scalar) min_scalar = ms;
        }

        // --- SIMD библиотека ---
        for (int r = 0; r < 10; ++r) {
            std::copy(orig.begin(), orig.end(), a.begin());
            simd_sort::radix_sort::sort(n, a.data());
        }
        double min_simd = 1e9;
        for (int r = 0; r < 15; ++r) {
            std::copy(orig.begin(), orig.end(), a.begin());
            auto t0 = Clock::now();
            simd_sort::radix_sort::sort(n, a.data());
            auto t1 = Clock::now();
            double ms = duration<double, std::milli>(t1 - t0).count();
            if (ms < min_simd) min_simd = ms;
        }

        std::printf("%-8d | %-12.4f | %-12.4f | %-10.2fx\n", 
                    n, min_scalar, min_simd, min_scalar / min_simd);
    }
    return 0;
}
