#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <immintrin.h>

using Clock = std::chrono::high_resolution_clock;
using std::chrono::duration;

// ============================================================================
// 1. Скалярный 4-битный MSD Radix Sort
// ============================================================================
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

// ============================================================================
// 2. Побитовый MSD Radix Sort на AVX-512 (1 бит за проход)
// ============================================================================
void msd_radix_bitwise_avx512(int* a, int* tmp, int left, int right, int bit) {
    if (right - left < 128) {
        std::sort(a + left, a + right + 1);
        return;
    }
    if (bit < 0) return;

    int mask_bit = 1 << bit;
    int n = right - left + 1;
    int i = left;
    int count_zero = 0;

    // Пишем нули
    for (; i + 16 <= right + 1; i += 16) {
        __m512i v = _mm512_loadu_si512((__m512i*)(a + i));
        __mmask16 mask = _mm512_test_epi32_mask(v, _mm512_set1_epi32(mask_bit));
        __mmask16 zero_mask = ~mask;
        int zeros = __builtin_popcount(zero_mask);
        if (zeros > 0) {
            __m512i compressed = _mm512_maskz_compress_epi32(zero_mask, v);
            _mm512_storeu_si512((__m512i*)(tmp + left + count_zero), compressed);
            count_zero += zeros;
        }
    }
    for (; i <= right; ++i) if ((a[i] & mask_bit) == 0) tmp[left + count_zero++] = a[i];

    // Пишем единицы
    int count_one = count_zero;
    i = left;
    for (; i + 16 <= right + 1; i += 16) {
        __m512i v = _mm512_loadu_si512((__m512i*)(a + i));
        __mmask16 mask = _mm512_test_epi32_mask(v, _mm512_set1_epi32(mask_bit));
        int ones = __builtin_popcount(mask);
        if (ones > 0) {
            __m512i compressed = _mm512_maskz_compress_epi32(mask, v);
            _mm512_storeu_si512((__m512i*)(tmp + left + count_one), compressed);
            count_one += ones;
        }
    }
    for (; i <= right; ++i) if (a[i] & mask_bit) tmp[left + count_one++] = a[i];

    std::memcpy(a + left, tmp + left, n * sizeof(int));

    if (count_zero > 1) msd_radix_bitwise_avx512(a, tmp, left, left + count_zero - 1, bit - 1);
    if (n - count_zero > 1) msd_radix_bitwise_avx512(a, tmp, left + count_zero, right, bit - 1);
}

// ============================================================================
// 3. 4-битный MSD Radix Sort на AVX-512 (compress + store)
// ============================================================================
void msd_radix_4bit_avx512(int* a, int* tmp, int left, int right, int shift) {
    if (right - left < 128) {
        std::sort(a + left, a + right + 1);
        return;
    }
    if (shift < 0) return;

    int count[16] = {0};
    int i = left;
    __m512i v_mask = _mm512_set1_epi32(0xF);
    __m512i v_shift = _mm512_set1_epi32(shift);

    for (; i + 16 <= right + 1; i += 16) {
        __m512i v = _mm512_loadu_si512((__m512i*)(a + i));
        __m512i v_digits = _mm512_and_si512(_mm512_srlv_epi32(v, v_shift), v_mask);
        int digits[16];
        _mm512_storeu_si512((__m512i*)digits, v_digits);
        for (int j = 0; j < 16; ++j) count[digits[j]]++;
    }
    for (; i <= right; ++i) count[(a[i] >> shift) & 0xF]++;

    int offset[16];
    int sum = 0;
    for (int j = 0; j < 16; ++j) { offset[j] = left + sum; sum += count[j]; }

    for (int bucket = 0; bucket < 16; ++bucket) {
        int write_pos = offset[bucket];
        for (i = left; i <= right; i += 16) {
            __m512i v = _mm512_loadu_si512((__m512i*)(a + i));
            __m512i v_digits = _mm512_and_si512(_mm512_srlv_epi32(v, v_shift), v_mask);
            __mmask16 mask = _mm512_cmpeq_epi32_mask(v_digits, _mm512_set1_epi32(bucket));
            if (mask) {
                __m512i compressed = _mm512_maskz_compress_epi32(mask, v);
                _mm512_storeu_si512((__m512i*)(tmp + write_pos), compressed);
                write_pos += __builtin_popcount(mask);
            }
        }
    }

    std::memcpy(a + left, tmp + left, (right - left + 1) * sizeof(int));

    sum = 0;
    for (int j = 0; j < 16; ++j) {
        int new_left = left + sum;
        int new_right = new_left + count[j] - 1;
        if (count[j] > 1) msd_radix_4bit_avx512(a, tmp, new_left, new_right, shift - 4);
        sum += count[j];
    }
}

// ============================================================================
// Бенчмарк
// ============================================================================
int main() {
    // Добавлены маленькие размеры для поиска точки окупаемости
    std::vector<int> sizes = {100, 500, 1000, 5000, 10000, 20000, 50000, 
                              100000, 500000, 1000000, 2000000, 4000000};
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1000000000, 1000000000);

    std::printf("%-10s | %-12s | %-12s | %-12s | %-10s | %-10s\n", 
                "N", "Scalar(ms)", "Bitwise(ms)", "4-bit(ms)", "Speed bit", "Speed 4b");
    std::printf("------------------------------------------------------------------------------------\n");

    for (int n : sizes) {
        // ДОБАВЛЯЕМ ПАДДИНГ (+64) чтобы vectorized store не ломал память
        std::vector<int> orig(n + 64), a(n + 64), tmp(n + 64);
        for (int i = 0; i < n; ++i) orig[i] = dist(rng);

        double min_scalar = 1e9;
        for (int r = 0; r < 5; ++r) {
            std::copy(orig.begin(), orig.end(), a.begin());
            auto t0 = Clock::now();
            msd_radix_scalar(a.data(), tmp.data(), 0, n - 1, 28);
            auto t1 = Clock::now();
            double ms = duration<double, std::milli>(t1 - t0).count();
            if (ms < min_scalar) min_scalar = ms;
        }

        double min_bitwise = 1e9;
        for (int r = 0; r < 5; ++r) {
            std::copy(orig.begin(), orig.end(), a.begin());
            auto t0 = Clock::now();
            msd_radix_bitwise_avx512(a.data(), tmp.data(), 0, n - 1, 31);
            auto t1 = Clock::now();
            double ms = duration<double, std::milli>(t1 - t0).count();
            if (ms < min_bitwise) min_bitwise = ms;
        }

        double min_4bit = 1e9;
        for (int r = 0; r < 5; ++r) {
            std::copy(orig.begin(), orig.end(), a.begin());
            auto t0 = Clock::now();
            msd_radix_4bit_avx512(a.data(), tmp.data(), 0, n - 1, 28);
            auto t1 = Clock::now();
            double ms = duration<double, std::milli>(t1 - t0).count();
            if (ms < min_4bit) min_4bit = ms;
        }

        std::printf("%-10d | %-12.3f | %-12.3f | %-12.3f | %-10.2fx | %-10.2fx\n", 
                    n, min_scalar, min_bitwise, min_4bit, min_scalar/min_bitwise, min_scalar/min_4bit);
    }
    return 0;
}
