#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <immintrin.h>

using Clock = std::chrono::high_resolution_clock;
using std::chrono::duration;

// ==========================================
// 1. Скалярный MSD Radix Sort (база для сравнения)
// ==========================================
void msd_radix_scalar(int* a, int* tmp, int left, int right, int shift) {
    if (right - left < 64) {
        std::sort(a + left, a + right + 1);
        return;
    }
    if (shift < 0) return;

    int count[16] = {0};
    for (int i = left; i <= right; ++i) {
        count[(a[i] >> shift) & 0xF]++;
    }

    int offset[16];
    int sum = 0;
    for (int i = 0; i < 16; ++i) {
        offset[i] = left + sum;
        sum += count[i];
    }

    for (int i = left; i <= right; ++i) {
        int val = a[i];
        int bucket = (val >> shift) & 0xF;
        tmp[offset[bucket]++] = val;
    }

    std::memcpy(a + left, tmp + left, (right - left + 1) * sizeof(int));

    sum = 0;
    for (int i = 0; i < 16; ++i) {
        int new_left = left + sum;
        int new_right = new_left + count[i] - 1;
        if (count[i] > 1) {
            msd_radix_scalar(a, tmp, new_left, new_right, shift - 4);
        }
        sum += count[i];
    }
}

// ==========================================
// 2. Векторизованный MSD Radix Sort (AVX-512 compress-store)
// ==========================================
void msd_radix_avx512(int* a, int* tmp, int left, int right, int shift) {
    if (right - left < 128) { // Порог, ниже которого скалярный быстрее
        msd_radix_scalar(a, tmp, left, right, shift);
        return;
    }
    if (shift < 0) return;

    // Фаза 1: Векторизованный подсчет гистограммы
    int count[16] = {0};
    int i = left;
    __m512i v_mask = _mm512_set1_epi32(0xF);
    __m512i v_shift = _mm512_set1_epi32(shift);
    
    for (; i + 16 <= right + 1; i += 16) {
        __m512i v = _mm512_loadu_si512((__m512i*)(a + i));
        __m512i v_digits = _mm512_and_si512(_mm512_srlv_epi32(v, v_shift), v_mask);
        
        int digits[16];
        _mm512_storeu_si512((__m512i*)digits, v_digits);
        for (int j = 0; j < 16; ++j) {
            count[digits[j]]++;
        }
    }
    for (; i <= right; ++i) {
        count[(a[i] >> shift) & 0xF]++;
    }

    // Префиксные суммы (смещения для каждой корзины)
    int offset[16];
    int sum = 0;
    for (int j = 0; j < 16; ++j) {
        offset[j] = left + sum;
        sum += count[j];
    }

    // Фаза 2: Векторизованная запись через compress-store
    // Мы проходим по данным блоками и для каждой корзины вызываем compress-store
    // Это 16 проходов по данным, но каждый проход — это просто маска и запись.
    for (int bucket = 0; bucket < 16; ++bucket) {
        int write_pos = offset[bucket];
        for (i = left; i <= right; i += 16) {
            int n_elems = std::min(16, right - i + 1);
            __m512i v = _mm512_loadu_si512((__m512i*)(a + i));
            __m512i v_digits = _mm512_and_si512(_mm512_srlv_epi32(v, v_shift), v_mask);
            
            // Сравниваем с номером корзины и получаем маску
            __mmask16 mask = _mm512_cmpeq_epi32_mask(v_digits, _mm512_set1_epi32(bucket));
            if (mask) {
                _mm512_mask_compressstoreu_epi32(tmp + write_pos, mask, v);
                write_pos += __builtin_popcount(mask);
            }
        }
    }

    std::memcpy(a + left, tmp + left, (right - left + 1) * sizeof(int));

    // Рекурсивный спуск
    sum = 0;
    for (int j = 0; j < 16; ++j) {
        int new_left = left + sum;
        int new_right = new_left + count[j] - 1;
        if (count[j] > 1) {
            msd_radix_avx512(a, tmp, new_left, new_right, shift - 4);
        }
        sum += count[j];
    }
}

// ==========================================
// Бенчмарк
// ==========================================
int main() {
    std::vector<int> sizes = {1000, 5000, 10000, 20000, 50000, 100000, 
                              200000, 500000, 1000000, 2000000, 4000000};
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1000000000, 1000000000);

    std::printf("%-10s | %-15s | %-15s | %-10s\n", "N", "Scalar (ms)", "AVX-512 (ms)", "Speedup");
    std::printf("-----------------------------------------------------------------\n");

    for (int n : sizes) {
        std::vector<int> orig(n), a(n), tmp(n);
        for (int i = 0; i < n; ++i) orig[i] = dist(rng);

        // Warmup & Measure Scalar
        double min_scalar = 1e9;
        for (int r = 0; r < 5; ++r) {
            std::copy(orig.begin(), orig.end(), a.begin());
            auto t0 = Clock::now();
            msd_radix_scalar(a.data(), tmp.data(), 0, n - 1, 28);
            auto t1 = Clock::now();
            double ms = duration<double, std::milli>(t1 - t0).count();
            if (ms < min_scalar) min_scalar = ms;
        }

        // Warmup & Measure AVX-512
        double min_avx = 1e9;
        for (int r = 0; r < 5; ++r) {
            std::copy(orig.begin(), orig.end(), a.begin());
            auto t0 = Clock::now();
            msd_radix_avx512(a.data(), tmp.data(), 0, n - 1, 28);
            auto t1 = Clock::now();
            double ms = duration<double, std::milli>(t1 - t0).count();
            if (ms < min_avx) min_avx = ms;
        }

        double speedup = min_scalar / min_avx;
        std::printf("%-10d | %-15.3f | %-15.3f | %-10.2fx\n", n, min_scalar, min_avx, speedup);
    }
    return 0;
}
