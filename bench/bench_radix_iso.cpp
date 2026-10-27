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
// 1. Скалярный Radix Sort (как в 16WPMSP)
// ==========================================
void radix_sort_scalar(int* a, int* tmp, int n) {
    if (n <= 32) { std::sort(a, a + n); return; }
    int count[256];
    int* src = a; int* dst = tmp;
    for (int shift = 0; shift < 32; shift += 8) {
        std::memset(count, 0, sizeof(count));
        for (int i = 0; i < n; ++i) {
            int v = src[i]; if (shift == 24) v ^= (int)0x80000000;
            count[(unsigned)(v >> shift) & 0xFF]++;
        }
        int sum = 0;
        for (int i = 0; i < 256; ++i) { int c = count[i]; count[i] = sum; sum += c; }
        for (int i = 0; i < n; ++i) {
            int v = src[i]; if (shift == 24) v ^= (int)0x80000000;
            dst[count[(unsigned)(v >> shift) & 0xFF]++] = src[i];
        }
        std::swap(src, dst);
    }
    if (src != a) std::memcpy(a, src, n * 4);
}

// ==========================================
// 2. Векторизованный Radix Sort (AVX-512)
// ==========================================
void radix_sort_avx512(int* a, int* tmp, int n) {
    if (n <= 32) { std::sort(a, a + n); return; }
    int count[256];
    int* src = a; int* dst = tmp;
    
    for (int shift = 0; shift < 32; shift += 8) {
        std::memset(count, 0, sizeof(count));
        
        // --- Фаза 1: Подсчет гистограммы ---
        for (int i = 0; i < n; ++i) {
            int v = src[i]; if (shift == 24) v ^= (int)0x80000000;
            count[(unsigned)(v >> shift) & 0xFF]++;
        }
        
        int sum = 0;
        for (int j = 0; j < 256; ++j) { int c = count[j]; count[j] = sum; sum += c; }
        
        // --- Фаза 2: Векторизованный Scatter (работает корректно) ---
        int i = 0;
        __m512i v_mask = _mm512_set1_epi32(0xFF);
        __m512i v_xor = _mm512_set1_epi32(shift == 24 ? 0x80000000 : 0);
        __m512i v_inc = _mm512_set1_epi32(1);
        
        for (; i + 16 <= n; i += 16) {
            __m512i v = _mm512_loadu_si512((__m512i*)(src + i));
            v = _mm512_xor_si512(v, v_xor);
            __m512i v_bytes = _mm512_and_si512(_mm512_srli_epi32(v, shift), v_mask);
            
            // 1. Собираем текущие индексы из count
            __m512i v_idx = _mm512_i32gather_epi32(v_bytes, count, 4);
            
            // 2. Записываем 16 значений по 16 адресам в dst
            _mm512_i32scatter_epi32(dst, v_idx, v, 4);
            
            // 3. Обновляем count (скалярно из L1 кэша, т.к. scatter может перезаписать дубликаты)
            int idx[16];
            int bytes[16];
            _mm512_storeu_si512((__m512i*)idx, v_idx);
            _mm512_storeu_si512((__m512i*)bytes, v_bytes);
            for (int j = 0; j < 16; ++j) {
                count[bytes[j]] = idx[j] + 1;
            }
        }
        // Хвост
        for (; i < n; ++i) {
            int v = src[i]; if (shift == 24) v ^= (int)0x80000000;
            dst[count[(unsigned)(v >> shift) & 0xFF]++] = src[i];
        }
        std::swap(src, dst);
    }
    if (src != a) std::memcpy(a, src, n * 4);
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

        double min_scalar = 1e9;
        for (int r = 0; r < 5; ++r) {
            std::copy(orig.begin(), orig.end(), a.begin());
            auto t0 = Clock::now();
            radix_sort_scalar(a.data(), tmp.data(), n);
            auto t1 = Clock::now();
            double ms = duration<double, std::milli>(t1 - t0).count();
            if (ms < min_scalar) min_scalar = ms;
        }

        double min_avx = 1e9;
        for (int r = 0; r < 5; ++r) {
            std::copy(orig.begin(), orig.end(), a.begin());
            auto t0 = Clock::now();
            radix_sort_avx512(a.data(), tmp.data(), n);
            auto t1 = Clock::now();
            double ms = duration<double, std::milli>(t1 - t0).count();
            if (ms < min_avx) min_avx = ms;
        }

        double speedup = min_scalar / min_avx;
        std::printf("%-10d | %-15.3f | %-15.3f | %-10.2fx\n", n, min_scalar, min_avx, speedup);
    }
    return 0;
}
