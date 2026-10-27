#pragma once
#include "16wpmsp/thread_pool.hpp"
#include "16wpmsp/third_party/qi_radix.hpp"
// 16WPMSP — 16-Way Partitioned Merge Sort Parallel (header-only)

// ============================================================================
// 16WPMSP v6 ADAPTIVE FULL
// 16-Way Partitioned Merge Sort Parallel
// ----------------------------------------------------------------------------
// Features:
//   • Core auto-detection: sched_getaffinity → sysconf → hardware_concurrency → 1
//   • Defense-in-depth: parallel_16wpmsp_sort re-clamps its cap against
//     detect_num_cores(), so a caller passing num_threads=32 on an 8-core
//     box will NOT spawn 32 threads.
//   • Two-dimensional adaptivity:
//       1) TH    = clamp((N/K_eff)/8, 2000, 50000)
//       2) K_eff = pick_K_eff(N, K_max)   [step function]
//   • NUM_THREADS is a compile-time CAP (default 0 = fully auto)
//   • Covers any K_max >= 1 (1, 2, 4, 6, 8, 10, 12, 16, 24, 32, ...)
//
// K=1: pure sort_local_16wpmsp
// K=2: 2 chunks + single par_merge (a → scratch → a)
// K>=4: slab-splitters (nslabs = min(16, K_max)) + tree par_merge
// ============================================================================

#define NOMINMAX
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <climits>
#include <cstdio>
#include <cmath>
#include <thread>
#include <atomic>
#include <immintrin.h>

#if defined(__linux__)
#  include <sched.h>
#  include <unistd.h>
#endif

// ----------------------------------------------------------------------------
// NUM_THREADS is an UPPER CAP. 0 = auto-detect from hardware.
// ----------------------------------------------------------------------------
#ifndef NUM_THREADS
#define NUM_THREADS 0
#endif
#ifndef TH_FLOOR
#define TH_FLOOR 2000
#endif
#ifndef TH_CAP
#define TH_CAP   50000
#endif
#ifndef TH_DIVISOR
#define TH_DIVISOR 8
#endif

const int MEASURE_RUNS = 7;
const int WARMUP_RUNS  = 2;
const int VEC_SIZE = 16;
const int FIXED_BUF_SIZE = 256;
const int SAMPLES_PER_RUN = 64;
const int RUNS_COUNT = 16;

using Clock = std::chrono::high_resolution_clock;
using std::chrono::duration;

// ============================================================================
// CORE DETECTION
// ============================================================================
inline int detect_num_cores() {
#if defined(__linux__)
    // 1. sched_getaffinity — honours taskset and cgroup CPU limits.
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int n = CPU_COUNT(&set);
        if (n > 0) return n;
    }
    // 2. sysconf — online CPUs system-wide.
    {
        long sc = sysconf(_SC_NPROCESSORS_ONLN);
        if (sc > 0) return (int)sc;
    }
#endif
    // 3. C++11 fallback.
    unsigned hc = std::thread::hardware_concurrency();
    if (hc > 0) return (int)hc;
    return 1;
}

inline int resolve_K_max() {
    int detected = detect_num_cores();
#if defined(NUM_THREADS) && NUM_THREADS > 0
    return std::min(detected, (int)NUM_THREADS);
#else
    return detected;
#endif
}

// ============================================================================
// SIMD primitives
// ============================================================================
inline __m512i reverse16(__m512i v) {
    return _mm512_permutexvar_epi32(_mm512_setr_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0), v);
}
inline __m512i bitonic_merge_16(__m512i v) {
    __m512i vp = _mm512_permutexvar_epi32(_mm512_setr_epi32(8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7), v);
    v = _mm512_mask_blend_epi32(0xFF00, _mm512_min_epi32(v, vp), _mm512_max_epi32(v, vp));
    vp = _mm512_permutexvar_epi32(_mm512_setr_epi32(4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11), v);
    v = _mm512_mask_blend_epi32(0xF0F0, _mm512_min_epi32(v, vp), _mm512_max_epi32(v, vp));
    vp = _mm512_permutexvar_epi32(_mm512_setr_epi32(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13), v);
    v = _mm512_mask_blend_epi32(0xCCCC, _mm512_min_epi32(v, vp), _mm512_max_epi32(v, vp));
    vp = _mm512_permutexvar_epi32(_mm512_setr_epi32(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14), v);
    v = _mm512_mask_blend_epi32(0xAAAA, _mm512_min_epi32(v, vp), _mm512_max_epi32(v, vp));
    return v;
}
inline __m512i sort16(__m512i v) {
    __m512i vp = _mm512_permutexvar_epi32(_mm512_setr_epi32(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14), v);
    v = _mm512_mask_blend_epi32(0x6666, _mm512_min_epi32(v, vp), _mm512_max_epi32(v, vp));
    vp = _mm512_permutexvar_epi32(_mm512_setr_epi32(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13), v);
    v = _mm512_mask_blend_epi32(0x3C3C, _mm512_min_epi32(v, vp), _mm512_max_epi32(v, vp));
    vp = _mm512_permutexvar_epi32(_mm512_setr_epi32(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14), v);
    v = _mm512_mask_blend_epi32(0x5A5A, _mm512_min_epi32(v, vp), _mm512_max_epi32(v, vp));
    vp = _mm512_permutexvar_epi32(_mm512_setr_epi32(4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11), v);
    v = _mm512_mask_blend_epi32(0x0FF0, _mm512_min_epi32(v, vp), _mm512_max_epi32(v, vp));
    vp = _mm512_permutexvar_epi32(_mm512_setr_epi32(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13), v);
    v = _mm512_mask_blend_epi32(0x33CC, _mm512_min_epi32(v, vp), _mm512_max_epi32(v, vp));
    vp = _mm512_permutexvar_epi32(_mm512_setr_epi32(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14), v);
    v = _mm512_mask_blend_epi32(0x55AA, _mm512_min_epi32(v, vp), _mm512_max_epi32(v, vp));
    return bitonic_merge_16(v);
}
inline void bitonic_sort_64_from_bitonic(__m512i& x0, __m512i& x1, __m512i& x2, __m512i& x3) {
    __m512i a0 = _mm512_min_epi32(x0, x2), a2 = _mm512_max_epi32(x0, x2);
    __m512i a1 = _mm512_min_epi32(x1, x3), a3 = _mm512_max_epi32(x1, x3);
    __m512i b0 = _mm512_min_epi32(a0, a1), b1 = _mm512_max_epi32(a0, a1);
    __m512i b2 = _mm512_min_epi32(a2, a3), b3 = _mm512_max_epi32(a2, a3);
    x0 = bitonic_merge_16(b0); x1 = bitonic_merge_16(b1);
    x2 = bitonic_merge_16(b2); x3 = bitonic_merge_16(b3);
}
inline void merge_128_128_regs(__m512i* a, __m512i* b, __m512i* out) {
    __m512i br[8];
    for (int i = 0; i < 8; ++i) br[i] = reverse16(b[7-i]);
    __m512i L[8], H[8];
    for (int i = 0; i < 8; ++i) { L[i] = _mm512_min_epi32(a[i], br[i]); H[i] = _mm512_max_epi32(a[i], br[i]); }
    __m512i Ls[8], Hs[8];
    for (int i = 0; i < 4; ++i) {
        Ls[i] = _mm512_min_epi32(L[i], L[i+4]); Ls[i+4] = _mm512_max_epi32(L[i], L[i+4]);
        Hs[i] = _mm512_min_epi32(H[i], H[i+4]); Hs[i+4] = _mm512_max_epi32(H[i], H[i+4]);
    }
    bitonic_sort_64_from_bitonic(Ls[0],Ls[1],Ls[2],Ls[3]);
    bitonic_sort_64_from_bitonic(Ls[4],Ls[5],Ls[6],Ls[7]);
    bitonic_sort_64_from_bitonic(Hs[0],Hs[1],Hs[2],Hs[3]);
    bitonic_sort_64_from_bitonic(Hs[4],Hs[5],Hs[6],Hs[7]);
    for (int i = 0; i < 8; ++i) { out[i] = Ls[i]; out[i+8] = Hs[i]; }
}
inline void sort_16x16_to_256(const int* src, __m512i* out) {
    __m512i r[16];
    for (int i = 0; i < 16; ++i) { r[i] = _mm512_loadu_si512((__m512i*)(src+i*16)); r[i] = sort16(r[i]); }
    for (int i = 0; i < 16; i += 2) {
        __m512i br = reverse16(r[i+1]);
        __m512i L = _mm512_min_epi32(r[i], br), H = _mm512_max_epi32(r[i], br);
        r[i] = bitonic_merge_16(L); r[i+1] = bitonic_merge_16(H);
    }
    for (int i = 0; i < 16; i += 4) {
        __m512i a0=r[i],a1=r[i+1],b0=r[i+2],b1=r[i+3];
        __m512i b1r=reverse16(b1),b0r=reverse16(b0);
        __m512i L0=_mm512_min_epi32(a0,b1r),H0=_mm512_max_epi32(a0,b1r);
        __m512i L1=_mm512_min_epi32(a1,b0r),H1=_mm512_max_epi32(a1,b0r);
        __m512i LL0=_mm512_min_epi32(L0,L1),LL1=_mm512_max_epi32(L0,L1);
        __m512i HH0=_mm512_min_epi32(H0,H1),HH1=_mm512_max_epi32(H0,H1);
        r[i]=bitonic_merge_16(LL0); r[i+1]=bitonic_merge_16(LL1);
        r[i+2]=bitonic_merge_16(HH0); r[i+3]=bitonic_merge_16(HH1);
    }
    for (int i = 0; i < 16; i += 8) {
        __m512i a0=r[i],a1=r[i+1],a2=r[i+2],a3=r[i+3];
        __m512i b0=r[i+4],b1=r[i+5],b2=r[i+6],b3=r[i+7];
        __m512i b3r=reverse16(b3),b2r=reverse16(b2),b1r=reverse16(b1),b0r=reverse16(b0);
        __m512i L0=_mm512_min_epi32(a0,b3r),H0=_mm512_max_epi32(a0,b3r);
        __m512i L1=_mm512_min_epi32(a1,b2r),H1=_mm512_max_epi32(a1,b2r);
        __m512i L2=_mm512_min_epi32(a2,b1r),H2=_mm512_max_epi32(a2,b1r);
        __m512i L3=_mm512_min_epi32(a3,b0r),H3=_mm512_max_epi32(a3,b0r);
        bitonic_sort_64_from_bitonic(L0,L1,L2,L3);
        bitonic_sort_64_from_bitonic(H0,H1,H2,H3);
        r[i]=L0;r[i+1]=L1;r[i+2]=L2;r[i+3]=L3;r[i+4]=H0;r[i+5]=H1;r[i+6]=H2;r[i+7]=H3;
    }
    __m512i A[8],B[8];
    for (int i = 0; i < 8; ++i) { A[i]=r[i]; B[i]=r[i+8]; }
    merge_128_128_regs(A,B,out);
}
void merge_scalar_small(int* dst, const int* A, int la, const int* B, int lb) {
    if (la>0 && lb>0 && A[la-1]<=B[0]) { std::memcpy(dst,A,la*4); std::memcpy(dst+la,B,lb*4); return; }
    if (la>0 && lb>0 && B[lb-1]<A[0]) { std::memcpy(dst,B,lb*4); std::memcpy(dst+lb,A,la*4); return; }
    int i=0,j=0,k=0;
    while (i<la && j<lb) { int t=(A[i]<=B[j]); dst[k++]=t?A[i]:B[j]; if(t)i++;else j++; }
    while (i<la) dst[k++]=A[i++];
    while (j<lb) dst[k++]=B[j++];
}
void custom_merge_tree(int* flat, int* temp, int n) {
    if (n >= 64 && n <= 256) {
        for (int j = n; j < 256; ++j) flat[j] = INT_MAX;
        __m512i out[16]; sort_16x16_to_256(flat, out);
        for (int i = 0; i < 16; ++i) _mm512_storeu_si512((__m512i*)(flat+i*16), out[i]);
        return;
    }
    if (n < 64) {
        int* src = flat; int* dst = temp;
        for (int i = 0; i < n; i += 16) {
            if (i+16<=n) { __m512i v = _mm512_loadu_si512((__m512i*)(src+i)); _mm512_storeu_si512((__m512i*)(src+i), sort16(v)); }
            else std::sort(src+i, src+n);
        }
        int rs = 16;
        while (rs < n) {
            for (int i = 0; i < n; i += rs*2) {
                int lA=std::min(rs,n-i), lB=std::min(rs,n-i-rs);
                if (lA==rs && lB==rs && rs==16) {
                    __m512i a=_mm512_loadu_si512((__m512i*)(src+i)),b=_mm512_loadu_si512((__m512i*)(src+i+16));
                    __m512i br=reverse16(b);
                    _mm512_storeu_si512((__m512i*)(dst+i), bitonic_merge_16(_mm512_min_epi32(a,br)));
                    _mm512_storeu_si512((__m512i*)(dst+i+16), bitonic_merge_16(_mm512_max_epi32(a,br)));
                } else if (lB<=0) std::memcpy(dst+i,src+i,lA*4);
                else merge_scalar_small(dst+i,src+i,lA,src+i+rs,lB);
            }
            std::swap(src,dst); rs*=2;
        }
        if (src != flat) std::memcpy(flat,src,n*4);
    }
}
void radix_sort_i32(int* a, int* tmp, int n) {
    if (n <= 32) { std::sort(a, a+n); return; }
    int count[256]; int* src = a; int* dst = tmp;
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
    if (src != a) std::memcpy(a, src, n*4);
}
void merge16_threshold_fast(int* out, const int* const* runs, int* lengths, int total, int buf_size) {
    int* ptrs[16]; int rem[16];
    int* flat = (int*)_mm_malloc((buf_size+32)*4, 64);
    int* temp = (int*)_mm_malloc((buf_size+32)*4, 64);
    int ih[16]; int active = 0;
    for (int r = 0; r < 16; ++r) {
        ptrs[r] = const_cast<int*>(runs[r]); rem[r] = lengths[r];
        if (rem[r] > 0) { ih[r] = ptrs[r][0]; active++; } else ih[r] = INT_MAX;
    }
    __m512i vheads = _mm512_loadu_si512((__m512i*)ih);
    int oi = 0, valid = 0, fi = 0;
    while (oi < total) {
        while (fi + VEC_SIZE <= buf_size && active > 0) {
            int mv = _mm512_reduce_min_epi32(vheads);
            if (mv == INT_MAX) break;
            __mmask16 m = _mm512_cmpeq_epi32_mask(vheads, _mm512_set1_epi32(mv));
            int mi = (int)_tzcnt_u32((unsigned)m);
            int take = (rem[mi] < VEC_SIZE) ? rem[mi] : VEC_SIZE;
            if (take == VEC_SIZE) _mm512_storeu_si512((__m512i*)(flat+fi), _mm512_loadu_si512((__m512i*)(ptrs[mi])));
            else { std::memcpy(flat+fi, ptrs[mi], take*4); for (int j=take;j<VEC_SIZE;++j) flat[fi+j]=INT_MAX; }
            ptrs[mi]+=take; rem[mi]-=take; valid+=take; fi+=VEC_SIZE;
            int nh;
            if (rem[mi] > 0) nh = ptrs[mi][0]; else { nh = INT_MAX; active--; }
            vheads = _mm512_mask_blend_epi32((__mmask16)(1u<<mi), vheads, _mm512_set1_epi32(nh));
        }
        if (fi == 0) break;
        custom_merge_tree(flat, temp, fi);
        int thr = _mm512_reduce_min_epi32(vheads);
        int k = 0; __m512i b = _mm512_set1_epi32(thr);
        while (k + 16 <= valid) {
            __m512i v = _mm512_loadu_si512((__m512i*)(flat+k));
            __mmask16 le = _mm512_cmple_epi32_mask(v, b);
            if (le == 0xFFFF) k += 16;
            else { k += (int)_tzcnt_u32((unsigned)(~le & 0xFFFF)); break; }
        }
        while (k < valid && flat[k] <= thr) k++;
        if (k > valid) k = valid;
        if (k == 0 && valid > 0) k = 1;
        int ii = 0;
        for (; ii+16<=k; ii+=16) _mm512_storeu_si512((__m512i*)(out+oi+ii), _mm512_loadu_si512((__m512i*)(flat+ii)));
        for (; ii < k; ++ii) out[oi+ii] = flat[ii];
        int rm = valid - k;
        if (rm > 0) std::memmove(flat, flat+k, rm*4);
        int pad = (VEC_SIZE - (rm % VEC_SIZE)) % VEC_SIZE;
        for (int j=0;j<pad;++j) flat[rm+j]=INT_MAX;
        valid = rm; fi = rm + pad; oi += k;
    }
    _mm_free(flat); _mm_free(temp);
}

void radix_sort_i32_qi(int* a, int* tmp, int n) {
    uint32_t* enc = (uint32_t*)tmp;
    for (int i = 0; i < n; ++i) enc[i] = (uint32_t)(a[i]) ^ 0x80000000;
    qi::radix_11(enc, n);
    for (int i = 0; i < n; ++i) a[i] = (int)(enc[i] ^ 0x80000000);
}

void sort_local_16wpmsp(int* a, int* buf, int n, int th) {
    if (n <= 32) { std::sort(a, a+n); return; }
    int* src = a; int* dst = buf;
    for (int i = 0; i < n; i += th) { int e = std::min(i+th,n); radix_sort_i32_qi(src+i, dst+i, e-i); }
    int rs = th;
    while (rs < n) {
        int step = rs * RUNS_COUNT;
        for (int i = 0; i < n; i += step) {
            const int* runs[RUNS_COUNT]; int lens[RUNS_COUNT]; int vr = 0;
            for (int r = 0; r < RUNS_COUNT; ++r) {
                int off = i + r*rs;
                if (off >= n) { runs[r]=nullptr; lens[r]=0; }
                else { runs[r]=src+off; lens[r]=std::min(rs,n-off); vr++; }
            }
            if (vr == 1) std::memcpy(dst+i, src+i, std::min(step,n-i)*4);
            else {
                int total = 0; for (int r=0;r<RUNS_COUNT;++r) total += lens[r];
                for (int r=vr;r<RUNS_COUNT;++r) runs[r]=nullptr;
                merge16_threshold_fast(dst+i, runs, lens, total, FIXED_BUF_SIZE);
            }
        }
        std::swap(src, dst); rs *= RUNS_COUNT;
    }
    if (src != a) std::memcpy(a, src, n*4);
}
void mpi(const int* A, int la, const int* B, int lb, int d, int& i, int& j) {
    int lo = std::max(0, d-lb), hi = std::min(d, la);
    while (lo < hi) {
        int mid = (lo+hi)>>1;
        int am = (mid<la)?A[mid]:INT_MAX;
        int bm = (d-mid-1>=0 && d-mid-1<lb)?B[d-mid-1]:INT_MIN;
        if (am > bm) hi = mid; else lo = mid+1;
    }
    i = lo; j = d - lo;
}
void par_merge(const int* A, int la, const int* B, int lb, int* out, int nt) {
    int total = la + lb;
    if (total == 0) return;
    if (nt <= 1 || total < 50000) { std::merge(A, A+la, B, B+lb, out); return; }
    nt = std::min(nt, total/25000+1);
    std::vector<int> d(nt+1);
    for (int t=0;t<=nt;++t) d[t] = (int)((long long)t*total/nt);
    std::vector<std::pair<int,int>> in(nt+1);
    for (int t=0;t<=nt;++t) { int i,j; mpi(A,la,B,lb,d[t],i,j); in[t]={i,j}; }
    std::vector<std::thread> w;
    for (int t=0;t<nt;++t) {
        int a0=in[t].first,b0=in[t].second,a1=in[t+1].first,b1=in[t+1].second;
        if (a0>=a1 && b0>=b1) continue;
        w.emplace_back([A,B,out,a0,b0,a1,b1,d,t]() { std::merge(A+a0,A+a1,B+b0,B+b1,out+d[t]); });
    }
    for (auto& x : w) x.join();
}
void parallel_16way_slab(int* in_block, int* out_block, int total, int cs, int nslabs) {
    if (total <= 0) return;
    std::vector<int> samples; samples.reserve(16 * SAMPLES_PER_RUN);
    for (int r = 0; r < 16; ++r) {
        int off = r * cs; int len = std::min(cs, total - off);
        if (len <= 0) continue;
        for (int s = 0; s < SAMPLES_PER_RUN; ++s) {
            int idx = (int)((long long)s * len / SAMPLES_PER_RUN);
            samples.push_back(in_block[off + idx]);
        }
    }
    std::sort(samples.begin(), samples.end());
    std::vector<int> splitters(nslabs - 1);
    for (int i = 0; i < nslabs - 1; ++i) {
        int idx = (int)((long long)(i+1) * samples.size() / nslabs);
        splitters[i] = samples[idx];
    }
    std::vector<std::vector<std::pair<int,int>>> pieces(nslabs, std::vector<std::pair<int,int>>(16));
    for (int r = 0; r < 16; ++r) {
        int off = r * cs; int len = std::min(cs, total - off);
        if (len <= 0) continue;
        const int* run = in_block + off;
        for (int s = 0; s < nslabs; ++s) {
            int lo = 0, hi = len;
            if (s > 0) lo = (int)(std::lower_bound(run, run+len, splitters[s-1]) - run);
            if (s < nslabs-1) hi = (int)(std::lower_bound(run, run+len, splitters[s]) - run);
            pieces[s][r] = {lo, hi};
        }
    }
    std::vector<long long> slab_off(nslabs + 1, 0);
    for (int s = 0; s < nslabs; ++s) {
        long long t = 0;
        for (int r = 0; r < 16; ++r) t += pieces[s][r].second - pieces[s][r].first;
        slab_off[s+1] = slab_off[s] + t;
    }
    if (slab_off[nslabs] != total) {
        std::sort(in_block, in_block + total);
        std::memcpy(out_block, in_block, (size_t)total * sizeof(int));
        return;
    }
#ifdef USE_THREAD_POOL
    wpmsp::ThreadPool::instance().parallel_for(0, nslabs, [&](int s) {
        int sz = (int)(slab_off[s+1] - slab_off[s]);
        if (sz <= 0) return;
        int* out_slab = out_block + slab_off[s];
        const int* runs[16]; int lens[16]; int cnt = 0;
        for (int r = 0; r < 16; ++r) {
            int lo = pieces[s][r].first, hi = pieces[s][r].second;
            if (hi > lo) { runs[cnt] = in_block + r*cs + lo; lens[cnt] = hi - lo; cnt++; }
        }
        for (int r = cnt; r < 16; ++r) { runs[r] = nullptr; lens[r] = 0; }
        if (cnt == 1) std::memcpy(out_slab, runs[0], lens[0]*4);
        else merge16_threshold_fast(out_slab, runs, lens, sz, FIXED_BUF_SIZE);
    });
#else
    std::vector<std::thread> workers;
    for (int s = 0; s < nslabs; ++s) {
        int sz = (int)(slab_off[s+1] - slab_off[s]);
        if (sz <= 0) continue;
        int* out_slab = out_block + slab_off[s];
        workers.emplace_back([=, &pieces]() {
            const int* runs[16]; int lens[16]; int cnt = 0;
            for (int r = 0; r < 16; ++r) {
                int lo = pieces[s][r].first, hi = pieces[s][r].second;
                if (hi > lo) { runs[cnt] = in_block + r*cs + lo; lens[cnt] = hi - lo; cnt++; }
            }
            for (int r = cnt; r < 16; ++r) { runs[r] = nullptr; lens[r] = 0; }
            if (cnt == 1) std::memcpy(out_slab, runs[0], lens[0]*4);
            else merge16_threshold_fast(out_slab, runs, lens, sz, FIXED_BUF_SIZE);
        });
    }
    for (auto& w : workers) w.join();
#endif
}
bool has_int_max_or_min(const int* a, int n, int) {
    __m512i vmax = _mm512_set1_epi32(INT_MAX);
    __m512i vmin = _mm512_set1_epi32(INT_MIN);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i v0 = _mm512_loadu_si512((__m512i*)(a + i));
        __m512i v1 = _mm512_loadu_si512((__m512i*)(a + i + 16));
        __m512i v2 = _mm512_loadu_si512((__m512i*)(a + i + 32));
        __m512i v3 = _mm512_loadu_si512((__m512i*)(a + i + 48));
        __mmask16 m0 = _mm512_cmpeq_epi32_mask(v0, vmax) | _mm512_cmpeq_epi32_mask(v0, vmin);
        __mmask16 m1 = _mm512_cmpeq_epi32_mask(v1, vmax) | _mm512_cmpeq_epi32_mask(v1, vmin);
        __mmask16 m2 = _mm512_cmpeq_epi32_mask(v2, vmax) | _mm512_cmpeq_epi32_mask(v2, vmin);
        __mmask16 m3 = _mm512_cmpeq_epi32_mask(v3, vmax) | _mm512_cmpeq_epi32_mask(v3, vmin);
        if (m0 || m1 || m2 || m3) return true;
    }
    for (; i + 16 <= n; i += 16) {
        __m512i v = _mm512_loadu_si512((__m512i*)(a + i));
        if (_mm512_cmpeq_epi32_mask(v, vmax) || _mm512_cmpeq_epi32_mask(v, vmin)) return true;
    }
    for (; i < n; ++i) if (a[i] == INT_MAX || a[i] == INT_MIN) return true;
    return false;
}

// ============================================================================
// ADAPTIVITY
// ============================================================================
inline int compute_adaptive_th(int n, int num_threads) {
    const char* env = std::getenv("WPMSP_FORCE_TH");
    if (env) return std::atoi(env);

    int per_thread = (n + num_threads - 1) / num_threads;

    // Ступенчатая функция, найденная эмпирически на Zen 5:
    if (per_thread <= 100000) return 100000; // Весь чанк уходит в qi
    if (per_thread <= 250000) return 96000;  // Оптимум для 3.2M (L2 cache)
    if (per_thread <= 4000000) return 24000; // Оптимум для 16M-64M
    if (per_thread <= 8000000) return 48000; // Оптимум для 128M
    return 96000;                            // Оптимум для 256M+ (магия числа 166)
}
inline int pick_K_eff(int n, int K_max) {
    if (K_max <= 2) return (n < 1000000) ? 1 : K_max;
    if (n <   150000) return std::min(K_max,  1);
    if (n <   600000) return std::min(K_max, 10);
    if (n <  1200000) return std::min(K_max, 12);
    if (n <  1800000) return std::min(K_max, 16);
    if (n <  2500000) return std::min(K_max, 24);
    return K_max;
}

// ============================================================================
// PARALLEL SORT — adaptive over K and TH
// ============================================================================
// num_threads is the caller's CAP. The function re-clamps it against actual
// hardware cores (detect_num_cores) — so calling with num_threads=32 on an
// 8-core box will use at most 8.
void parallel_16wpmsp_sort(int* a, int* scratch, int n, int num_threads, double& p1_ms, double& p2_ms) {
    p1_ms = 0; p2_ms = 0;
    if (n < 50000) { std::sort(a, a+n); return; }
    if (has_int_max_or_min(a, n, num_threads)) { std::sort(a, a+n); return; }

    // Defense-in-depth: never exceed real hardware, even if caller passed more.
    int cap = (num_threads <= 0) ? detect_num_cores() : std::min(num_threads, detect_num_cores());
    if (cap < 1) cap = 1;

    int K_eff = pick_K_eff(n, cap);
    int th    = compute_adaptive_th(n, K_eff);

    if (K_eff == 1) {
        auto t0 = Clock::now();
        sort_local_16wpmsp(a, scratch, n, th);
        auto t1 = Clock::now();
        p1_ms = duration<double, std::milli>(t1-t0).count();
        return;
    }

    int cs = (n + K_eff - 1) / K_eff;

    auto t0 = Clock::now();
#ifdef USE_THREAD_POOL
    wpmsp::ThreadPool::instance().parallel_for(0, K_eff, [&](int t) {
        int off = t * cs, len = std::min(cs, n - off);
        if (len > 0) sort_local_16wpmsp(a + off, scratch + off, len, th);
    });
#else
    {
        std::vector<std::thread> workers;
        for (int t = 0; t < K_eff; ++t) {
            int off = t * cs, len = std::min(cs, n - off);
            if (len <= 0) break;
            workers.emplace_back([a, scratch, off, len, th]() {
                sort_local_16wpmsp(a + off, scratch + off, len, th);
            });
        }
        for (auto& w : workers) w.join();
    }
#endif
    auto t1 = Clock::now();
    p1_ms = duration<double, std::milli>(t1-t0).count();

    auto t2 = Clock::now();

    if (K_eff == 2) {
        if (cs < n) {
            par_merge(a, cs, a + cs, n - cs, scratch, K_eff);
            std::memcpy(a, scratch, n * sizeof(int));
        }
        auto t3 = Clock::now();
        p2_ms = duration<double, std::milli>(t3-t2).count();
        return;
    }

    int nslabs = std::min(16, cap);
    int group_size = 16 * cs;
    int G = (n + group_size - 1) / group_size;

    if (G == 1) {
        // Fast path: single group, no outer spawn, use pool inside
        parallel_16way_slab(a, scratch, n, cs, nslabs);
    } else {
        std::vector<std::thread> workers;
        for (int g = 0; g < G; ++g) {
            int start = g * group_size;
            int end = std::min(start + group_size, n);
            if (start >= n) break;
            int total_g = end - start;
            workers.emplace_back([=]() {
                parallel_16way_slab(a + start, scratch + start, total_g, cs, nslabs);
            });
        }
        for (auto& w : workers) w.join();
    }

    int* src = scratch; int* dst = a;
    int stride = group_size; int cur_G = G;
    while (cur_G > 1) {
        for (int g = 0; g < cur_G; g += 2) {
            int lo = g * stride;
            int mid = std::min((g + 1) * stride, n);
            int hi = std::min((g + 2) * stride, n);
            if (lo >= n) break;
            if (mid >= hi) std::memcpy(dst + lo, src + lo, (mid - lo) * sizeof(int));
            else par_merge(src + lo, mid - lo, src + mid, hi - mid, dst + lo, cap);
        }
        std::swap(src, dst);
        cur_G = (cur_G + 1) / 2;
        stride *= 2;
    }
    if (src != a) std::memcpy(a, src, n * sizeof(int));

    auto t3 = Clock::now();
    p2_ms = duration<double, std::milli>(t3-t2).count();
}

// Legacy alias — keeps existing code calling parallel_ptms_sort working.
#define parallel_ptms_sort parallel_16wpmsp_sort

// ============================================================================
