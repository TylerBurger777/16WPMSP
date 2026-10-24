// Single-algorithm benchmark. Compile 4 binaries with different -DALGO_*.
// Each binary runs ONLY ONE algorithm in ONE process, guaranteeing isolation.
#include "16wpmsp/16wpmsp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <execution>

#include "ips4o.hpp"
#include "hwy/contrib/sort/vqsort.h"

#if defined(__linux__)
#  include <sched.h>
#  include <unistd.h>
#endif

using Clock = std::chrono::high_resolution_clock;
using std::chrono::duration;

struct Stats {
    double min=0, p25=0, med=0, p75=0, p95=0, max=0, mean=0, stddev_pct=0;
    bool ok=false;
};

static Stats calc_stats(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    int n = (int)v.size();
    auto q = [&](double p) {
        double idx = p*(n-1);
        int lo=(int)std::floor(idx), hi=(int)std::ceil(idx);
        return (lo==hi) ? v[lo] : v[lo]+(v[hi]-v[lo])*(idx-lo);
    };
    s.min=v.front(); s.max=v.back();
    s.p25=q(0.25); s.med=q(0.50); s.p75=q(0.75); s.p95=q(0.95);
    double sum=0; for (double x:v) sum+=x;
    s.mean=sum/n;
    double sq=0; for (double x:v) sq+=(x-s.mean)*(x-s.mean);
    s.stddev_pct = (s.mean>0) ? std::sqrt(sq/n)/s.mean*100.0 : 0.0;
    s.ok=true;
    return s;
}

int main(int argc, char** argv) {
    int warmup=5, measure=21; bool pin=false;
    std::vector<int> Ns;
    for (int i=1; i<argc; ++i) {
        std::string s=argv[i];
        if (s=="--quick") { warmup=2; measure=7; }
        else if (s=="--pin") pin=true;
        else Ns.push_back(std::atoi(argv[i]));
    }
    if (Ns.empty()) Ns = {100000, 250000, 500000, 1000000, 2000000,
                          4000000, 8000000, 16000000, 32000000, 64000000};
    std::sort(Ns.begin(), Ns.end());
    Ns.erase(std::unique(Ns.begin(), Ns.end()), Ns.end());

    if (pin) {
#if defined(__linux__)
        cpu_set_t set; CPU_ZERO(&set);
        int nc = (int)sysconf(_SC_NPROCESSORS_ONLN);
        for (int c=0; c<nc; ++c) CPU_SET(c, &set);
        sched_setaffinity(0, sizeof(set), &set);
#endif
    }

#if defined(ALGO_16WPMSP)
    const char* name = "16WPMSP";
#elif defined(ALGO_IPS4O)
    const char* name = "IPS4o";
#elif defined(ALGO_VQSORT)
    const char* name = "vqsort";
#elif defined(ALGO_TBB)
    const char* name = "std::par";
#else
#  error "define one of ALGO_16WPMSP / ALGO_IPS4O / ALGO_VQSORT / ALGO_TBB"
#endif

    int maxN = *std::max_element(Ns.begin(), Ns.end());
    std::vector<int> orig(maxN), a(maxN), sc(maxN);
    double p1, p2;

    auto sort_fn = [&](int* p, int n) {
#if defined(ALGO_16WPMSP)
        parallel_16wpmsp_sort(p, sc.data(), n, 0, p1, p2);
#elif defined(ALGO_IPS4O)
        ips4o::parallel::sort(p, p + n);
#elif defined(ALGO_VQSORT)
        hwy::VQSort(p, (size_t)n, hwy::SortAscending());
#elif defined(ALGO_TBB)
        std::sort(std::execution::par, p, p + n);
#endif
    };

    std::printf("ALGO=%s warmup=%d measure=%d\n", name, warmup, measure);
    std::printf("%-12s %12s %12s %12s %12s %12s %8s\n",
                "N", "min", "median", "p75", "p95", "max", "stddev%");
    std::printf("--------------------------------------------------------------------------\n");

    for (int N : Ns) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> d(-1000000000, 1000000000);
        for (int i=0; i<N; ++i) orig[i] = d(rng);

        for (int r=0; r<warmup; ++r) {
            std::memcpy(a.data(), orig.data(), (size_t)N*4);
            sort_fn(a.data(), N);
        }
        std::vector<double> t; t.reserve(measure);
        for (int r=0; r<measure; ++r) {
            std::memcpy(a.data(), orig.data(), (size_t)N*4);
            auto t0=Clock::now(); sort_fn(a.data(), N); auto t1=Clock::now();
            t.push_back(duration<double,std::milli>(t1-t0).count());
            if (!std::is_sorted(a.begin(), a.begin()+N)) { t.clear(); break; }
        }
        Stats s = calc_stats(t);
        if (!s.ok) {
            std::printf("%-12d FAIL\n", N);
        } else {
            std::printf("%-12d %12.3f %12.3f %12.3f %12.3f %12.3f %7.1f%%\n",
                        N, s.min, s.med, s.p75, s.p95, s.max, s.stddev_pct);
        }
        std::fflush(stdout);
    }
    return 0;
}
