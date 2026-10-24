// 16WPMSP benchmark: isolated passes per algorithm (avoids thread-pool contention).
// warmup=5, measure=21, full stats (min/p25/median/p75/p95/max/stddev).
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

template <class F>
static Stats run_stats(int N, int warmup, int measure,
                       std::vector<int>& orig, std::vector<int>& a, F&& fn)
{
    for (int r=0; r<warmup; ++r) {
        std::memcpy(a.data(), orig.data(), (size_t)N*4);
        fn(a.data(), N);
    }
    std::vector<double> t; t.reserve(measure);
    for (int r=0; r<measure; ++r) {
        std::memcpy(a.data(), orig.data(), (size_t)N*4);
        auto t0=Clock::now(); fn(a.data(), N); auto t1=Clock::now();
        t.push_back(duration<double,std::milli>(t1-t0).count());
        if (!std::is_sorted(a.begin(), a.begin()+N)) return {};
    }
    return calc_stats(t);
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
        std::printf("[pin] affinity -> all %d CPUs\n", nc);
#endif
    }

    int maxN = *std::max_element(Ns.begin(), Ns.end());
    std::vector<int> orig(maxN), a(maxN), sc(maxN);

    std::printf("16WPMSP benchmark (isolated passes)\n");
    std::printf("Cores=%d K_max=%d warmup=%d measure=%d\n\n",
                detect_num_cores(), resolve_K_max(), warmup, measure);

    // Pre-generate all datasets once (same seed → identical data for every algorithm)
    std::vector<std::vector<int>> data(Ns.size());
    for (size_t i=0; i<Ns.size(); ++i) {
        int N = Ns[i];
        data[i].resize(N);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> d(-1000000000, 1000000000);
        for (int j=0; j<N; ++j) data[i][j] = d(rng);
    }

    // --- Pass 1: 16WPMSP ---
    std::vector<Stats> s16(Ns.size());
    std::printf("[pass 1/4] 16WPMSP ... "); std::fflush(stdout);
    for (size_t i=0; i<Ns.size(); ++i) {
        int N = Ns[i];
        std::memcpy(orig.data(), data[i].data(), (size_t)N*4);
        double p1, p2;
        s16[i] = run_stats(N, warmup, measure, orig, a,
            [&](int* p, int n){ parallel_16wpmsp_sort(p, sc.data(), n, 0, p1, p2); });
    }
    std::printf("done\n");

    // --- Pass 2: IPS4o ---
    std::vector<Stats> sips(Ns.size());
    std::printf("[pass 2/4] IPS4o ... "); std::fflush(stdout);
    for (size_t i=0; i<Ns.size(); ++i) {
        int N = Ns[i];
        std::memcpy(orig.data(), data[i].data(), (size_t)N*4);
        sips[i] = run_stats(N, warmup, measure, orig, a,
            [](int* p, int n){ ips4o::parallel::sort(p, p+n); });
    }
    std::printf("done\n");

    // --- Pass 3: vqsort ---
    std::vector<Stats> svq(Ns.size());
    std::printf("[pass 3/4] vqsort ... "); std::fflush(stdout);
    for (size_t i=0; i<Ns.size(); ++i) {
        int N = Ns[i];
        std::memcpy(orig.data(), data[i].data(), (size_t)N*4);
        svq[i] = run_stats(N, warmup, measure, orig, a,
            [](int* p, int n){ hwy::VQSort(p, (size_t)n, hwy::SortAscending()); });
    }
    std::printf("done\n");

    // --- Pass 4: std::par ---
    std::vector<Stats> spar(Ns.size());
    std::printf("[pass 4/4] std::par ... "); std::fflush(stdout);
    for (size_t i=0; i<Ns.size(); ++i) {
        int N = Ns[i];
        std::memcpy(orig.data(), data[i].data(), (size_t)N*4);
        spar[i] = run_stats(N, warmup, measure, orig, a,
            [](int* p, int n){ std::sort(std::execution::par, p, p+n); });
    }
    std::printf("done\n\n");

    // --- Main table (median) ---
    std::printf("%-12s %18s %14s %14s %14s\n",
                "N", "16WPMSP", "IPS4o", "vqsort", "std::par");
    std::printf("---------------------------------------------------------------------------------------\n");
    auto fmt = [](const Stats& s) -> std::string {
        if (!s.ok) return "FAIL";
        char b[24]; std::snprintf(b, sizeof(b), "%.3f", s.med); return b;
    };
    for (size_t i=0; i<Ns.size(); ++i) {
        int K = pick_K_eff(Ns[i], resolve_K_max());
        std::printf("%-12d %10s(K=%-2d) %14s %14s %14s\n",
                    Ns[i], fmt(s16[i]).c_str(), K,
                    fmt(sips[i]).c_str(), fmt(svq[i]).c_str(), fmt(spar[i]).c_str());
    }

    // --- Stability ---
    std::printf("\n=== Stability (stddev %% of mean) ===\n");
    std::printf("%-12s %10s %10s %10s %10s\n", "N", "16WPMSP", "IPS4o", "vqsort", "std::par");
    std::printf("-----------------------------------------------------------\n");
    auto fs = [](const Stats& s) { char b[16]; std::snprintf(b, sizeof(b), "%.1f%%", s.stddev_pct); return std::string(b); };
    for (size_t i=0; i<Ns.size(); ++i) {
        std::printf("%-12d %10s %10s %10s %10s\n", Ns[i],
                    s16[i].ok ? fs(s16[i]).c_str() : "FAIL",
                    sips[i].ok ? fs(sips[i]).c_str() : "FAIL",
                    svq[i].ok ? fs(svq[i]).c_str() : "FAIL",
                    spar[i].ok ? fs(spar[i]).c_str() : "FAIL");
    }

    // --- 16WPMSP detail ---
    std::printf("\n=== 16WPMSP detail (min/p25/med/p75/p95/max, ms) ===\n");
    std::printf("%-12s %8s %8s %8s %8s %8s %8s\n", "N","min","p25","median","p75","p95","max");
    std::printf("---------------------------------------------------------------\n");
    for (size_t i=0; i<Ns.size(); ++i) {
        if (!s16[i].ok) { std::printf("%-12d FAIL\n", Ns[i]); continue; }
        std::printf("%-12d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n",
                    Ns[i], s16[i].min, s16[i].p25, s16[i].med,
                    s16[i].p75, s16[i].p95, s16[i].max);
    }
    return 0;
}
