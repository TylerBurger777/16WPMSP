#include "16wpmsp/16wpmsp.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

static int fails = 0;
static void chk(bool c, const char* n, int N) {
    std::printf(c ? "[ OK ] %-28s N=%d\n" : "[FAIL] %-28s N=%d\n", n, N);
    if (!c) fails++;
}
static void run(const char* n, std::vector<int>& v) {
    int N = (int)v.size();
    std::vector<int> a = v, s(N);
    double p1, p2;
    parallel_16wpmsp_sort(a.data(), s.data(), N, 0, p1, p2);
    chk(std::is_sorted(a.begin(), a.end()), n, N);
}

int main() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> d(-1000000000, 1000000000);
    std::printf("16WPMSP stress, cores=%d\n\n", detect_num_cores());

    { std::vector<int> v(4000000); for (auto& x : v) x = d(rng); run("random 4M", v); }
    { std::vector<int> v(1000000, 7); run("all equal 1M", v); }
    { std::vector<int> v(1000000); for (int i=0;i<(int)v.size();++i) v[i]=(int)v.size()-i; run("reverse 1M", v); }
    { std::vector<int> v(1000000); for (int i=0;i<(int)v.size();++i) v[i]=i; run("sorted 1M", v); }
    { std::vector<int> v(1000000); for (auto& x : v) x = (int)(rng() & 1); run("binary 1M", v); }
    { std::vector<int> v(200000); for (auto& x : v) x = d(rng); v[12345]=INT_MAX; run("with INT_MAX", v); }
    { std::vector<int> v(200000); for (auto& x : v) x = d(rng); v[54321]=INT_MIN; run("with INT_MIN", v); }
    { std::vector<int> v(1000000); for (auto& x : v) x = d(rng) % 100; run("100 distinct values", v); }
    { std::vector<int> v(1000000); for (int i=0;i<(int)v.size();++i) v[i]=i;
      for (int k=0;k<10000;++k){int i=(int)(rng()%v.size());int j=(int)(rng()%v.size());std::swap(v[i],v[j]);}
      run("nearly sorted 1M", v); }
    { std::vector<int> v(1000); for (auto& x : v) x = d(rng); run("N=1000 fallback", v); }
    { std::vector<int> v(50000); for (auto& x : v) x = d(rng); run("N=50000 boundary", v); }
    { std::vector<int> v(50001); for (auto& x : v) x = d(rng); run("N=50001", v); }
    { std::vector<int> v(4000001); for (auto& x : v) x = d(rng); run("N=4000001 odd", v); }

    std::printf("\n%s (%d failures)\n", fails == 0 ? "ALL OK" : "FAILED", fails);
    return fails == 0 ? 0 : 1;
}
