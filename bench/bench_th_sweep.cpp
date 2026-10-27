#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include "16wpmsp/16wpmsp.hpp"

using Clock = std::chrono::high_resolution_clock;
using std::chrono::duration;

int main() {
    std::vector<int> Ns = {1000000, 4000000, 16000000};
    std::vector<int> THs = {4000, 8000, 12000, 16000, 24000, 32000, 40000, 50000, 60000, 80000, 100000};
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1000000000, 1000000000);

    for (int N : Ns) {
        std::vector<int> orig(N), a(N), buf(N);
        for (int i = 0; i < N; ++i) orig[i] = dist(rng);

        std::printf("\n=== N = %d (per_thread = %d) ===\n", N, N / 16);
        std::printf("%-10s | %-12s | %-12s\n", "TH", "Time (ms)", "Runs/thread");
        std::printf("----------------------------------------\n");

        for (int th : THs) {
            double min_ms = 1e9;
            // 3 warmup, 7 measure
            for (int r = 0; r < 3; ++r) {
                std::copy(orig.begin(), orig.end(), a.begin());
                sort_local_16wpmsp(a.data(), buf.data(), N, th);
            }
            for (int r = 0; r < 7; ++r) {
                std::copy(orig.begin(), orig.end(), a.begin());
                auto t0 = Clock::now();
                sort_local_16wpmsp(a.data(), buf.data(), N, th);
                auto t1 = Clock::now();
                double ms = duration<double, std::milli>(t1 - t0).count();
                if (ms < min_ms) min_ms = ms;
            }
            int runs = (N / 16 + th - 1) / th; // сколько runs в одном потоке
            std::printf("%-10d | %-12.3f | %-12d\n", th, min_ms, runs);
            std::fflush(stdout);
        }
    }
    return 0;
}
