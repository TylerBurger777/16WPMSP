#pragma once
// ThreadPool V2: main thread coordinates only (does NOT work).
// All tasks distributed across nthreads_ persistent workers.
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <cstdio>

#if defined(__linux__)
#  include <sched.h>
#  include <unistd.h>
#endif

namespace wpmsp {

inline int detect_num_cores() {
#if defined(__linux__)
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int n = CPU_COUNT(&set);
        if (n > 0) return n;
    }
    {
        long sc = sysconf(_SC_NPROCESSORS_ONLN);
        if (sc > 0) return (int)sc;
    }
#endif
    unsigned hc = std::thread::hardware_concurrency();
    if (hc > 0) return (int)hc;
    return 1;
}

class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    int size() const { return nthreads_; }

    void parallel_for(int begin, int end, std::function<void(int)> body) {
        int count = end - begin;
        if (count <= 0) return;
        if (count == 1) { body(begin); return; }
        if (nthreads_ <= 0) {
            for (int i = begin; i < end; ++i) body(i);
            return;
        }

        int total = nthreads_;

        {
            std::lock_guard<std::mutex> lock(mtx_);
            job_ = [&, begin, count, total](int wid) {
                int lo = begin + (int)((long long)wid * count / total);
                int hi = begin + (int)((long long)(wid + 1) * count / total);
                for (int i = lo; i < hi; ++i) body(i);
            };
            done_count_ = 0;
            ++generation_;
        }
        cv_start_.notify_all();

        // Main thread waits (does NOT work)
        std::unique_lock<std::mutex> lock(mtx_);
        cv_done_.wait(lock, [&] { return done_count_ == nthreads_; });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stop_.store(true);
        }
        cv_start_.notify_all();
        for (auto& t : workers_) if (t.joinable()) t.join();
    }

private:
    ThreadPool() {
        nthreads_ = detect_num_cores();
        workers_.reserve(nthreads_);
        for (int i = 0; i < nthreads_; ++i) {
            workers_.emplace_back([this, i] { worker_loop(i); });
        }
    }

    void worker_loop(int wid) {
        int my_gen = 0;
        while (true) {
            std::function<void(int)> job;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_start_.wait(lock, [&] { return generation_ != my_gen || stop_.load(); });
                if (stop_.load()) return;
                my_gen = generation_;
                job = job_;
            }
            job(wid);
            {
                std::lock_guard<std::mutex> lock(mtx_);
                if (++done_count_ == nthreads_) cv_done_.notify_one();
            }
        }
    }

    int nthreads_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mtx_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    std::function<void(int)> job_;
    int generation_ = 0;
    int done_count_ = 0;
    std::atomic<bool> stop_{false};
};

} // namespace wpmsp
