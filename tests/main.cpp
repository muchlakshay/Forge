#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>
#include "../core/tensor/Storage/memory_pool_cpu.h"

using Clock = std::chrono::high_resolution_clock;

void test_large_allocations(Forge::MemoryPoolCPU& pool) {
    const size_t size = 512ULL * 1024 * 1024;
    const int iterations = 10;

    std::cout << "\n=== Large Allocation Test (512MB) ===\n";

    for (int i = 0; i < iterations; ++i) {
        auto t1 = Clock::now();
        void* p1 = pool.allocate(size);
        auto t2 = Clock::now();
        pool.free(p1);
        auto t3 = Clock::now();

        void* p2;
        auto t4 = Clock::now();
        p2 = std::malloc(size);
        auto t5 = Clock::now();
        std::free(p2);
        auto t6 = Clock::now();

        auto poolAlloc = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        auto poolFree  = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        auto osAlloc   = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
        auto osFree    = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();

        std::cout << "Iter " << i+1
                  << " | Pool alloc: " << poolAlloc << "us"
                  << " free: " << poolFree << "us"
                  << " | OS alloc: " << osAlloc << "us"
                  << " free: " << osFree << "us\n";
    }
}

void test_fragmentation(Forge::MemoryPoolCPU& pool) {
    std::cout << "\n=== Fragmentation Test (Random Sizes) ===\n";

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(8, 64 * 1024); // 8B – 64KB

    const int ops = 200000;
    std::vector<void*> ptrs;
    ptrs.reserve(ops);

    auto t1 = Clock::now();
    for (int i = 0; i < ops; ++i) {
        size_t sz = dist(rng);
        ptrs.push_back(pool.allocate(sz));
    }
    auto t2 = Clock::now();

    std::shuffle(ptrs.begin(), ptrs.end(), rng);

    for (void* p : ptrs)
        pool.free(p);
    auto t3 = Clock::now();

    auto allocTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto freeTime  = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

    std::cout << "Pool random alloc time: " << allocTime << " ms\n";
    std::cout << "Pool random free  time: " << freeTime  << " ms\n";
}

void test_mixed_lifetimes(Forge::MemoryPoolCPU& pool) {
    std::cout << "\n=== Mixed Lifetime Test ===\n";

    std::mt19937 rng(123);
    std::uniform_int_distribution<size_t> sizeDist(16, 4096);
    std::bernoulli_distribution freeChance(0.5);

    const int ops = 300000;
    std::vector<void*> live;

    auto t1 = Clock::now();

    for (int i = 0; i < ops; ++i) {
        if (!live.empty() && freeChance(rng)) {
            size_t idx = rng() % live.size();
            pool.free(live[idx]);
            live[idx] = live.back();
            live.pop_back();
        } else {
            live.push_back(pool.allocate(sizeDist(rng)));
        }
    }

    for (void* p : live)
        pool.free(p);

    auto t2 = Clock::now();

    auto total = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "Mixed workload total time: " << total << " ms\n";
}

int main() {
    Forge::MemoryPoolCPU pool;
    test_large_allocations(pool);
    test_fragmentation(pool);
    test_mixed_lifetimes(pool);
    test_fragmentation(pool);
    pool.hard_clear_cache();

    return 0;
}
