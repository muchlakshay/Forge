#pragma once
#include <vector>
#include <format>
#include "../memory_pool_abstract.h"
#include <utility>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <list>
#include <memory>
#include <cassert>

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(size, alignment) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#include <cstdlib>
#define ALIGNED_ALLOC(size, alignment) std::aligned_alloc(alignment, size)
#define ALIGNED_FREE(ptr) std::free(ptr)
#endif

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
    #include <xmmintrin.h>
    #define PREFETCH(addr, rw, locality) _mm_prefetch((const char*)addr, (locality > 0) ? _MM_HINT_T0 : _MM_HINT_NTA)
#elif defined(__GNUC__) || defined(__clang__)
    #define PREFETCH(addr, rw, locality) __builtin_prefetch(addr, rw, locality)
#else
    #define PREFETCH(addr, rw, locality) do { (void)addr; (void)rw; (void)locality; } while (0)
#endif

class Forge::MemoryPoolCPU : public Forge::MemoryPoolAbstract {

    static constexpr std::size_t INDEXING_BITS {9};
    struct RadixNode {void* entries[1<<INDEXING_BITS]{};};

    class Bin {
        struct freeBlock {freeBlock* next_block{};};
        struct Span {std::size_t m_blocks_used{}; Bin* bin_ptr{};};
        std::vector<byte*> m_mem_allocated;
        std::vector<Span> m_spans_allocated;
        freeBlock* m_freeListHead {};
        std::size_t m_class_size{}, m_totalBlocks {}, m_freeBlocks{}, m_chunks_allocated{};

        void free_cache();

        friend MemoryPoolCPU;
    public:
        static constexpr std::size_t CHUNK_SIZE {64*1024}, ALIGNMENT{64};
        explicit Bin(std::size_t class_size, std::size_t chunks);
        Bin(const Bin&) = delete;
        Bin& operator=(const Bin&) = delete;
        void extendBlocks(std::size_t chunks);
        void* allocate();
        void free(void* ptr);
        ~Bin();
    };

    class RadixTree {
        RadixNode* m_radix_root {};

        static void destroy_node(RadixNode* node, int level);
    public:
        void radix_insert(void* ptr, Forge::MemoryPoolCPU::Bin* bin_ptr);
        Bin* radix_lookup(void* ptr) const;
        void clear() {destroy_node(m_radix_root, 0); m_radix_root = nullptr;}
        ~RadixTree(){destroy_node(m_radix_root, 0);};
    };

    RadixTree m_radix_tree;
    std::unique_ptr<Bin> m_bins [64] {};

public:
    MemoryPoolCPU() = default;
    MemoryPoolCPU(const MemoryPoolCPU&) = delete;
    MemoryPoolCPU& operator=(const MemoryPoolCPU&) = delete;
    void* allocate(std::size_t size) override;
    void free(void* ptr) override;
    void hard_clear_cache();
};
