#pragma once
#include <vector>
#include <format>
#include "memory_pool_abstract.h"
#include <utility>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <list>

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(size, alignment) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#include <cstdlib>
#define ALIGNED_ALLOC(size, alignment) std::aligned_alloc(alignment, size)
#define ALIGNED_FREE(ptr) std::free(ptr)
#endif

class MemoryPoolCPU final : public MemoryPoolAbstract {
public:

    class SizeClass {
    public:
        struct Span{SizeClass* bin_class_ptr{};};
    private:
        struct FreeBlock{FreeBlock* m_next_block_ptr{};};
        std::size_t m_class_size {}, m_total_blocks{}, m_free_blocks{};
        FreeBlock* m_freeList_head{};
        std::vector<Span*> m_spans;
    public:
        static constexpr std::size_t ALIGNMENT {64};
        explicit SizeClass(std::size_t class_size, std::size_t pages);
        SizeClass(const SizeClass&) = delete;
        SizeClass& operator=(const SizeClass&) = delete;
        void add_blocks(std::size_t pages);
        void returnBlock(void* ptr);
        void* getBlock();
        [[nodiscard]] std::size_t size() const {return m_class_size;}
        [[nodiscard]] std::size_t total_blocks() const {return m_total_blocks;}
        [[nodiscard]] std::size_t free_blocks() const {return m_free_blocks;}
        ~SizeClass();
    };

    std::list<SizeClass> m_sizes;
    std::size_t m_dynamic_size_class_threshold {5*1024*1024};
    static constexpr std::size_t MIN_BIN_SIZE {SizeClass::ALIGNMENT}, PAGE_SIZE{8*1024};
    static std::vector<std::size_t> DEFAULT_BINS, DEFAULT_PAGES;
public:
    explicit MemoryPoolCPU(std::vector<std::size_t>& bins=DEFAULT_BINS,
        std::vector<std::size_t>& pages_count=DEFAULT_PAGES);
    MemoryPoolCPU(const MemoryPoolCPU&) = delete;
    MemoryPoolCPU& operator=(const MemoryPoolCPU&) = delete;
    void* getMem(std::size_t size) override;
    void returnMem(void*) override;
};

std::vector<std::size_t> MemoryPoolCPU::DEFAULT_BINS = {64, 128, 256, 512, 1024, 2048, 4096, 8192},
                               MemoryPoolCPU::DEFAULT_PAGES = {2, 2, 2, 2, 2, 4, 6, 6};

inline MemoryPoolCPU::MemoryPoolCPU(std::vector<std::size_t>& bins, std::vector<std::size_t>& pages_count) {
    if (pages_count.size() != bins.size()) throw std::invalid_argument("Number of pages does not match number of bins");

    auto powerOfTwo{
        [](const std::size_t num) -> bool {
            return (num>0) && ((num & (num-1))==0);
        }
    };

    if (!(std::ranges::all_of(bins.begin(), bins.end(), powerOfTwo))) throw
    std::invalid_argument(
        std::format("Invalid Bin Size(s) Found, Bin Size Must Be A Power Of {}", SizeClass::ALIGNMENT));

    std::ranges::sort(bins);
    if (bins[0]==0) throw std::invalid_argument("Bin Size Cant Be Zero");
    else if (bins[0]!=MIN_BIN_SIZE) throw
        std::invalid_argument(std::format("Minimum Bin Size is {}", MIN_BIN_SIZE));

    for (auto it1 = bins.begin(), it2 = pages_count.begin();
        it1!=bins.end() && it2!=pages_count.end(); ++it1, ++it2) m_sizes.emplace_back(*it1, *it2);
}

inline MemoryPoolCPU::SizeClass::SizeClass(std::size_t class_size, std::size_t pages)
        : m_class_size {class_size}{
    add_blocks(pages);
}

inline void* MemoryPoolCPU::getMem(std::size_t size) {
    for (auto& bin : m_sizes) {
        if (bin.size()>=size) {
            if (auto* block_ptr{bin.getBlock()}; block_ptr) return block_ptr;
            else {bin.add_blocks(1); return bin.getBlock();}
        }
    }
    std::size_t rounded_size {std::bit_ceil(size)};
    m_sizes.emplace_back(rounded_size, size<=m_dynamic_size_class_threshold?2:1);
    return m_sizes.back().getBlock();
}

inline void MemoryPoolCPU::returnMem(void* ptr) {
    constexpr std::uintptr_t SIZE {PAGE_SIZE}, MASK {~(SIZE-1)};
    const auto* span_ptr {reinterpret_cast<SizeClass::Span*>(reinterpret_cast<std::uintptr_t>(ptr) & MASK)};
    span_ptr->bin_class_ptr->returnBlock(ptr);
}

inline void MemoryPoolCPU::SizeClass::add_blocks(std::size_t pages) {
    for (std::size_t page{}; page<pages; ++page) {

        auto* mem {static_cast<byte*>(ALIGNED_ALLOC(PAGE_SIZE, PAGE_SIZE))};
        if (!mem) throw std::bad_alloc();
        auto* span {reinterpret_cast<Span*>(mem)};
        span->bin_class_ptr = this;
        m_spans.push_back(span);

        constexpr std::size_t usable_space {PAGE_SIZE - sizeof(Span)};
        const std::size_t blocks {static_cast<std::size_t>(usable_space / m_class_size)};

        byte* current_block_ptr = mem + sizeof(Span);

        for (std::size_t i {}; i < blocks; i++) {
            auto* block {reinterpret_cast<FreeBlock*>(current_block_ptr)};
            block->m_next_block_ptr = m_freeList_head;
            m_freeList_head = block;
            current_block_ptr += m_class_size;
        }

        m_total_blocks += blocks;
        m_free_blocks += blocks;
    }
}

inline void* MemoryPoolCPU::SizeClass::getBlock() {
    std::cout<<"Allocating From "<<m_class_size<<"\n";
    if (!m_freeList_head) return nullptr;
    auto* block {m_freeList_head};
    m_freeList_head = block->m_next_block_ptr;
    auto* block_to_return {reinterpret_cast<void*>(block)};
    m_free_blocks--;
    return block_to_return;
}

inline void MemoryPoolCPU::SizeClass::returnBlock(void* ptr) {
    std::cout<<"Deallocating To "<<m_class_size<<"\n";
    auto* returned_block {static_cast<FreeBlock*>(ptr)};
    returned_block->m_next_block_ptr = m_freeList_head;
    m_freeList_head = returned_block;
    m_free_blocks++;
}

inline MemoryPoolCPU::SizeClass::~SizeClass() {
    for (const auto& block_ptr : MemoryPoolCPU::SizeClass::m_spans) ALIGNED_FREE(block_ptr);
}